import numpy as np
from pathlib import Path
from astropy.io import fits
import time
from multiprocessing import Pool, cpu_count
import functools


def extract_features(wav, flux):
    flux_norm = flux / np.max(flux)
    
    features = {}
    
    idx_4200 = np.argmin(np.abs(wav - 4200))
    idx_5500 = np.argmin(np.abs(wav - 5500))
    idx_7000 = np.argmin(np.abs(wav - 7000))
    idx_8500 = np.argmin(np.abs(wav - 8500))
    
    features['ci_4200_7000'] = flux_norm[idx_4200] / flux_norm[idx_7000]
    features['ci_5500_8500'] = flux_norm[idx_5500] / flux_norm[idx_8500]
    
    def line_depth(wav_center, width=50):
        idx = np.argmin(np.abs(wav - wav_center))
        linea = flux_norm[idx]
        cont = np.median(flux_norm[max(0, idx-width):min(len(flux_norm), idx+width)])
        return 1.0 - (linea / cont) if cont > 0 else 0
    
    features['caII_K'] = line_depth(3934, width=30)
    features['caII_H'] = line_depth(3968, width=30)
    features['Hbeta'] = line_depth(4861, width=50)
    features['Halpha'] = line_depth(6563, width=50)
    
    balmer_region = flux_norm[(wav > 3600) & (wav < 4200)]
    if len(balmer_region) > 0:
        features['balmer_jump'] = np.mean(balmer_region)
    else:
        features['balmer_jump'] = 0
    
    ir_region = flux_norm[(wav > 7000) & (wav < 8500)]
    features['ir_strength'] = np.mean(ir_region) if len(ir_region) > 0 else 0

    uv_region = flux_norm[(wav > 3800) & (wav < 4500)]
    opt_region = flux_norm[(wav > 5500) & (wav < 6500)]
    features['uv_opt_ratio'] = np.mean(uv_region) / np.mean(opt_region) if np.mean(opt_region) > 0 else 0
    
    return np.array([features[k] for k in sorted(features.keys())])


def _load_fits_file(ruta): #Carga y procesa FITS
    try:
        with fits.open(ruta) as hdul:
            hdr = hdul[1].header
            sptype = str(hdr.get("SPTYPE", "??"))
            real_type = sptype.split()[0]
            
            data = hdul[1].data
            wav = data["wavelength"][0]
            flux = data["spectrum"][0]
            flux = flux / np.max(flux)
            
            features = extract_features(wav, flux)
            return ruta, features, real_type
    except Exception as e:
        print(f"Error al cargar {ruta}: {e}")
        return None


def _compute_distances_row(args): #Calcula distancias de una fila del mapa
    row_idx, x, weights_row = args
    distances = np.array([np.sqrt(np.sum((x - w) ** 2)) for w in weights_row])
    return row_idx, distances


def _compute_bmu_batch(args): #Encuentra BMU para un lote de muestras
    x, weights, map_height, map_width = args
    distances = np.zeros((map_height, map_width))
    
    for i in range(map_height):
        for j in range(map_width):
            distances[i, j] = np.sqrt(np.sum((x - weights[i, j]) ** 2))
    
    bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
    return bmu_idx, np.min(distances)


class SimpleSOM:
    def __init__(self, map_width=13, map_height=13, feature_dim=9, learning_rate=0.5):
        self.map_width = map_width
        self.map_height = map_height
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        
        np.random.seed(42)
        self.weights = np.random.randn(map_height, map_width, feature_dim) * 0.5
        
        self.type_map = np.full((map_height, map_width), "?", dtype=object)
        self.count_map = np.zeros((map_height, map_width), dtype=int)
    
    def find_bmu(self, x): 
        num_cores = cpu_count()
        
        args_list = [(i, x, self.weights[i]) for i in range(self.map_height)]
        
        with Pool(num_cores) as pool:
            results = pool.map(_compute_distances_row, args_list)
        
        distances = np.zeros((self.map_height, self.map_width))
        for row_idx, row_distances in results:
            distances[row_idx] = row_distances
        
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx, np.min(distances)
    
    def gaussian_kernel(self, center, position, sigma): #Calcula distribucion de vecindad
        center = np.array(center)
        position = np.array(position)
        distance = np.sqrt(np.sum((center - position) ** 2))
        return np.exp(-(distance ** 2) / (2 * (sigma ** 2)))
    
    def train(self, data, epochs=100):
        n_samples = data.shape[0]
        num_cores = cpu_count()
        
        for epoch in range(epochs):
            sigma = 1.0 * np.exp(-epoch / (epochs / 3))
            learning_rate = self.learning_rate * np.exp(-epoch / epochs)
            
            # Procesar muestras en lotes con BMU paralelizado
            lote_size = max(1, n_samples // (num_cores * 2))
            sample_indices = np.random.permutation(n_samples)
            
            for batch_start in range(0, n_samples, lote_size):
                batch_end = min(batch_start + lote_size, n_samples)
                batch_indices = sample_indices[batch_start:batch_end]
                
                args_list = [(data[i], self.weights, self.map_height, self.map_width) 
                            for i in batch_indices]
                
                with Pool(num_cores) as pool:
                    bmu_results = pool.map(_compute_bmu_batch, args_list)
                
                # Actualizar pesos secuencialmente (evita race conditions)
                for sample_idx, (bmu_idx, _) in zip(batch_indices, bmu_results):
                    x = data[sample_idx]
                    for i in range(self.map_height):
                        for j in range(self.map_width):
                            influence = self.gaussian_kernel(bmu_idx, (i, j), sigma)
                            self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])
    
    def predict(self, x): 
        bmu_idx, _ = self.find_bmu(x)
        return self.type_map[bmu_idx]
    
    def assign_types(self, training_data, training_labels):
        self.type_map[:] = "?"
        self.count_map[:] = 0
        
        for x, label in zip(training_data, training_labels):
            bmu_idx, _ = self.find_bmu(x)
            if self.count_map[bmu_idx] == 0:
                self.type_map[bmu_idx] = label[0]
            self.count_map[bmu_idx] += 1
    


def cargar_datos_paralelo(rutas):
    num_cores = cpu_count()
    
    with Pool(num_cores) as pool:
        results = pool.map(_load_fits_file, rutas)
    
    # Filtrar resultados None (errores)
    results = [r for r in results if r is not None]
    
    if not results:
        return None, None, None
    
    rutas_validas, features_list, labels = zip(*results)
    return list(rutas_validas), np.array(features_list), list(labels)


def procesar_carpeta(carpeta, train_epochs=200):
    rutas = sorted(Path(carpeta).glob("*.fits"))
    
    if len(rutas) == 0:
        print(f"No FITS files found in {carpeta}")
        return
    
    # Carga dats
    print("=" * 80)
    print("FASE 1: Extrayendo características espectrales (paralelizado)...")
    print("=" * 80)
    
    rutas_validas, training_data, training_labels = cargar_datos_paralelo(rutas)
    
    if training_data is None:
        print("Data no valida")
        return
    
    # Normalizar características
    feature_mean = np.mean(training_data, axis=0)
    feature_std = np.std(training_data, axis=0)
    feature_std[feature_std == 0] = 1.0
    training_data_norm = (training_data - feature_mean) / feature_std
    
    print(f"Datos cargados: {len(training_data)} espectros")
    print(f"Características extraídas: {training_data.shape[1]}")
    
    # Entrenar SOM
    print("\n" + "=" * 80)
    print("FASE 2: Entrenando SOM (paralelizado)...")
    print("=" * 80)
    
    som = SimpleSOM(map_width=13, map_height=13, feature_dim=training_data.shape[1], learning_rate=0.5)
    som.train(training_data_norm, epochs=train_epochs)
    som.assign_types(training_data_norm, training_labels)
    
    print(f"SOM entrenado ({som.map_height}x{som.map_width}, {train_epochs} épocas)")
    
    # Comparacion
    print("\n" + "=" * 80)
    print("FASE 3: Clasificación y Evaluación")
    print("=" * 80)
    
    aciertos = 0
    total = 0
    
    print(f"{'Archivo'.ljust(25)} | {'Real'.ljust(10)} | {'Predicho'.ljust(10)} | {'Resultado'}")
    print("-" * 60)
    
    for ruta, features, real in zip(rutas_validas, training_data_norm, training_labels):
        pred = som.predict(features)
        
        correcto = (real == pred) or (real.startswith(pred) if pred != "?" else False)
        
        if correcto:
            aciertos += 1
            flag = "V"
        else:
            flag = "X"
        
        total += 1
        print(f"{ruta.name.ljust(25)} | {real.ljust(10)} | {pred.ljust(10)} | {flag}")
    
    print("-" * 60)
    exactitud = aciertos / total if total > 0 else 0
    print(f"Exactitud Total: {aciertos}/{total} = {exactitud:.1%}")
    
    return som, feature_mean, feature_std


if __name__ == "__main__":
    for epochs in [20, 30, 50, 80]:
        inicio = time.time()
        procesar_carpeta("data/BINTABLE", train_epochs=epochs)
        final = time.time()
        tiempo_total = final - inicio
        print(f"El entrenamiento tardó: {tiempo_total:.4f} s\n")
