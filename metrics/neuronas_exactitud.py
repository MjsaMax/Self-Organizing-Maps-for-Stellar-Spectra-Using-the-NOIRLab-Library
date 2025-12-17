import matplotlib.pyplot as plt

neuronas = [20, 30, 50, 80]
exactitud_secuencial = [80.6, 80.1, 77.8, 79.4]
exactitud_procesos  = [80.4, 77.5, 76.8, 78.3]
exactitud_hilos    = [79.9, 78.9, 80.8, 79.5] 

plt.figure(figsize=(10, 6)) 

plt.plot(neuronas, exactitud_secuencial, marker='o', label='Secuencial')
for x, y in zip(neuronas, exactitud_secuencial):
    plt.text(x, y + 0.3, f'{y}', ha='center', va='bottom', fontsize=9, color='blue')

plt.plot(neuronas, exactitud_procesos, marker='s', label='Paralelo (Procesos)')
for x, y in zip(neuronas, exactitud_procesos):
    plt.text(x, y - 0.8, f'{y}', ha='center', va='top', fontsize=9, color='orange')

plt.plot(neuronas, exactitud_hilos, marker='^', label='Paralelo (Hilos)')
for x, y in zip(neuronas, exactitud_hilos):
    plt.text(x, y + 0.3, f'{y}', ha='center', va='bottom', fontsize=9, color='green')

plt.xlabel("Numero de neuronas")
plt.ylabel("Exactitud") 
plt.title("Exactitud vs Número de neuronas")

plt.legend()
plt.ylim(40, 90) 
plt.grid(True)
plt.savefig('grafica_exactitud.png', dpi=300, bbox_inches='tight')
plt.show()
