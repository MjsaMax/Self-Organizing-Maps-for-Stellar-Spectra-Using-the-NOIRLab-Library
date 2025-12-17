import matplotlib.pyplot as plt

neuronas = [20, 30, 50, 80]
tiempo_hilos      = [211, 301, 486, 763]
tiempo_procesos   = [205, 282, 423, 642]
tiempo_secuencial = [193, 276, 429, 691]

plt.figure(figsize=(10, 6)) 

plt.plot(neuronas, tiempo_secuencial, marker='o', label='Secuencial')
for x, y in zip(neuronas, tiempo_secuencial):
    plt.text(x, y + 0.3, f'{y}', ha='center', va='bottom', fontsize=9, color='blue')

plt.plot(neuronas, tiempo_procesos, marker='s', label='Paralelo (Procesos)')
for x, y in zip(neuronas, tiempo_procesos):
    plt.text(x, y - 0.8, f'{y}', ha='center', va='top', fontsize=9, color='orange')

plt.plot(neuronas, tiempo_hilos, marker='^', label='Paralelo (Hilos)')
for x, y in zip(neuronas, tiempo_hilos):
    plt.text(x, y + 0.3, f'{y}', ha='center', va='bottom', fontsize=9, color='green')

plt.xlabel("Numero de neuronas")
plt.ylabel("Tiempo (s)") 
plt.title("Tiempo vs Número de neuronas")

plt.legend()
plt.ylim(150, 800) 
plt.grid(True)
plt.savefig('grafica_tiempo.png', dpi=300, bbox_inches='tight')
plt.show()
