import numpy as np
import matplotlib.pyplot as plt

# ejericicio 1


def norma(vector, p):
    n = vector.size
    res = 0

    if p == "inf":
        for i in range(n):
            vector[i] = abs(vector[i])
            res = max(vector)
    else:
        for i in range(n):
            res += abs(vector[i]) ** p
            res = res ** (1 / p)

    return res


def normaliza(vectores, p):
    n = np.range(vectores)[0]

    for i in range(n):
        vectores[i] = norma(i, p)

    return vectores


"""graficamos los vectores con norma 1 de R**2 según las normas p = 1, 2, 5, 10, 100, 200"""

p_values = [1, 2, 5, 10, 100, 200]
theta = np.linspace(0, 2 * np.pi, 500)

plt.figure(figsize=(8, 8))

for p in p_values:
    puntos = []
    for t in theta:
        x, y = np.cos(t), np.sin(t)
        v = np.array([x, y])
        v_norm = v / norma(v, p)  # normalizamos para que quede con norma p = 1
        puntos.append(v_norm)
    puntos = np.array(puntos)
    plt.plot(puntos[:, 0], puntos[:, 1], label=f"p = {p}")

# Norma infinito: el cuadrado
square = np.array([[1,1], [1,-1], [-1,-1], [-1,1], [1,1]])
plt.plot(square[:,0], square[:,1], label=r"$p = \infty$")

plt.gca().set_aspect("equal")
plt.grid(True)
plt.legend()
plt.title("Bolas unitarias en R^2 para distintas normas p")
plt.show()

"""
v = np.array([1, 2, 3, 4])
print(norma(v, 1)) # 10.0
print(norma(v, 2)) # 5.477225575051661
print(norma(v, 3)) # 16.97343023666491
"""

# ejecicio 2

def normaMatMc(matriz, q, p, Np):
    m = matriz.shape[0]
    n = matriz.shape[1]

    max_norm = -np.inf
    x_max = None
    
    for _ in range(Np):
        # Generar vector aleatorio
        x = np.random.randn(n)
        # Normalizar con norma p = 1
        x = x / norma(x, p)
        
        # Evaluar norma q de Ax
        Ax = matriz @ x
        val = norma(Ax, q)
        
        if val > max_norm:
            max_norm = val
            x_max = x.copy()
    
    return max_norm, x_max

A = np.array([[1, 2], 
              [3, 4]])

valor, x = normaMatMc(A, q=2, p=2, Np=10000)
print("Norma inducida aproximada:", valor)
print("Vector donde se alcanza:", x)
