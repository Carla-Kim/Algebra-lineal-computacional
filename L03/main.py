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
    n = matriz.shape[1]

    maxNorm = -np.inf # valor inicial muy pequeño
    xMax = None
    
    for _ in range(Np):
        # generar vector aleatorio
        x = np.random.randn(n)

        # Normalizar con norma p = 1
        x = x / norma(x, p)
        
        # Evaluar norma q de Ax
        Ax = matriz @ x
        val = norma(Ax, q)

        if val > maxNorm:
            maxNorm = val
            xMax = x
        
    return maxNorm, xMax

A = np.array([[1, 2], 
              [3, 4]])

valor, x = normaMatMc(A, q=2, p=2, Np=10000)
print("Norma inducida aproximada:", valor) # 7.05103848916485 
print("Vector donde se alcanza:", x) #[-1.75071981 -0.40376836]

def normaExacta(matriz, p=[1, "inf"]):
    n = matriz.shape[0]
    m = matriz.shape[1]
    matrizAux1 = np.zeros(n)
    matrizAux2 = np.zeros(m)
    normaInf = 0
    normaUno = 0

    for i in range(n):
        matrizAux1[i] = sum(abs(matriz[i, :]))
    normaInf = np.max(matrizAux1)

    for i in range(m):
        matrizAux2[i] = sum(abs(matriz[:, i]))
    normaUno = np.max(matrizAux2)

    if p == 1:
        return normaUno
    elif p == "inf":
        return normaInf
    else:
        return "p debe ser 1 o 'inf'"


A = np.array([[1, -2, 3],
              [4,  0, -1]])

print("Norma 1:   ", normaExacta(A, 1)) # Norma 1:    5.0
print("Norma inf: ", normaExacta(A, "inf")) # Norma inf:  6.0

# ejercicio 3
def condMC(matriz, p):
    return normaMatMc(matriz, p, p, 1000)[0] * normaMatMc(np.linalg.inv(matriz), p, p, 1000)[0]

def condExacta(matriz, p):
    return normaExacta(matriz, p) * normaExacta(np.linalg.inv(matriz), p)