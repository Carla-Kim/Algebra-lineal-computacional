import numpy as np

# ejercicio 1 
def esCuadrada(matriz):
    return matriz.shape[0] == matriz.shape[1]

# ejercicio 2 
    # shape tupla(fila, columna)
    # size total elementos
def triangSup(matriz):

    columna = matriz.shape[1]
    fila = matriz.shape[0]
    matrizNueva = np.zeros((fila, columna))

    for i in range(fila):
        for j in range(columna):
            if j > i:
                matrizNueva[i][j] = matriz[i][j]

    return matrizNueva

# ejercicio 3
def triangInf(matriz):

    columna = matriz.shape[1]
    fila = matriz.shape[0]
    matrizNueva = np.zeros((fila, columna))

    for i in range(fila):
        for j in range(columna):
            if j < i:
                matrizNueva[i][j] = matriz[i][j]

    return matrizNueva

# ejercicio 4
def diagonal(matriz):
    columna = matriz.shape[1]
    fila = matriz.shape[0]
    matrizNueva = np.zeros((fila, columna))

    for i in range(fila):
        for j in range(columna):
            if j == i:
                matrizNueva[i][j] = matriz[i][j]

    return matrizNueva

# ejercicio 5
def traza(matriz):
    return np.sum(diagonal(matriz))

# ejercicio 6
def traspuesta(matriz):
    columna = matriz.shape[1]
    fila = matriz.shape[0]
    matrizT = np.zeros((fila, columna))

    for i in range(fila):
        for j in range(columna):
            matrizT[j][i] = matriz[i][j]

    return matrizT

# ejercicio 7
    # con == se compara elemento a elemento
    # allclose verifica igualdad aproximada con flotantes
def esSimetrica(matriz):
    ##return np.allclose(matriz,traspuesta(matriz))

    res = True
    if not esCuadrada(matriz):
        return False
    
    matrizT = traspuesta(matriz)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i][j] != matrizT[i][j]:
                return False
            
    return res

# ejercicio 8
    # un vector n se interpreta como nx1
def calcularAx(matriz, x):
    return matriz @ x if matriz.shape[1] == x.shape[0] else "No se puede"

# ejercicio 9
    # el doble corchete devuelve un array que contiene varias filas, en el orden que pongas en la lista y mantiene en orden las demas filas que no selecciona
def intercambiarFilas(matriz, i, j):
    A[[i-1, j-1]] = A[[j-1, i-1]]
    return A

# ejercicio 10 
def sum_fila_multiplo(matriz, i, j, s):
    A[i-1] += A[s*(j-1)]
    return A

# ejercicio 11
def esDiagonalmenteDominante(matriz):
    if esCuadrada(matriz):
        res = True
        fila = matriz.shape[0]

        for i in range(fila):
            res = abs(matriz[i][i]) > np.sum(abs(matriz[i])) - abs(matriz[i][i])
    else:
        return False

    return res

# ejercicio 12
    # secuencia[inicio:fin:paso]
        #inicio: índice desde donde arranca (incluido).
        #fin: índice hasta donde corta (excluido).
        #paso: cada cuántos elementos avanza.
    
def matrizCirculante(v):
    n = v.size
    matriz = np.zeros((n,n))

    for i in range(n):
        matriz[i] = np.concatenate((v[-i:], v[:-i]))
             # v[-k:] ultimos k elementos
             # c[:-k] todos los demas

    return matriz

# ejercicio 13
def matrizVandermode(v):
    return np.vander(v,increasing=True)

# ejercicio 14
def numeroAureo(n):
    """
    Calcula las aproximaciones de phi = F_{k+1}/F_k para k=1..n
    usando la primera fila de la matriz de Fibonacci.
    """
    M = matrizFibonacci(n+1)   # necesito hasta F_{n+1}
    fila = M[0]                # primera fila: [F0, F1, F2, ...]
    
    ks = np.arange(1, n+1)
    ratios = []
    for k in ks:
        Fk = fila[k]       # F_k
        Fk1 = fila[k+1]    # F_{k+1}
        ratios.append(Fk1 / Fk)
    
    return ks, np.array(ratios, dtype=float)

def matrizFibonacci(n):
    F = [0, 1]  # F0, F1
    for k in range(2, 2*n):
        F.append(F[k-1] + F[k-2])
    
    # Llenamos la matriz
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matriz[i, j] = F[i + j]
    
    return matriz
# ejercicio 16
def matrizHilbert(n):
    matriz = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matriz[i][j] = (1/(i+j+1))

    return matriz

# ejercicio 17

"""
Usando las funciones previamente desarrolladas donde sea posible, escriba una rutina que calcule los valores, en el intervalo [-1,1], de los siguientes polinomios:

p_1(x) = x^5 - x^4 + x^3 - x^2 + x - 1
p_2(x) = x^2 + 3
p_3(x) = x^{10} - 2
Grafique el valor de los polinomios en el rango indicado y calcule:

La cantidad de operaciones necesarias.
El espacio en memoria requerido para generar 100 puntos equiespaciados entre -1 y 1.
Finalmente, responda:

¿Cómo crecen estos valores con n?
¿Qué modificaría para hacer el cálculo más eficiente?
"""

# ejercicio 18
def row_echelon(M):
    """Return Row Echelon Form of matrix M con pivoteo parcial"""
    A = M.astype(float)
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # -------------------------------
    # Pivoteo parcial (nuevo)
    max_row = np.argmax(np.abs(A[:,0]))  # fila con valor de mayor módulo en la primera columna
    if max_row != 0:
        A[[0, max_row]] = A[[max_row, 0]]  # swap de filas
    # -------------------------------

    # -------------------------------
    # Pivoteo anterior (solo buscaba primer elemento != 0)
    # for i in range(len(A)):
    #     if A[i,0] != 0:
    #         break
    # else:
    #     # si toda la primera columna es cero, pasamos a la submatriz desde la segunda columna
    #     B = row_echelon(A[:,1:])
    #     return np.hstack([A[:,:1], B])
    #
    # if i > 0:
    #     ith_row = A[i].copy()
    #     A[i] = A[0]
    #     A[0] = ith_row
    # -------------------------------

    # si el pivote es cero (toda la columna era cero), pasamos directo
    if A[0,0] == 0:
        B = row_echelon(A[:,1:])
        return np.hstack([A[:,:1], B])

    # normalizamos la fila pivote
    A[0] = A[0] / A[0,0]

    # eliminamos las entradas debajo del pivote
    A[1:] -= A[0] * A[1:,0:1]

    # recursión sobre el bloque inferior derecho
    B = row_echelon(A[1:,1:])

    return np.vstack([A[:1], np.hstack([A[1:,:1], B])])

### 

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [1, 2, 3],
    [2, 5, 6],
    [3, 6, 9]
])

C = np.array([
    [10, 2, 3],
    [2, 10, 6],
    [3, 6, 10]
])

x = np.array([1,0,-1])
y = np.array([1,0])


#print(esCuadrada(A))
#print(triangSup(A))
#print(triangInf(A))
#print(diagonal(A))
#print(traza(A))
#print(traspuesta(A))
#print(esSimetrica(A))
#print(esSimetrica(B))
#print(calcularAx(A,x))
#print(calcularAx(A,y))
#print(intercambiarFilas(A, 1, 3))
#print(esDiagonalmenteDominante(A))
#print(esDiagonalmenteDominante(C))
#print(matrizHilbert(3))
#print(matrizVandermode(np.array([1,2,3])))
#print(matrizCirculante(np.array([1,2,3])))