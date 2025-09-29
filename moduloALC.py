import numpy as np

##L01
def error(x,y):
    """Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64"""

    return abs(x - y)

def error_relativo(x,y):
    """Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64"""

    return abs(x - y) / abs(x)

def matricesIguales(A,B):
    """ Devuelve True si ambas matrices son iguales y False en otro caso. Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores."""

    if A.shape != B.shape:
        return False
    
    epsilon = np.finfo(float).eps
    
    # recorre la matriz con bucles for
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # Valor absoluto para comparar la diferencia de elementos
            abs_diff = np.abs(A[i, j] - B[i, j])

            # Maneja el caso de que el valor B sea cero
            if B[i, j] != 0:
                # Compara usando el error relativo
                rel_diff = abs_diff / np.abs(B[i, j])
                if rel_diff > epsilon:
                    return False
            else:
                # Compara usando el error absoluto
                if abs_diff > epsilon:
                    return False
    
    return True

##L02
def rota(theta):
    """
    Devuelve la matriz de rotación 2x2 para un ángulo theta (en radianes).
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def escala(s):
    """
    Devuelve la matriz diagonal que escala cada coordenada por los valores de s.
    """
    return np.diag(s)

def rota_y_escala(theta, s):
    """
    Rota un vector por theta y después lo escala por s.
    """
    R = rota(theta)
    S = escala(s)
    return S @ R   # primero rota, luego escala

def afin(theta, s, b):
    """
    Devuelve la matriz afín 3x3 que rota, escala y luego traslada.
    """
    M = rota_y_escala(theta, s)
    A = np.eye(3)
    A[:2, :2] = M
    A[:2, 2] = b
    return A

def transafin(v, theta, s, b):
    """
    Aplica la transformación afín al vector v.
    """
    A = afin(theta, s, b)
    vh = np.array([v[0], v[1], 1])  # coordenadas homogéneas
    wh = A @ vh
    return wh[:2]

##L03
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


def condMC(matriz, p):
    return normaMatMc(matriz, p, p, 1000)[0] * normaMatMc(np.linalg.inv(matriz), p, p, 1000)[0]

def condExacta(matriz, p):
    return normaExacta(matriz, p) * normaExacta(np.linalg.inv(matriz), p)

##L04
def calculaLU(A):
    """
    Calcula la factorización LU de la matriz A y retorna las matrices L y U,
    junto con el número de operaciones realizadas. 

    Notar que a medida que el tamaño de la matriz aumenta, la estabilidad numérica va reduciendose debido a la cantidad de operaciones que se realizan, especialmente por la división y la limitada representación de los números irracionales. 

    """
    cant_op = 0
    m = A.shape[0]
    n = A.shape[1]
    Ac = A.copy()
    
    if m != n:
        print('Matriz no cuadrada')
        return

    L = np.eye(n)
    U = np.zeros((n, n))

    for j in range(n-1):  # recorrer columnas
        for i in range(j+1, n):  # recorrer filas i debajo del pivote j
            if Ac[i, j] != 0:  # pivote distinto a cero
                t = Ac[i, j] / Ac[j, j]
                Ac[i, j:] = Ac[i, j:] - t * Ac[j, j:]  # actualizar fila
                L[i, j] = t

                cant_op += 1 + 2*(n-j) # 1 división, (n-j) multiplicaciones y (n-j) restas
    
    U = Ac.copy()
            
    return L, U, cant_op
    


def res_tri(L, b, inferior=True):
    """
    Resuelve el sistema Lx = b, donde L es triangular. 
    Se puede indicar si es triangular inferior o superior usando el argumento 
    'inferior' (por defecto, se asume triangular inferior).
    """
    n = len(b)
    x = np.zeros(n)

    if inferior:  # Sustitución hacia adelante
        for i in range(n):
            suma = 0
            for j in range(i):
                suma += L[i][j] * x[j]
            x[i] = (b[i] - suma) / L[i][i]
    else:  # Sustitución hacia atrás
        for i in range(n-1, -1, -1):
            suma = 0
            for j in range(i+1, n):
                suma += L[i][j] * x[j]
            x[i] = (b[i] - suma) / L[i][i]
    
    return x


def inversa(A):
    """
    Calcula la inversa de A empleando la factorización LU
    y las funciones que resuelven sistemas triangulares.
    """
    n = A.shape[0]
    L, U, _ = calculaLU(A)
    
    invA = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1
        y = res_tri(L, ei, inferior=True)   # Ly = e_i
        x = res_tri(U, y, inferior=False)   # Ux = y
        invA[:, i] = x
    
    return invA


def calculaLDV(A):
    """
    Calcula la factorización LDV de la matriz A, de forma tal que A = L D V,
    con L triangular inferior, D diagonal y V triangular superior.
    En caso de que la matriz no pueda factorizarse, retorna None.
    """
    n = A.shape[0]
    L, U, _ = calculaLU(A)
    D = np.diag(np.diag(U))
    
    V = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            V[i, j] = U[i, j] / D[i, i]
    
    # Revisar si hay pivotes cero
    if np.any(np.diag(D) == 0):
        return None
    
    return L, D, V


def esSDP(A, atol=1e-8):
    """
    Verifica si la matriz A es simétrica definida positiva (SDP)
    usando la factorización LDV.
    """
    if not np.allclose(A, A.T, atol=atol):
        return False
    
    res = calculaLDV(A)
    if res is None:
        return False
    
    _, D, _ = res
    if np.any(np.diag(D) <= 0):
        return False
    
    return True

##L05
def QRconGS(A, tol=1e-12, retorna_nops=False):
    """
    Factorización QR con el método de Gram-Schmidt.

    Parámetros:
    - A: matriz de n x n.
    - tol: tolerancia con la que se filtran elementos nulos en R.
    - retorna_nops: permite (opcionalmente) retornar el número de operaciones realizadas.

    Retorna:
    - Q y R calculadas con Gram-Schmidt.
    - Como tercer argumento opcional, el número de operaciones.
    - Si la matriz A no es de n x n, retorna None.
    """
    pass


def QRconHH(A, tol=1e-12):
    """
    Factorización QR con reflexiones de Householder.

    Parámetros:
    - A: matriz de m x n (con m >= n).
    - tol: tolerancia con la que se filtran elementos nulos en R.

    Retorna:
    - Q y R calculadas con reflexiones de Householder.
    - Si la matriz A no cumple m >= n, retorna None.
    """
    pass


def calculaQR(A, metodo='RH', tol=1e-12):
    """
    Calcula la factorización QR de una matriz.

    Parámetros:
    - A: matriz de n x n.
    - tol: tolerancia con la que se filtran elementos nulos en R.
    - metodo: 'RH' usa reflexiones de Householder, 'GS' usa Gram-Schmidt.

    Retorna:
    - Q y R calculadas con el método indicado.
    - Como tercer argumento opcional, el número de operaciones.
    - Si el método no está entre las opciones, retorna None.
    """
    pass
