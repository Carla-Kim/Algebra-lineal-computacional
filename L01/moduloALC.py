import numpy as np

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