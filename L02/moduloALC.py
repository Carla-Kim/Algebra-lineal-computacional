import numpy as np

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


# Ejemplo de uso
theta = np.pi/2         # 90 grados
s = [2, 0.5]            # escala en x e y
b = [1, -1]             # traslación
v = [1, 0]              # vector inicial

print("Matriz rotación:\n", rota(theta))
print("Matriz escala:\n", escala(s))
print("Rota y escala:\n", rota_y_escala(theta, s))
print("Matriz afin:\n", afin(theta, s, b))
print("Vector transformado:", transafin(v, theta, s, b))
