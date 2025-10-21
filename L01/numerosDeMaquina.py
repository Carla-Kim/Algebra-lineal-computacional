import numpy as np
import matplotlib.pyplot as plt


# ejercicio 1

## Comparen el resultado de hacer 0.3 + 0.25 con el de hacer 0.3 − 0.25. ¿En ambos casos obtienen el resultado esperado? ¿Por qué?
print("0.3 + 0.25 =", 0.3 + 0.25) # 0.55 
print("0.3 - 0.25 =", 0.3 - 0.25) # 0.05 pero nos da 0.04999999999999999
"""porque 0.3 no se puede representar exactamente en binario, es una fracción periódica"""

## Escriban el número 0.25 en base 2. ¿Cómo queda expresado en términos de su mantisa y exponente?
"""0 01111101 00000000000000000000000
    - en binario: 0.01 x 2^(-2)
    - signo 0 (positivo)
    - exponente 01111101 (125 en decimal) -> 125 - 127 = -2
    - mantisa 00000000000000000000000 (1.0 en decimal)
"""

## Escriban el número 0.3 en base 2. ¿Qué dificultades aparecen al escribir 0.3 en binario? ¿Se puede escribir exactamente con una mantisa finita?
"""0 01111101 00110011001100110011010
    - en binario: 0.0100110011001100110011001100110011... x 2^(-2)
    - signo 0 (positivo)
    - exponente 01111101 (125 en decimal) -> 125 - 127 = -2
    - mantisa 00110011001100110011010

(repite 0011 infinitamente. No se puede escribir exactamente con una mantisa finita)"""

# ejercicio 2

## ¿Cuánto da (np.sqrt(2)**2-2) en Python? ¿Y (np.sqrt(2)**2-2)==0? ¿Por qué?
print("np.sqrt(2)**2 - 2 =", np.sqrt(2)**2 - 2) # 0.0000000000000004440892098500626
print("Es igual np.sqrt(2)**2 - 2 a cero: ", (np.sqrt(2)**2 - 2) == 0) # da False
""" porque np.sqrt(2) no se puede representar exactamente en binario, es una fracción periódica. Al elevarlo al cuadrado y restarle 2, el error se amplifica """

## Comparen los resultados de las siguientes dos expresiones para valores pequeños de x (por ejemplo, x entre 0 y 5e-8). ¿Qué diferencias encuentran? ¿Por qué?
"""Encontramos que con la expresión a queda resultados cercanos a 0, mientras que la expresión b da resultados mucho más cercanos a 0. Esto se debe a que la expresión a sufre de cancelación numérica cuando x es pequeño, ya que sqrt(2x^2 + 1) se aproxima a 1, y al restar 1 se pierde precisión. En cambio, la expresión b está diseñada para evitar esta cancelación, proporcionando resultados más precisos para valores pequeños de x.

Lo que vemos como pelotitas en la gráfica de a son los errores de redondeo que se van acumulando en la expresión a. Se redondean de tal forma que el resultado final con una x creciente estas dan repetitivamente ceros. La expresión b evita la cancelación numérica y proporcionando resultados más precisos.
"""

# 100 valores de x en el intervalo [0, 5e-8]
x = np.linspace(0, 5e-8, 100)

a = np.sqrt(2*x**2 + 1) - 1
b = (2*x**2) / (np.sqrt(2*x**2 + 1) + 1)

# creamos la figura
plt.figure(figsize=(10,6))

# graficamos cada expresión
plt.plot(x, a, "o", label=r"$\sqrt{2x^2+1} - 1$")
plt.plot(x, b, "x", label=r"$\frac{2x^2}{\sqrt{2x^2+1}+1}$")

# etiquetas y detalles
plt.title("Comparación de expresiones numéricamente equivalentes")
plt.legend()
plt.grid(True)

# mostrar gráfico
# plt.show()

# ejercicio 3

# ejercicio 4
"""
n=6
    suma 1 
        32 bits:  14.357358
        64 bits:  14.392726788474306
    suma 2 
        32 bits:  15.403683
        64 bits:  16.00216430089742

n=7
    suma 1 
        32 bits:  15.403683
        64 bits:  16.695311431453085
    suma 2 
        32 bits:  15.403683
        64 bits:  18.304749306109485
"""

## ¿Porque la modificación cambia el resultado?
"""La modificación cambia el resultado debido a la forma en que se acumulan los errores de redondeo en las operaciones de suma. En la suma, los términos más pequeños se suman primero, lo que minimiza el impacto de los errores de redondeo. Cuando los términos más grandes se suman primero, lo que puede generarse es que se amplifiquen los errores de redondeo cuando se suman con términos mucho más pequeños. Esto es especialmente notable en aritmética de punto flotante de menor precisión (como float32), donde la capacidad para representar números con alta precisión es limitada."""
n=6
s = np.float32(0)
for i in range(2*10**n,0,-1):
    s = s + np.float32(1/i)
#print("suma = ", s)


""""Sumas con n=6"""
n = 6
s = np.float32(0)
for i in range(1,10**n+1):
    s = s + np.float32(1/i)
#print("suma 1 32 bits con n=6: ", s)

n = 6
s = np.float64(0)
for i in range(1,10**n+1):
    s = s + np.float32(1/i)
#print("suma 1 64 bits con n=6: ", s)

n = 6
s = np.float32(0)
for i in range(1,5*10**n+1):
    s = s + np.float32(1/i)
#print("suma 2 32 bits con n=6: ", s)

n = 6
s = np.float64(0)
for i in range(1,5*10**n+1):
    s = s + np.float32(1/i)
#print("suma 2 64 bits con n=6: ", s)


""""Sumas con n=7"""
n = 7
s = np.float32(0)
for i in range(1,10**n+1):
    s = s + np.float32(1/i)
#print("suma 1 32 bits con n=7: ", s)

n = 7
s = np.float64(0)
for i in range(1,10**n+1):
    s = s + np.float32(1/i)
#print("suma 1 64 bits con n=7: ", s)

n = 7
s = np.float32(0)
for i in range(1,5*10**n+1):
    s = s + np.float32(1/i)
#print("suma 2 32 bits con n=7: ", s)

n = 7
s = np.float64(0)
for i in range(1,5*10**n+1):
    s = s + np.float32(1/i)
#print("suma 2 64 bits con n=7: ", s)

# ejercicio 5
A = np.array([[4., 2., 1.], [2., 7., 9.], [0., 5., 22/3]])
L = np.array([[1., 0., 0.], [0.5, 1., 0.], [0., 5/6, 1.]])
U = np.array([[4., 2., 1.], [0., 6., 8.5], [0., 0., 0.25]])

print("¿La descomposicón LU es la misma?", alc.matricesIguales(A, L@U)) # deberia dar true

# ejercicio 6
A = np.array(np.random.rand(4,4))
A_mod1 = (A * 0.25)/0.25
A_mod2 = (A * 0.2)/0.2

def esCuadrada(matriz):
    return matriz.shape[0] == matriz.shape[1]

def traspuesta(matriz):
    columna = matriz.shape[1]
    fila = matriz.shape[0]
    matrizT = np.zeros((fila, columna))

    for i in range(fila):
        for j in range(columna):
            matrizT[j][i] = matriz[i][j]

    return matrizT

def esSimetrica(matriz):
    ##return np.allclose(matriz,traspuesta(matriz))

    if not esCuadrada(matriz):
        return False
    
    matrizT = traspuesta(matriz)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i][j] != matrizT[i][j]:
                return False
            
    return True

## .T es la traspuesta de A de numpy
print("esSimetrica(A.T @ A):", esSimetrica(A.T @ A)) # True
print("esSimetrica(A.T@A_mod1):", esSimetrica(A.T@A_mod1)) # True
print("esSimetrica(A.T@A_mod2):", esSimetrica(A.T@A_mod2)) # False, por errores de redondeo 0,2 no se puede representar exactamente en binario