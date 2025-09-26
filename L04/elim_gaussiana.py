"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m = A.shape[0]
    n = A.shape[1]
    Ac = A.copy()
    
    if m != n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR

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
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    return L, U, cant_op

# def main():
#     n = 7
#     B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
#     B[:n,n-1] = 1
#     print('Matriz B \n', B)
    
#     L,U,cant_oper = elim_gaussiana(B)
    
#     print('Matriz L \n', L)
#     print('Matriz U \n', U)
#     print('Cantidad de operaciones: ', cant_oper)
#     print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
#     print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

# if __name__ == "__main__":
#     main()
    

