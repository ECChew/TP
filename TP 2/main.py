import numpy as np


def MIGenerale(M, N, b, x0, epsilon, Nitermax):
    Minv = np.linalg.inv(M)
    j = np.dot(Minv, N)
    k = np.dot(Minv, b)
    VP = map(abs, np.linalg.eigvals(j))
    rho = max(VP)
    xsave = x0
    xp = np.dot(j, xsave) + k  # Calcul de x1
    n = 1
    err = np.linalg.norm(xsave - xp)
    if rho >= 1:
        print("Rayon spectral supérieur ou égal à 1, divergence, arrêt de l'algorithme")
        print("rho =", rho, "\n")
    else:

        while (err > epsilon) and (n < Nitermax):
            xsave = xp
            xp = np.dot(j, xsave) + k
            n += 1
            err = np.linalg.norm(xsave - xp)
    return xp, n, err


def MIJacobi(A, b, x0, epsilon, Nitermax):
    M = np.diag(np.diag(A))
    N = M - A
    x, niter, err = MIGenerale(M, N, b, x0, epsilon, Nitermax)
    return x, niter, err


def MIGaussSeidel(A, b, x0, epsilon, Nitermax):
    M = np.tril(A)
    N = M - A
    x, niter, err = MIGenerale(M, N, b, x0, epsilon, Nitermax)
    return x, niter, err


def Jacobi(A, b):
    M = np.diag(np.diag(A))
    N = M - A
    J = np.dot(np.linalg.inv(M), N)
    K = np.dot(np.linalg.inv(M), b)
    return J, K


def MI(J, K, x0, eps, nitermax):
    xs = x0
    xp = np.dot(J, xs) + K  # Calcul de x1
    n = 1
    err = np.linalg.norm(xs - xp)
    while (err > eps) and (n < nitermax):
        xs = xp
    xp = np.dot(J, xs) + K
    n += 1
    err = np.linalg.norm(xs - xp)
    return xp, n, err


def construct():
    n = int(input("Taille?\n"))
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    for i in range(n):
        b[i][0] = np.cos(i / 5)
        for j in range(n):
            if i == j:
                A[i][i] = 2
            else:
                A[i][j] = 1 / (10 + (3 * i - 4 * j) ** 2)
    return A, b


def construct2():
    n = int(input("Taille?\n"))
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    for i in range(n):
        b[i][0] = np.cos(i / 5)
        for j in range(n):
            A[i][j] = 1 / (1 + 5 * np.abs(i - j))
    return A, b


if __name__ == '__main__':
    # n=int(input("Insérez n\n"))
    # A = np.random.rand(n, n)
    # b = np.random.rand(n, 1)
    epsilon = float(input("Insérez epsilon\n"))
    Nitermax = int(input("Insérez le nombre d'itérations max\n"))
    A = np.array([[1, 2, -2], [1, 1, 1], [2, 2, 1]])
    b = (np.array([-1, 6, 9])).reshape(3, 1)
    la, ca = np.shape(A)
    x0 = np.zeros((la, 1))
    xn0 = np.zeros((100, 1))
    print("-------------Vraie solution-------------")
    sol = np.linalg.solve(A, b)
    print("A :", A)
    print("b :", b)
    print("Solution :", sol, "\n")
    Ae, be = construct()
    Ae2, be2 = construct2()
    print("-----------------Jacobi-----------------")
    xJ, niterJ, errJ = MIJacobi(A, b, x0, epsilon, Nitermax)
    #print("Solution Jacobi :", xJ, "\n")
    print("Nombre d'itérations Jacobi :", niterJ, "\n")
    print("Dernière erreur Jacobi :", errJ, "\n")
    """print("--------------Gauss-Siedel--------------")
    xG, niterG, errG = MIGaussSeidel(A, b, x0, epsilon, Nitermax)
    print("Solution Gauss-Seidel :", xG, "\n")
    print("Nombre d'itérations Gauss-Seidel :", niterG, "\n")
    print("Dernière erreur Gauss-Seidel :", errG, "\n")"""
    print("------------Expérimentation-------------")
    print("-----------------Jacobi-----------------")
    xJe, niterJe, errJe = MIJacobi(Ae, be, xn0, epsilon, Nitermax)
    #print("Solution Jacobi :", xJe, "\n")
    print("Nombre d'itérations Jacobi :", niterJe, "\n")
    print("Dernière erreur Jacobi :", errJe, "\n")
    print("--------------Gauss-Siedel--------------")
    xGe, niterGe, errGe = MIGaussSeidel(Ae, be, xn0, epsilon, Nitermax)
    #print("Solution Gauss-Seidel :", xGe, "\n")
    print("Nombre d'itérations Gauss-Seidel :", niterGe, "\n")
    print("Dernière erreur Gauss-Seidel :", errGe, "\n")
    print("------------Expérimentation-------------")
    print("-----------------Jacobi-----------------")
    xJe2, niterJe2, errJe2 = MIJacobi(Ae2, be2, xn0, epsilon, Nitermax)
    #print("Solution Jacobi :", xJe2, "\n")
    print("Nombre d'itérations Jacobi :", niterJe2, "\n")
    print("Dernière erreur Jacobi :", errJe2, "\n")
    print("--------------Gauss-Siedel--------------")
    xGe2, niterGe2, errGe2 = MIGaussSeidel(Ae2, be2, xn0, epsilon, Nitermax)
    #print("Solution Gauss-Seidel :", xGe2, "\n")
    print("Nombre d'itérations Gauss-Seidel :", niterGe2, "\n")
    print("Dernière erreur Gauss-Seidel :", errGe2, "\n")
