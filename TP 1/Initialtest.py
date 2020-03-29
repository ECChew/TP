import numpy as np
import time as time


def ResolutionSystTriInf(Taug):
    (m, n) = Taug.shape
    X = np.zeros((m, 1))
    for i in range(m):
        S = 0.
        for j in range(i):
            S += Taug[i, j] * X[j]
        X[i] = (Taug[i, -1] - S) / (Taug[i, i])
    X = X.reshape(X.size, 1)
    return X


def ReductionGauss(Aaug):
    (m, n) = Aaug.shape
    X = np.copy(Aaug)
    for i in range(m):
        for j in range(i + 1, m):
            g = X[j, i] / X[i, i]
            if g == 0:
                print("error")
                quit
            X[j, :] = X[j, :] - g * X[i, :]
    return X


def Gauss(A, B):
    T = np.hstack([A, B])
    R = ReductionGauss(T)
    return ResolutionSystTriSup(R)


def ResolutionSystTriSup(Taug):
    (m, n) = Taug.shape
    X = np.zeros((m, 1))
    for i in range(m):
        S = 0.
        for j in range(i):
            S += Taug[m - i - 1, m - j - 1] * X[m - j - 1]
        X[m - i - 1] = (Taug[m - i - 1, -1] - S) / (Taug[m - i - 1, m - i - 1])
    X = X.reshape(X.size, 1)
    return X


def creationmatrice():
    n = int(input("Nombre de lignes?"))
    m = int(input("Nombre de colonnes?"))
    A = np.empty([n, m])
    for i in range(n):
        for j in range(m):
            A[i, j] = float(input("Coefficient ligne " + str(i + 1) + " colonne " + str(j + 1) + "\n"))
    return (A)


def householder(v):
    l = len(v)
    n = np.linalg.norm(v)
    v.shape = (l, 1)
    return np.eye(l) - (2 / (n ** 2)) * np.dot(v, np.transpose(v))


def decompoQR(A):
    Aoriginal = A
    n, m = np.shape(A)
    L = []
    for k in range(m - 1):
        Ap = A[k:, k:]
        w = Ap[:, 0]
        wh = w - w
        normw = np.linalg.norm(w)
        if w[0] == normw:
            wh = np.zeros(n - k)
        elif w[0] < 0:
            wh[0] = normw
        else:
            wh[0] = -normw
        v = w - wh
        H = householder(v)
        if np.shape(H) != (n, n):
            F = np.eye(n)
            for i in range(k, n):
                for j in range(k, m):
                    F[i, j] = H[i - k, j - k]
        else:
            F = H
        L.append(F)
        A = np.dot(F, A)
    R = A
    Q = L[0]
    for p in range(1, len(L)):
        Q = np.dot(Q, L[p])
    return (Aoriginal, Q, R)


def ResolutionQR(A, b):
    Aoriginal, Q, R = decompoQR(A)
    n, m = np.shape(R)
    if n == m:
        Y = np.dot(Q.T, b)
        Aaug = np.c_[R, Y]
        X = ResolutionSystTriSup(Aaug)
        return (X)
    else:
        return ("Matrice non carrée")


if __name__ == '__main__':
    # A = np.array([[6, -12, 337], [-12, 9, 16], [-12, 30, 130]])
    n = int(input("Taille de la matrice carrée ?"))
    A = np.random.rand(n, n)
    b = np.random.rand(n, 1)
    # b = np.array([[-1], [2], [1]])
    ti = time.time()
    X = ResolutionQR(A, b)
    tn = time.time()
    print("Durée d'exécution (QR) : ", tn - ti, "secondes .\n")
    print("X=", X, "\n")
    print("AX-b=", np.dot(A, X) - b, "\n")
    print("||AX-b||²=", np.linalg.norm(np.dot(A, X) - b) ** 2, "\n")
    ti = time.time()
    Y = Gauss(A, b)
    tn = time.time()
    print("Durée d'exécution (Gauss) : ", tn - ti, "secondes .\n")
