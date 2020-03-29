# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:32:42 2017

@author: Administrateur
"""

import numpy as np
import time


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


def ReductionGaussChoixPivot1(Aaug):
    (m, n) = Aaug.shape
    X = np.copy(Aaug)
    t = 0
    for i in range(m - 1):
        for j in range(i + 1, m):
            if abs(X[j, i]) > abs(X[i, i]):
                t = j
        T = X[i, :].copy()
        X[i, :] = X[t, :]
        X[t, :] = T
        for j in range(i + 1, m):
            h = X[j, i] / X[i, i]
            if h == 0:
                print("error")
                quit
            X[j, :] = X[j, :] - h * X[i, :]
    return X


def GaussChoixPivotPartiel(A, B):
    T = np.hstack([A, B])
    R = ReductionGaussChoixPivot1(T)
    return ResolutionSystTriSup(R)


def ReductionGaussChoixPivotTotal(Aaug):
    (m, n) = Aaug.shape
    X = np.copy(Aaug)
    t = 0
    u = 0
    J = np.zeros((m - 1, 2))
    for i in range(m - 1):
        c = 0
        for k in range(i, m):
            for j in range(i, m):
                if abs(X[k, j]) > c:
                    t = k
                    u = j
                    c = abs(X[k, j])
        J[i, 0] = u
        J[i, 1] = i
        T = X[i, :].copy()
        X[i, :] = X[t, :]
        X[t, :] = T
        U = X[:, i].copy()
        X[:, i] = X[:, u]
        X[:, u] = U
        for j in range(i + 1, m):
            h = X[j, i] / X[i, i]
            if h == 0:
                print("error")
                quit
            X[j, :] = X[j, :] - h * X[i, :]
    R = ResolutionSystTriSup(X)
    for h in reversed(range(0, m - 1)):
        num = int(J[h, 0])
        inv = int(J[h, 1])
        copy = R[inv, 0]
        R[inv, 0] = R[num, 0]
        R[num, 0] = copy
    return R


def DecompositionLU(A):
    (m, n) = A.shape
    X = np.copy(A)
    L = np.eye(m, m)
    for i in range(m):
        for j in range(i + 1, m):
            g = X[j, i] / X[i, i]
            L[j, i] = g
            if g == 0:
                print("error")
                quit
            X[j, :] = X[j, :] - g * X[i, :]
    return L, X


def ResolutionLUsansLU(Aaug):
    A = Aaug[:, :-1]
    L, X = DecompositionLU(A)
    X2 = Aaug[:, -1]
    Laug = np.c_[L, X2]
    X3 = ResolutionSystTriInf(Laug)
    X4 = np.hstack([X, X3])
    R = ResolutionSystTriSup(X4)
    print(R)
    return R


def ResolutionLU(L, U, B):
    LB = np.c_[L, B]
    Y = ResolutionSystTriInf(LB)
    UY = np.c_[U, Y]
    X = ResolutionSystTriSup(UY)
    return X


print("____________________________________________________________________")
print("Résolution par pivot de Gauss")
print("____________________________________________________________________")

"""
Taug = np.array([[4,5,1,8],[0,3,-1,1],[0,0,2,2]])
print(ResolutionSystTriSup(Taug))
"""
av = time.clock()
print(
    'fonction de Gauss basique avec des matrices aléatoires : (on retourne ici seulement la norme du vecteur AX-B afin de vérifier la validité du résultat)')
A = np.random.random((100, 100))
B = np.random.random((100, 1))
Aaug = np.c_[A, B]
"""X = Gauss(A,B)
#print(X)
print("\n\n")
print("voici la norme du vecteur AX-B:",np.linalg.norm(A@X-B))

print("\n\n")
#Aaug = np.array([[12.,1,14,1,5,8,0],[4,1,50,5,4,7,58],[5,0,2,3,3,6,9],[8,4,7,9,10,5,5],[1,1,1,1,10,2,6],[2,5,6,85,7,9,4]])
#Aaug = np.array([[1.,1,1,1],[3,5,8,0],[2,1,6,6]])
Aaug  = np.array([[3.,5,8,9,7,8,10,2,5,3,12],[1,1,1,30,50,62,89,78,45,12,27],[60,66,68,69,67,98,99,12,50,63,78],[11,10,9,8,7,6,5,4,3,2,1],[27,28,29,50,3,2,7,9,4,5,100],[2,1,2,1,2,23,789,48,12,14,15],[101,102,103,104,10,45,78,12,15,9,1],[1,2,3,4,5,6,7,8,9,10,11],[3.,85,89,56,74,12,36,89,15,23,102],[3.,23,12,7,7,39,10,2,40,4,42]])
print(Aaug)
print("\n\n")
print("Pivot de Gauss : \n")
print(ResolutionSystTriSup(ReductionGauss(Aaug)))

print("\n\n")
print("Pivot de Gauss avec choix de pivot (1): \n")
print(ResolutionSystTriSup(ReductionGaussChoixPivot1(Aaug)))"""

print("\n\n")
print("choix total : \n")
print(ReductionGaussChoixPivotTotal(Aaug))

"""
print("\n\n")
print("Décomposition LU : \n")
ResolutionLUsansLU(Aaug)
Y = Aaug[:,:-1]
V = Aaug[:,-1]
L,U=DecompositionLU(Y)
print("\n\n")
print(ResolutionLU(L,U,V))"""

print("temps d'éxecution", time.clock() - av)
