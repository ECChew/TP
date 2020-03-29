import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gradientPasFixe(A, b, x0, rho, tol):
    iMax = 10 ** 6
    i = 1
    r = np.dot(A, x0) - b
    d = -r
    xp = x0 + rho * d
    i += 1
    xit = []
    xit.append(xp)
    while (np.linalg.norm(r) > tol) and (i < iMax):
        x = xp
        r = np.dot(A, x) - b
        d = -r
        xp = x + rho * d
        xit.append(xp)
        i += 1
    return xp, xit, i


def gradientPasOptimal(A, b, x0, tol):
    iMax = 10 ** 6
    i = 1
    r = np.dot(A, x0) - b
    d = -r
    rho = r.T * r / (np.dot(r.T, np.dot(A, r)))
    xp = x0 + rho * d
    xit = []
    xit.append(xp)
    while (np.linalg.norm(r) > tol) and (i < iMax):
        x = xp
        r = np.dot(A, x) - b
        d = -r
        rho = np.dot(r.T, r) / (np.dot(r.T, np.dot(A, r)))
        xp = x + rho * d
        xit.append(xp)
        i += 1
    return xp, xit, i


def gradientConjugue(A, b, x0, tol):
    iMax = 10 ** 6
    i = 1
    r = np.dot(A, x0) - b
    x = x0
    xit = []
    xit.append(x0)
    while (np.linalg.norm(r) > tol) and (i < iMax):
        rs = r
        r = np.dot(A, x) - b
        if (i == 1):
            d = -r

        else:
            beta = (np.linalg.norm(r)) ** 2 / (np.linalg.norm(rs)) ** 2
            d = -r + np.dot(beta, d)
        rho = np.dot(r.T, r) / (np.dot(d.T, np.dot(A, d)))
        x = x + rho * d
        i += 1
        xit.append(x)
    return x, xit, i


def fonctionnelle(c, X):
    return 0.5 * (c.T @ X.T @ X @ c - 2 * c.T @ X.T @ q + q.T @ q)


def F(c, X, q):
    Z = X.T @ X
    omega = X.T @ q
    s = q.T @ q
    return 0.5 * (Z[0, 0] * c[0] ** 2 + 2 * Z[0, 1] * c[0] * c[1] + Z[1, 1] * c[1] ** 2 - 2 * (
            omega[0] * c[0] + omega[1] * c[1]) + s)


if __name__ == '__main__':
    rho = 10 ** (-3)
    x0 = np.array([[-9], [7]])
    tol = 10 ** (-6)

    P = np.genfromtxt('dataP.dat', dtype='float').reshape((50, 1))
    Q = np.genfromtxt('dataQ.dat', dtype='float').reshape((50, 1))
    O = np.ones(50).reshape((50, 1))
    X = np.concatenate((O, P), axis=1)
    A = np.dot(X.T, X)
    b = np.dot(X.T, Q)
    plt.scatter(P, Q, label="Série", color='red')
    plt.xlabel("Âge")
    plt.ylabel("Taille")
    plt.title("Représentation graphique")
    plt.show()
    sol, xit, nit = gradientPasFixe(A, b, x0, rho, tol)
    sol2, xit2, nit2 = gradientPasOptimal(A, b, x0, tol)
    sol3, xit3, nit3 = gradientConjugue(A, b, x0, tol)
    print("Gradient pas fixe : ", sol, nit, "\n")
    print("Erreur : ", np.linalg.norm(np.dot(A, sol) - b), "\n")
    print("Gradient pas optimal : ", sol2, nit2, "\n")
    print("Erreur : ", np.linalg.norm(np.dot(A, sol2) - b), "\n")
    print("Gradient conjugué : ", sol3, nit3, "\n")
    print("Erreur : ", np.linalg.norm(np.dot(A, sol3) - b), "\n")
    # Partie 2
    delta = 0.5
    x = np.arange(-10, 10, delta)
    y = np.arange(-10, 10, delta)
    xx, yy = np.meshgrid(x, y)
    XY = [xx, yy]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zz = F(XY, X, Q)
    ax.plot_surface(xx, yy, zz, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
    fig, ax = plt.subplots()
    CS = ax.contour(xx, yy, zz, np.arange(0, 50000, 10000))
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title('Lignes de niveaux')
    plt.show()
    #Fonctionnelle dim 2
    
