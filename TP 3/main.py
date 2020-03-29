import numpy as np


def PuissanceVP(A, x0, maxiter, epsilon):
    xn = x0 / (np.linalg.norm(x0))  # x0
    yn = np.dot(A, xn)  # y0
    nyn = np.linalg.norm(yn)
    xnp = yn / nyn  # x1
    xnpn = xnp / (np.linalg.norm(xnp))  # x1 normé
    n = 0  # compteur
    while (np.linalg.norm(xnpn - xn) <= epsilon) and (np.linalg.norm(xnpn + xn) <= epsilon) and n < maxiter:
        xn = xnpn  # xn-1
        yn = np.dot(A, xn)  # yn-1
        nyn = np.linalg.norm(yn)  # norme yn-1
        xnp = yn / (nyn * np.linalg.norm(yn))  # xn
        n += 1
    l = np.linalg.norm(np.dot(A,xnp))
    pres = np.linalg.norm(np.dot(A, xnp) - l * xnp)
    return (l, xnp, n,pres)


if __name__ == '__main__':
    size=(3,3)
    epsilon = float(input("Epsilon?"))
    A = np.ones(size)
    x0 = np.array([[10], [100], [-500]])
    maxiter = int(input("Max iter?"))
    l, x, n, pres = PuissanceVP(A, x0, maxiter, epsilon)
    print("lambda = ", l)
    #print("nbiter = ", n)
    print("précision = ", pres)
    print("x = ", x)
