import numpy as np


def PuissanceInvVP(A, x0, maxiter, epsilon):
    xn = x0 / (np.linalg.norm(x0))  # x0
    yn = np.dot(np.linalg.inv(A), xn)  # y0
    nyn = np.linalg.norm(yn)
    xnp = yn / nyn  # x1
    xnpn = xnp / (np.linalg.norm(xnp))  # x1 normé
    n = 0  # compteur
    while (np.linalg.norm(xnpn - xn) <= epsilon) and (np.linalg.norm(xnpn + xn) <= epsilon) and n < maxiter:
        xn = xnpn  # xn-1
        yn = np.dot(np.linalg.inv(A), xn)  # yn-1
        nyn = np.linalg.norm(yn)  # norme yn-1
        xnp = yn / (nyn * np.linalg.norm(yn))  # xn
        n += 1
        print(n)
    l = np.linalg.norm(np.dot(A, xnp))
    pres = np.linalg.norm(np.dot(A, xnp) - l * xnp)
    return (l, xnp, n, pres)


def ProcheVP(A, x0, alpha, maxiter, epsilon):
    B=A-alpha*np.eye(3)
    xn = x0 / (np.linalg.norm(x0))  # x0
    yn = np.dot(np.linalg.inv(B), xn)  # y0
    nyn = np.linalg.norm(yn)
    xnp = yn / nyn  # x1
    xnpn = xnp / (np.linalg.norm(xnp))  # x1 normé
    n = 0  # compteur
    while (np.linalg.norm(xnpn - xn) <= epsilon) and (np.linalg.norm(xnpn + xn) <= epsilon) and n < maxiter:
        xn = xnpn  # xn-1
        yn = np.dot(np.linalg.inv(B), xn)  # yn-1
        nyn = np.linalg.norm(yn)  # norme yn-1
        xnp = yn / (nyn * np.linalg.norm(yn))  # xn
        n += 1
        print(n)
    l = np.linalg.norm(np.dot(B, xnp))
    pres = np.linalg.norm(np.dot(B, xnp) - l * xnp)
    return (l, xnp, n, pres)


if __name__ == '__main__':
    size = (3, 3)
    epsilon = float(input("Epsilon?"))
    A = np.ones(size)
    x0 = np.array([[10], [100], [-500]])
    A2 = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    maxiter = int(input("Max iter?"))
    l, x, n, pres = PuissanceInvVP(A2, x0, maxiter, epsilon)
    print("lambda = ", l)
    print("nbiter = ", n)
    print("précision = ", pres)
    print("x = ", x)
    alpha=2.05
    l, x, n, pres = ProcheVP(A2, x0, alpha, maxiter, epsilon)
    print("lambda = ", l)
    print("nbiter = ", n)
    print("précision = ", pres)
    print("x = ", x)
