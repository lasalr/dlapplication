import math
import random
import numpy as np

EPS = 0.000001
MAX_REL_EPS = 0.0001


def floatApproxEqual(x, y):
    if x == y:
        return True
    relError = 0.0
    if (abs(y) > abs(x)):
        relError = abs((x - y) / y);
    else:
        relError = abs((x - y) / x);
    if relError <= MAX_REL_EPS:
        return True
    if abs(x - y) <= EPS:
        return True
    return False


def getRadonPoint(S, r):
    alpha = []
    A = np.vstack((np.transpose(S), np.ones(S.shape[0])))
    z = np.zeros(S.shape[0])
    z[0] = 1.0
    A = np.vstack((A, z))
    b = np.zeros(S.shape[0])
    b[-1] = 1.0
    alpha = np.linalg.lstsq(A, b)[0]
    alpha_plus = np.zeros(len(alpha))
    alpha_minus = np.zeros(len(alpha))
    for i in range(len(alpha)):
        if alpha[i] > 0:
            alpha_plus[i] = alpha[i]
        if alpha[i] < 0:
            alpha_minus[i] = alpha[i]
    sumAlpha_plus = 1. * np.sum(alpha_plus)
    sumAlpha_minus = -1. * np.sum(alpha_minus)
    if not floatApproxEqual(sumAlpha_plus, sumAlpha_minus):
        print("Error: sum(a+) != sum(a-): " + str(sumAlpha_plus) + " != " + str(sumAlpha_minus) + " for |S| = " + str(
            S.shape) + " and R = " + str(r))
    alpha /= sumAlpha_plus
    r = np.zeros(S.shape[1])
    r_minus = np.zeros(S.shape[1])
    for i in range(len(alpha)):
        if alpha[i] > 0:
            r += alpha[i] * S[i]
        if alpha[i] < 0:
            r_minus += alpha[i] * S[i]
    rtest_plus = r * 1. / np.linalg.norm(r)  # normiert
    rtest_minus = r_minus * 1. / np.linalg.norm(r_minus)  # normiert
    if np.linalg.norm(rtest_plus + rtest_minus) > EPS:
        print("Something went wrong!!! r+ = " + str(r) + " but r- = " + str(-1 * r_minus) +
              ". They should be the same!")
    return r

def getRadonPointHierarchical(vectors):
    n, d = vectors.shape
    r = d + 2
    h = math.floor(math.log(n, r))
    S = np.array(random.choices(vectors.tolist(), k=r ** h))
    print(n, d, r, h, r ** h, S.shape)
    while S.shape[0] >= r:
        S_new = []
        print(S.shape[0] / r)
        for i in range(S.shape[0] // r):
            v = getRadonPoint(S[i * r:(i + 1) * r], r)
            S_new.append(v)
        S = np.array(S_new)
    if S.shape[0] > 1:
        print("Error: too few instances in S for radon point calculation! |S| = " + str(
            S.shape[0]) + " for radon number R = " + str(r) + " .")
    return S[0]
