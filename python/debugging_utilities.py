import numpy as np

class G0Keldysh :

    def __init__(self, g0_lesser, g0_greater, alpha, tmax):
        self.g0_lesser = g0_lesser
        self.g0_greater = g0_greater
        self.alpha = alpha
        self.tmax = tmax

    def __call__(self, tau1, tau2):

        s1, u1, a1 = tau1
        s2, u2, a2 = tau2
        
        if s1 != s2:
            return 0.0
        
        if u1 == u2 and a1 == a2:
            if u1 == self.tmax:
                return self.g0_lesser(u1 - u2)[0, 0]
            else:
                return self.g0_lesser(u1 - u2)[0, 0] - 1j * self.alpha
        
        if a1 == a2:
            is_greater = (a1 != (u1 > u2))
        else:
            is_greater = a1
            
        if is_greater:
            return self.g0_greater(u1 - u2)[0, 0]
        else:
            return self.g0_lesser(u1 - u2)[0, 0]

def matrix(function, list1, list2):
    n = len(list1)
    output = np.zeros((n, n), dtype=complex)
    for i, x in enumerate(list1):
        for j, y in enumerate(list2):
            output[i, j] = function(x, y)
            
    return output

def keldyshSumDetM1(g0, tau, taup, u):
    output = 0.0
    for a in [0, 1]:

        list1 = [[0, u, a], [1, u, a], tau]
        list2 = [[0, u, a], [1, u, a], taup]

        M = matrix(g0, list1, list2)
        output += pow(-1, a) * np.linalg.det(M)
    return output

def keldyshSumDetM2(g0, tau, taup, u1, u2):
    output = 0.0
    for a1 in [0, 1]:
        for a2 in [0, 1]:

            list1 = [[0, u1, a1], [1, u1, a1], [0, u2, a2], [1, u2, a2], tau]
            list2 = [[0, u1, a1], [1, u1, a1], [0, u2, a2], [1, u2, a2], taup]

            M = matrix(g0, list1, list2)
            output += pow(-1, a1+a2) * np.linalg.det(M)
    return output

