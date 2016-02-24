from sympy import *
import numpy as np
from numpy.polynomial import Polynomial as Poly
import scipy.stats as reg
import matplotlib.pyplot as plt
from scipy.misc import pade

def Rmatrix(Wn) :
    """ Input  : list of coefficients Wn of an expansion W(U) = sum_n Wn U^n (n = 1,2,...M)
        Output : The corresponding M x M matrix called R matrix. When two series are combined, ZW(U) = Z(W(U)) the coefficients
        WZ_n are gicen by the matrix vector product  ZWn = R(Wn) Zn """
    M = len(Wn)
    R = np.zeros((M,M))
    R[:,0]=Wn[:]
    for j in range(1,M):
        for i in range(j,M):
            res = 0
            for k in range(i): res += Wn[k] *R[i-1-k,j-1]
            R[i,j] = res
    return R

def InverseTrans(A):
    """ Input : polynom
        Output : polynom B such that B(A) = A(B) = 1"""
    assert(abs(A(0))<1e-20)
    An = A.coef[1:]
    R = Rmatrix(An)
    y = [ 1 ] + [ 0 for i in range(len(An) - 1) ]
    x = np.linalg.solve(R,y)
    B = Poly([0] + x.tolist() )    
    return B


x = symbols('x')
class ConformalTransform:
    """ W(U)"""
    
    def __init__(self,sympy_expression,order): 
        """ Input : string sympy_expression of the form (x**2 +1)/(x-5). Should verify expr(x=0) = 0
                    order : number of terms of the expansion
            """ 
        self.expr = sympify(sympy_expression)
        self.order = order
        A = series(self.expr,x,n = self.order)
        self.Wn = Poly([N(A.coeff(x,i)) for i in range(self.order)])
        self.Un = InverseTrans(self.Wn)
    def __call__(self,U): 
        return N(self.expr.subs(x,U))
    def Transform(self,series): 
        return series(self.Un).truncate(self.order)

def ConvRadius(P,plot="False"):
    PRECISION = 1e-15
    Y = P.coef.tolist()
    N=len(Y)
    X = range(N)
    for i in range(N):
        if np.abs(Y[N-1-i]) < PRECISION:
            del X[N-1-i]
            del Y[N-1-i]
    del X[0]
    del Y[0]
    Y = np.log(np.abs(np.array(Y)))
    X = np.array(X)
    slope, intercept, r_value, p_value, std_err = reg.linregress(X,Y)
    RES = np.exp(-slope)
    if plot == "True":
        plt.plot(X,Y,'o')
        plt.plot(X,slope*X+intercept)
        plt.xlabel('$n$')
        plt.ylabel('$\log Q_n$')
        plt.title('Convergence radius of a series $\sum_n Q_n U^n$ = '+str(RES))
        plt.show()
    return RES

def OneOverSerie(P,N = -1):
    """ Input : polynom P
                order of the inverse N (default : same as order of P)
        Output : polynom Q which is the expansion of 1/P provided the zeroth order term of P does not vanish."""
    Qs = P.coef.tolist()
    if N == -1: N = len(Qs)
    if N > len(Qs): Qs += (N-len(Qs))*[0] 
    Fs=[1./Qs[0]]
    for i,Q in enumerate(Qs[1:]):
        Fi=0
        for j in xrange(1,i+2): Fi += Qs[j]*Fs[i+1-j]
        Fs.append(-Fs[0]*Fi)
    return Poly(Fs)

def Pade(series,denom,plot = 'False',myaxis = (-10,10,-10,10),zeros = 'False'):
    """ Calculate the pade of a series with denom term in the denominator - plot the corresponding 
    singularities of the denominator """
    p,q = pade(series.coef,denom)
    a = np.roots(q.coef)
    xa = [i.real for i in a]
    ya = [i.imag for i in a]
    if plot == 'True':
       plt.plot(xa,ya,'s',label ='singularities')
       plt.plot([0],[0],'o')
       if zeros == 'True':
           a = np.roots(p.coef)
           xa_p = [i.real for i in a]
           ya_p = [i.imag for i in a]
           plt.plot(xa_p,ya_p,'s',label = 'zeros')
       plt.axis(myaxis)
       plt.legend()
       plt.title('Singularity analysis by Pade')
       plt.xlabel('Re U')
       plt.ylabel('Im U')
    return p,q,xa,ya

def Dlog(P):
    """ Calculate the derivative of the log of a series """
    Q = OneOverSerie(P)
    R = Poly(np.polynomial.polynomial.polyder(P.coef))
    return (Q*R).truncate(len(P.coef))


#fig,ax1,ax2 = OneParameterTransform(In,"x / ( (%s*sin(b)**2 + (x - %s*cos(b))**2 )**(1/2))" % (1.3,1.3),
#                                    np.linspace(1.,3.1,15), U = 7, myaxis = (0,3.14,0,0.05)) 
#fig.show()
def OneParameterTransform(series,transform,bs,U,myaxis = -1):
    """ Input : the series to be transform, a string for the transformation and the list of
    the parameter of the serie. the transformation should have a syntax of the form 
    x / ((x + b)**2 + 1)**(1/2) where x is the variable on the complex plane and b a parameter"""
    ys,zs = [],[]
    N = len(series.coef)
    print "# parameter - radius - target W(U) - Final value"
    for b in bs:
        T = ConformalTransform(transform.replace("b",str(b)),N)
        A = T.Transform(series)
        R = ConvRadius(A)
        ys.append(R/T(U))
        zs.append(A(T(U)))
        print b,R,T(U),A(T(U))
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot(bs,ys)
    ax2.plot(bs,zs,lw=2)
    ax1.plot(bs,1.+0*bs,"--")
    if myaxis != -1 :ax2.axis(myaxis)
    ax1.legend()
    ax2.set_xlabel("b")
    ax1.set_ylabel("Figure of merit")
    ax2.set_ylabel("Value")
    ax1.set_title("ConformalTransform : "+transform)
    return fig,ax1,ax2

if __name__ == "__main__":

    # TEST THAT THE R MATRIX CAN INDEED BE USED TO COMBINE TWO SERIES.
    # COMPARE THE R MATRIX IMPLEMENTATION WITH A COMBINATION OF TWO POLYNOMS.
    Wa = [0,1,2,3,4,5,5]
    Wb = [0,5,4,5,4,5,7]
    A = Poly(Wa)
    B = Poly(Wb)
    print " Two polynoms A and B, B(A) = ",B(A).truncate(len(A))
    A = Rmatrix(Wa[1:])
    B = np.array(Wb[1:])
    print " R matrix implementation of the same operation ",np.dot(A,B)
    
    # TEST THE INVERSE OF A SERIES
    A = Poly(Wa) 
    print " Test of the inverse B of A such that B(A) = A(B) = X "
    print InverseTrans(A)(A).truncate(A.degree()+1)
    print A(InverseTrans(A)).truncate(A.degree()+1)
    
    # TEST CONFORMAL TRANSFORM
    a = " x**2 + %s * x / ( 1 + %s * x**2)" % (1,0.5)
    T = ConformalTransform(a,10)
    init_printing(use_latex = 'mathjax')
    print T(0.),T(1.),T(2.)
    print T.Transform(A)
    
