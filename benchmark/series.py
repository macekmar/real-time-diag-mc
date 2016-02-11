import numpy as np
from numpy.polynomial import Polynomial as Poly
import matplotlib.pyplot as plt
#from scipy import optimize
import scipy.stats as reg
    
def composition_set(n):
    """
    compositions are slightly different from partitions as 
    (2,2,1) is considered as different from (1,2,2)
    
    Parameters
    ----------
    n : number to be compositionned    
    
    Examples
    --------
    >>> from series import composition_set
    >>> composition_set(4)
    [[4], [1, 3], [1, 1, 2], [1, 1, 1, 1], [1, 2, 1], [2, 2], [2, 1, 1], [3, 1]]
    
    """
    answer = list([])
    answer.append([n])
    for x in range(1, n):
        for y in composition_set(n - x):
             answer.append([x] + y)
    return answer

def InverseSerie(P,N):
    """ Given the N first terms a_n of a serie p(z) = sum_n a_nz^n, 
    return N terms for the inverse Q such that Q(P(z)) = z
    
    Parameters
    ----------
    P : serie under the form of a polynom.
    N : order desired for the return 
    
    Returns
    -------
    Q : the inverse Q such that Q(P(z)) = z
    
    Examples
    --------
    >>> from series import InverseSerie
    >>> P=Poly([0,1,-3,1,-6])
    >>> Q=InverseSerie(P,P.degree())
    >>> R=InverseSerie(Q,Q.degree())
    >>> print P
    poly([ 0.  1. -3.  1. -6.])
    >>> print Q
    poly([   0.    1.    3.   17.  126.])
    >>> print R
    poly([ 0.  1. -3.  1. -6.])
    """
    if P.degree()<1:
        print 'reciprocal function undefined'
        return Poly([0])
    if P.coef[0]!=0:
        print 'error, P(0)!=0'
        return Poly([0])
    if P.coef[1]==0:
        print 'error, P[1]=0'
        return Poly([0])
    Q = list([0,1./P.coef[1]])
    for i in range(2,N+1):
        res = 0 # at the end, =-d_i*c_1**i
        comp = composition_set(i)
        comp.remove([1 for k in range(i)])
        for p in comp:
            r2 = 1.
            for k in p:
                if k<P.degree()+1:
                    r2 *= P.coef[k]
                else : 
                    r2 = 0
                    break
            r2 *= Q[len(p)]
            res += r2
        Q.append(-res/P.coef[1]**i)
    return Poly(Q)
    
#NOT VERIFIED !!!!!!!!!
def Analytic_composition(S,w):
    """ 
    Parameters
    ----------
    S : polynom of a serie, 
    w : polynom of the change of variable
    
    Returns
    -------
    S' : such that S'(w(z))=S(z)
    
    """
    N = S.degree() #degree of the result
    P = InverseSerie(w,N)
    if P.degree()<1:
        print 'change of variable is constant'
        return Poly([0])
    if P.coef[0]!=0:
        print 'error, w(0)!=0'
        return Poly([0])
    if P.coef[1]==0:
        print 'error, w[1]=0'
        return Poly([0])
    Q = list([0,1./P.coef[1]])
    for i in range(2,N+1):
        res = 0 # at the end, =-d_i*c_1**i
        part = composition_set(i)
        part.remove([1 for k in range(i)])
        for p in part:
            r2 = 1.
            for k in p:
                r2 *= P.coef[k]
            r2 *= S.coef[len(p)]
            res += r2
        Q.append(res)
    return Poly(Q)

def ConvergenceRadius(P,if_plot=False, nmin=0):
    """ 
    Parameters
    ----------
    P : polynom, 
    plot : True to trace the fit
    nmin : minimal order of P's coefficients taken into account
    
    Returns
    -------
    rc : the approximated convergence radius
    
    """
    if (nmin>P.degree()):
        print "empty polynom, nmin>P.degree"
        return 0
    PRECISION = 1e-15
    Y = P.coef.tolist()
    X = range(len(Y))
    for i in range(len(Y)-1,nmin-1,-1):
        if np.abs(Y[i]) < PRECISION:
            del X[i]
            del Y[i]
    del X[0:nmin]
    del Y[0:nmin]
    Y = np.log(np.abs(np.array(Y)))
    X = np.array(X)
    slope, intercept, r_value, p_value, std_err = reg.linregress(X,Y)
    if if_plot:
        plt.plot(X,Y,'o')
        plt.xlabel(r'$n$')
        plt.ylabel(r'$\ln |P_n|$')
        plt.plot(X,slope*X+intercept)
        plt.show()
    return np.exp(-slope)
 
class Homographic(object):
    """ Given a,b,c,d construct (az+b)/(cz+d) """
    def __init__(self,a,b,c,d):
        self.M = np.array([a,b,c,d])
    def __call__(self,z):
        return ( self.M[0]* z + self.M[1] )/( self.M[2]*z + self.M[3])
    def __repr__(self):
        return '(%sz + %s )/(%sz + %s )' % (self.M[0], self.M[1], self.M[2], self.M[3])
    def inv(self):
        """returns the inverse, d,-b,-c,a"""
        return Homographic(self.M[3] , - self.M[1] , - self.M[2] , self.M[0])
    def coefs(self):
        """returns a,b,c,d"""
        return self.M[0],self.M[1],self.M[2],self.M[3]
    def taylor(self,n):
        """order of the serie"""
        a,b,c,d = self.coefs()
        if d==0:
            print "taylor serie undefined, d=0"
            return Poly([])
        d=float(d)
        P=list([b/d])
        fact = (a*d-b*c) / (d*d)
        for i in range(1,n+1):
            P.append(fact)
            fact *= -c/d
            #print 'fact=',fact
        return Poly(P);    

def prettyprint(P):
    """ Provide a slightly better printing of a polynom """
    res = ""
    nonzero = False
    for i,a in enumerate(P.coef):
        if a!= 0:
          if nonzero and a>0: res += "+"
          nonzero = True
          res += str(a)
          if i == 1: res+=" X"
          if i > 1: res+=" X^"+str(i)
          res += " "
    if not nonzero: print 0.0
    else: print res

def QuotientSerie(P,N):
    """ Given the N first terms a_n of a serie p(z)=sum_n a_nz^n, 
    return N terms for the quotient Q such that Q(z) = 1/P(z)
    provided the zeroth order term does not vanish.
    
    P: polynom,
    
    returns a polynom"""
    Q = P.copy().cutdeg(N)
    a0 = float(Q.coef[0])
    if a0 == 0:
        print 'P(0)=0: inverse undefined at zero'
        return Poly([0])
    Q.coef[0] = 0
    Q.coef /= a0 # an/a0 for n>0
    F = Poly([1]) # we divide by Qs[0] later
    Ptemp = -Q.copy()
    for i in range(1,N):
        F += Ptemp
        Ptemp = (-Q*Ptemp).cutdeg(N)
    F += Ptemp
    return F/a0

def TransformPoints(x,y,f):
    """ given points in the plane (x,y) return the point (f(x+iy).real,f(x+iy).imag) """
    z=[u+1j*v for (u,v) in zip (x,y)]
    z=[f(u) for u in z]
    xout=[u.real for u in z]
    yout=[u.imag for u in z]
    return xout,yout 
    
#NOT VERIFIED  !!!!!!!!!
def HomographicTransform(P,h):
    """ Input: P: polynomial, h homographic transform.
        Output: Serie of P(h(X)) """
    w = h.taylor(P.degree())
    return Analytic_composition(P,w)
    
def serie_colorplot(P, Nmax, R=2, npts=25, vmax=2):
    """draw a colorplot corresponding to the polynom P values (real part) at order Nmax 
    for a square of size R in the complex plane
    npts: number of points on a side of the square 
    vmax: max value of the colorbar
    """
    res = np.empty((npts,npts))
    x = np.linspace(-R,R,npts)
    z = np.array([ [x[i]+ 1j * x[j] for j in range(npts)] for i in range(npts)])
    res = ((P.cutdeg(Nmax))(z).real)
    plt.imshow(res, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.show()
    
def plot_quality(oP, legend, U_wanted, amin=-1, amax=1, ymax=3, npts=100): 
    """ oP: array of series
    legend: description of the series
    amin, amax : range of values for the homography
    ymax
    npts: number of different a's"""
    A = np.linspace(amin,amax,npts)
    plt.figure(figsize = [12, 9])
    for i in range(len(oP)):
          P = Poly(oP[i])
          R = []
          for a in A:
                H = Homographic(a,0,1,1).inv()
                S = HomographicTransform(P,H)
                R.append( ConvergenceRadius(S)/abs(H(U_wanted)))
          plt.plot(A, R, label=legend[i])
    plt.xlabel('a')
    plt.ylabel('quality factor')
    plt.plot([amin,amax],[1,1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim([0,ymax])
    plt.show()
                
if __name__ == "__main__":
    
    #example of composition_sets
    print "composition_sets of 4"
    print composition_set(4)
    print

    #verification of the inverse of a serie
    P=Poly([0,1,-3,1,-6])
    print "both lines should be equal (InverseSerie)"
    print 'P = ',P
    print 'P = ',InverseSerie(InverseSerie(P,P.degree()),P.degree())
    print
    print "identity (QuotientSerie)"
    P=Poly([1,1,-3,1,-6])
    print 'P*1/P=',P*QuotientSerie(P,P.degree())
    print 
    a=-1.; b=0.; c=2.; d=1.2
    h = Homographic(a,b,c,d)
    k = Homographic(-d,b,c,-a)
    print "both lines should be equal (InverseSerie)"
    print 'k=', k.taylor(5)
    print 'k=', InverseSerie(h.taylor(5),5)
    print
    a=0; b=1; c=1.; d=1.
    h = Homographic(a,b,c,d) # h = 1/(z+1)
    k = Homographic(c,d,a,b) # k = z+1
    print "both lines should be equal (QuotientSerie)"
    print 'h=', h.taylor(5)
    print 'h=', QuotientSerie(k.taylor(5),5)
    print
    print "both lines should be equal (QuotientSerie)"
    print 'k=', k.taylor(5)
    print 'k=', QuotientSerie(h.taylor(5),5)
    print 
    
    print "testing holographic transform and its inverse"
    A = Homographic(2,1,3,4)
    z=0+2.3j
    print 'z=',z
    w = A(z)
    B = A.inv()
    print 'z=',B(w)
    
    P=Poly([1,0,3,2])
    amin = -10
    amax = 10
    n = P.degree()
    A = np.linspace(amin,amax,200)
    b=0
    c=1
    d=1
    h = [Homographic(aa,b,c,d).taylor(n) for aa in A]
    r = ConvergenceRadius(h, if_plot=False)

          
