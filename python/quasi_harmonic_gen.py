import numpy as np

def phi(d):
    """
    Evaluate the d-th Harmonious number phi(d) to longfloat precision.
    """
    if d == 1:
        return (np.sqrt(np.longfloat(5.0))+1.0)/2.0
    else:
        x = np.longfloat(2.0)
        for i in range(50):
            xprev = x
            x = np.power(1.0+x,1.0/(d+1))
            if x == xprev:
                break
        return x

class HARQuasirand():
    """
    d-dimensional Additive (Kronecker) Recurrence quasi-random number generator
    based on the Harmonious numbers phi(d).   
    
    Additive Recurrence generators produce outputs in (0,1) of the form 
    (shift+count*alpha) mod 1, where alpha is an irrational constant and shift 
    is an optional fractional shift. They are very fast and (only!) for 
    well-chosen alpha they are among the very best of the currently known 
    quasi-random number generators by uniformity / dispersion measures. 
      
    The "Harmonious numbers" phi(d) are the unique positive real solutions of
    x**(d+1)=x+1 or the corresponding continued root expansion
    [x]=(1+[x])**1/(d+1). They have many elegant number-theoretic properties, 
    with each phi(d) having deep links to d+1 dimensional geometric recursion. 
    phi(1) is the Golden Ratio (sqrt(5)+1)/2 and phi(2) is the so-called 
    "plastic number". phi(d),d>3 has no algebriac form but it is easily 
    estimated by continued root iteration.
    
    Multi-dimensional quasi-random number generation is intrinsically difficult
    and all of the known approaches tend to produce samples that lie close to 
    specific hyperplanes or other submanifolds during at least some finite 
    intervals of their cycles. Additive Recurrence generators are prone to
    this if their alpha values are chosen poorly, but very resistant to it with
    well-chosen alpha's. Empirically the choice
    alpha[i] = (phi(d))**-i for 1<=i<=d seems to be an unusually good one and 
    it is even conjectured [1,2] to be (in some sense to be defined) the best 
    possible.
    
    Here we use the Harmonious alpha[:] values by default. We don't use a shift
    but we do seed the count to a large value. Additive Recurrence generators 
    have the weakness that for small count values their modulo operations wrap
    at predictable times so each output dimension scans repetitively across its
    range as the count changes, and also pairs of dimensions tend to scan 
    together. Users who like this "quasi-regular" initial sample behaviour 
    can seed with count=1. Otherwise, to ensure that the modulo operations wrap
    at incoherent times seed the count to at least 10-100 times 1/gap where 
    gap = diff(sort([0,alpha[:],1])) is the smallest distance between (the 
    fractional parts of) any two alpha[:] values including 0,1. The gap is
    ~ 1/(2*dim) for the Harmonious alpha's.
    
    We use long double precision for alpha calculations to reduce the loss of
    randomness in the low-order (double float) output bits at large count 
    values. The generator cycle time is that of its ulonglong counter. 
   
    [1] Random walks with badly approximable numbers, D.Hensley and F.E.Su,
        DIMACS Unusual Applications of Number Theory, 2002.
    [2] http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
        Martin Roberts, May 2018.
    
    """
    def __init__(self,dim=1,seed=1<<30):
        """
        qh = QH(d=1, seed=1<<30) creates a generator for d-dimensional 
        quasi-random vectors based on Harmonious Number Additive Recurrence.    
        """
        self.dim = dim
        self.count = np.ulonglong(seed)
        self.alpha = np.zeros((dim),dtype=np.longfloat)
        h = phi(dim)
        for i in range(dim):
            self.alpha[i] = np.power(h,-(i+1))
        
    def rand(self,n=1,out=None):
        """
        out = qh.rand(n=1, out=None) creates/fills out=array((n,d),float) with
        n quasi-random d-vectors in the unit interval (0,1). 
        If input out is given, its n value is used and input n is ignored.
        """
        if out == None:
            out = np.zeros((n,self.dim),dtype=np.float)
        else:
            n = np.shape(out)[:-2]
        for i in range(n):
            out[i,:] = (self.count * self.alpha) %1
            self.count = self.count+1
        return out

    def rand_fast(self, n=1):
        counts = np.arange(self.count, self.count+n,1)
        self.count += n
        return np.outer(counts, self.alpha) % 1
 

