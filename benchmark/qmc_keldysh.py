from pytriqs.applications.impurity_solvers.ctint_new import CtintSolver
from pytriqs.utility import mpi
from pytriqs.archive import HDFArchive
import numpy as np
                    
class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
       self.__dict__.update(kwargs)
       
def adjust_U(S, qmc_param, phys_param, Nmax, Nmin):
    """
    Find the optimal U by running some small simulations (100 max). 
    Equilibrates the two largest probabilities to minimize the error
    (can be ameliorated)
    
    Parameters
    ----------
    S : solver CT-int-new
    phys_param : contains beta, Gamma, epsilon_d, muL, muR, GF_type, tmax, alpha
    qmc_params : contains U, is_current, Last_order, n_cycles, 
                 n_warmup_cycles, length_cycle, p_dbl, random_seed
    Nmax : maximal order
    Nmin : minimal order 
    
    """
    pmin = 0.35
    pmax = 0.65
    Umin = 0.
    Umax = 1000.
    p = 0.
    count = 0
    while ( (p>pmax or p<pmin) and count<100):
        S.solve(
            U = qmc_param.U_qmc,  
            is_current = phys_param.is_current, 
            max_perturbation_order = Nmax, 
            min_perturbation_order = Nmin, 
            tmax = phys_param.tmax, 
            alpha = phys_param.alpha,
            verbosity = 0,
            n_cycles = max([100,qmc_param.n_cycles/100]),
            n_warmup_cycles = qmc_param.n_warmup_cycles, 
            length_cycle = qmc_param.length_cycle, 
            p_dbl = qmc_param.p_dbl,
            random_seed = qmc_param.random_seed+count+1)
        # we want that the last proba and the second larger proba are ~identical
        p = S.CnSn[0,Nmax]
        p2 = max(S.CnSn[0,:Nmax])
        order2 = S.CnSn[0,:Nmax].argmax()
        p = p/(p+p2)
        oldU = qmc_param.U_qmc
        # update of the boundaries
        if(p>pmax):
            Umax = min([Umax,qmc_param.U_qmc])
        elif(p<pmin):
            Umin = max([Umin,qmc_param.U_qmc])
        #extreme situations
        if p==0:
            qmc_param.U_qmc *=2.
        elif p==1:
            qmc_param.U_qmc /=2.
        #normal situation
        elif(p>pmax or p<pmin):
            qmc_param.U_qmc *= pow((1.-p)/p,1./(Nmax-order2))
        #if qmc_param.U_qmc>Umax:
        #    qmc_param.U_qmc=0.5*(Umax+oldU)
        elif qmc_param.U_qmc<Umin:
            qmc_param.U_qmc=0.5*(Umin+oldU)
        count+=1
    print 'Nmax=',Nmax,' final newU=',qmc_param.U_qmc

def QMC(phys_param, 
        qmc_param, 
        if_adjust_U = True, 
        verbosity = 3, 
        if_ph = False):
    """
    Designed to use the recurrence relation between n and n+?
    
    Parameters
    ----------
    phys_param : contains beta, Gamma, epsilon_d, muL, muR, GF_type, tmax, alpha
    qmc_params : contains U, is_current, Last_order, n_cycles, n_warmup_cycles, 
      length_cycle, random_seed,
    if_adjust_U
    verbosity : as it says
    if_ph : if True, only even orders are computed (ph = particule-hole symmetric)
    
    """
    S = CtintSolver(beta = phys_param.beta, 
           Gamma = phys_param.Gamma, 
           epsilon_d = phys_param.epsilon_d, 
           muL = phys_param.muL, 
           muR = phys_param.muR, 
           GF_type = phys_param.GF_type)
 
    # value of the observable in the non interacting case
    if phys_param.is_current:
        c_norm = S.I0_L # current
        print "Non interacting current : ", c_norm
    else :
        c_norm = S.c0 # density
        print "Non interacting charge : ", c_norm
    # terms of the series for different orders in U (for instance, order 0)
    pn_last = list([1.])           # proba to be in order n for each order
    sn_last = list([np.sign(c_norm)]) # average sign for each order
    cn_last = list([np.abs(c_norm)])  # sum(abs(M(cn)))
    on_last = list([c_norm])       # On=cn*sn
    
    Nmax_step = 1   # Nmax_step is the increment between two successive interesting orders. 
                    # Nmax_step is 1 or 2 (case of only even orders)
    delta_order = 1 # One simulation can run over one or two orders
    if( (not if_ph) and qmc_param.p_dbl>0.8):
        print "incoherent parameters in QMC: if_ph=False and p_dbl=",qmc_param.p_dbl
        return None
    if(if_ph):
        Nmax_step = 2
        qmc_param.p_dbl = 1
    if(qmc_param.p_dbl>0):
        delta_order = 2
    for Nmax in range(Nmax_step, qmc_param.LastOrder+1, Nmax_step) :
        if(Nmax_step==2):
          pn_last.append(0)
          sn_last.append(0)
          cn_last.append(0)
          on_last.append(0)
        Nmin = max(Nmax-delta_order,0)
        if mpi.rank == 0 and verbosity>0 :
            print "--------- Nmax = ",Nmax," Nmin = ",Nmin
        if(if_adjust_U):
            adjust_U(S, qmc_param, phys_param, Nmax, Nmin)
        S.solve(
            U = qmc_param.U_qmc,  
            is_current = phys_param.is_current, 
            max_perturbation_order = Nmax, 
            min_perturbation_order = Nmin, 
            tmax = phys_param.tmax, 
            alpha = phys_param.alpha,
            verbosity = verbosity, 
            n_cycles = qmc_param.n_cycles,
            n_warmup_cycles = qmc_param.n_warmup_cycles, 
            length_cycle = qmc_param.length_cycle, 
            p_dbl = qmc_param.p_dbl,
            random_seed = qmc_param.random_seed)
        # Bare cn,sn coming from the simulations with an extra factor (U_qmc)**n and 
        pn = S.CnSn[0,0:Nmax+1] # proba to be in order n for each order
        if(pn[-1]==1 or pn[-1]==0):
             print 'bad Uqmc: pn[Nmax]=',pn[-1]
             return 0,0,0,0
        sn = S.CnSn[1,0:Nmax+1] # average sign for each order
        sn_last.append(S.CnSn[1,Nmax])
        # c_(n+1)/c_n = 1 / Uqmc * p_(n+1) / p_n
        # c_(n+k)/c_n = 1 / Uqmc**k * p_(n+1)/p_(n+1-k)
        delta_order2 = Nmax - S.CnSn[0,:Nmax].argmax() # max of all except the last 
        if(verbosity>2):
            print "for Nmax=",Nmax,", delta_order2=",delta_order2
        #new_cn = cn_last[-1] / float( qmc_param.U_qmc )  *pn[-1] / pn[-2] 
        new_cn = cn_last[-delta_order2] / float( qmc_param.U_qmc**delta_order2 ) *pn[-1] / pn[-1-delta_order2] 
        cn_last.append(new_cn)
        on_last.append(new_cn*sn[-1])
        pn_last.append(pn[-1])
        if(verbosity>2):
            print " cn = ", cn_last
            print " sn = ", sn_last
            print " on = ", on_last
            print " pn = ", pn_last
    del S
    return cn_last, sn_last, on_last, pn_last
    
def save(p,p_qmc,res, filename):
    if(res==None):
        print "saving is impossible: no results"
        return
    R = HDFArchive(filename, 'w')
    R['alpha'] = p.alpha
    R['Gamma'] = p.Gamma
    R['epsilon_d'] = p.epsilon_d
    R['V'] = p.muL - p.muR
    R['GF_type'] = p.GF_type
    R['muL'] = p.muL
    R['muR'] = p.muR
    R['beta'] = p.beta
    R['GF_type'] = p.GF_type
    R['is_current'] = p.is_current
    R['tmax'] = p.tmax
    R['U_qmc'] = p_qmc.U_qmc
    R['length_cycle'] = p_qmc.length_cycle
    R['n_cycles'] = p_qmc.n_cycles
    R['n_warmup_cycles'] = p_qmc.n_warmup_cycles
    R['cn'] = res[0]
    R['sn'] = res[1]
    R['on'] = res[2]
    R['pn'] = res[3]
    R['random_seed'] = p_qmc.random_seed
    del R
    
def QMC_different_V(phys_param,
                    qmc_param,
                    tab_V, 
                    if_adjust_U=True, 
                    verbosity=3, 
                    if_ph=False):
    """
    phys_param contains beta, Gamma, epsilon_d, muL, muR, GF_type, tmax, alpha
    qmc_params contains U, is_current, Last_order, n_cycles, n_warmup_cycles, length_cycle
    tab_tmax is a table of tmax
    """
    table_pn=list([])
    table_sn=list([])
    table_cn=list([])
    table_on=list([])
    table_Un=list([])
    # one simulation with terms from n=0 to n=Nmax, with varying Nmax
    for i in range(len(tab_V)):
         phys_param.muL =  tab_V[i]*0.5
         phys_param.muR = -tab_V[i]*0.5
         print 'V=',phys_param.muL*2
         cn_last, sn_last, on_last, pn_last = QMC(phys_param, qmc_param, 
                                                  if_adjust_U, verbosity,
                                                  if_ph)       
         table_cn.append(cn_last)
         table_sn.append(sn_last)
         table_on.append(on_last)
         table_pn.append(pn_last)
         table_Un.append(qmc_param.U_qmc)
    return table_cn, table_sn, table_on, table_pn, table_Un
   
def save_different_V(p,p_qmc,res, filename, V_tab):
    if(res==None):
        print "saving is impossible: no results"
        return
    R = HDFArchive(filename, 'w')
    R['alpha'] = p.alpha
    R['Gamma'] = p.Gamma
    R['epsilon_d'] = p.epsilon_d
    R['GF_type'] = p.GF_type
    R['V'] = p.muL - p.muR
    R['muL'] = p.muL
    R['muR'] = p.muR
    R['beta'] = p.beta
    R['is_current'] = p.is_current
    R['tmax'] = p.tmax
    R['V_tab'] = V_tab
    R['length_cycle'] = p_qmc.length_cycle
    R['n_cycles'] = p_qmc.n_cycles
    R['n_warmup_cycles'] = p_qmc.n_warmup_cycles
    R['cn'] = res[0]
    R['sn'] = res[1]
    R['on'] = res[2]
    R['pn'] = res[3]
    R['U_qmc'] = res[4]
    R['random_seed'] = p_qmc.random_seed
    del R
    
    
def QMC_different_eps_d(phys_param,
                    qmc_param,
                    tab_epsd, 
                    if_adjust_U=True, 
                    verbosity=3, 
                    if_ph=False):
    """
    phys_param contains beta, Gamma, epsilon_d, muL, muR, GF_type, tmax, alpha
    qmc_params contains U, is_current, Last_order, n_cycles, n_warmup_cycles, length_cycle
    tab_tmax is a table of tmax
    """
    table_pn=list([])
    table_sn=list([])
    table_cn=list([])
    table_on=list([])
    table_Un=list([])
    # one simulation with terms from n=0 to n=Nmax, with varying Nmax
    for i in range(len(tab_epsd)):
         phys_param.epsilon_d =  tab_epsd[i]
         print 'epsilon_d=',tab_epsd[i]
         cn_last, sn_last, on_last, pn_last = QMC(phys_param, qmc_param, 
                                                  if_adjust_U, verbosity,
                                                  if_ph)       
         table_cn.append(cn_last)
         table_sn.append(sn_last)
         table_on.append(on_last)
         table_pn.append(pn_last)
         table_Un.append(qmc_param.U_qmc)
    return table_cn, table_sn, table_on, table_pn, table_Un
   
def save_different_epsd(p,p_qmc,res, filename, tab_epsd):
    if(res==None):
        print "saving is impossible: no results"
        return
    R = HDFArchive(filename, 'w')
    R['alpha'] = p.alpha
    R['Gamma'] = p.Gamma
    R['GF_type'] = p.GF_type
    R['V'] = p.muL - p.muR
    R['muL'] = p.muL
    R['muR'] = p.muR
    R['beta'] = p.beta
    R['is_current'] = p.is_current
    R['tmax'] = p.tmax
    R['tab_epsd'] = tab_epsd
    R['length_cycle'] = p_qmc.length_cycle
    R['n_cycles'] = p_qmc.n_cycles
    R['n_warmup_cycles'] = p_qmc.n_warmup_cycles
    R['cn'] = res[0]
    R['sn'] = res[1]
    R['on'] = res[2]
    R['pn'] = res[3]
    R['U_qmc'] = res[4]
    R['random_seed'] = p_qmc.random_seed
    del R
    
def QMC_different_tmax(phys_param,
                       qmc_param,
                       tab_tmax, 
                       if_adjust_U=True, 
                       verbosity=3, 
                       if_ph=False):
    """
    phys_param contains beta, Gamma, epsilon_d, muL, muR, GF_type, tmax, alpha
    qmc_params contains U, is_current, Last_order, n_cycles, n_warmup_cycles, length_cycle
    tab_tmax is a table of tmax
    """
    table_pn=[]
    table_sn=[]
    table_cn=[]
    table_on=[]
    table_Un=[]
    for i in range(len(tab_tmax)):
         phys_param.tmax=tab_tmax[i]
         print 'tmax=',phys_param.tmax
         cn_last, sn_last, on_last, pn_last = QMC(phys_param, qmc_param, 
                                                  if_adjust_U, verbosity, if_ph)
         table_cn.append(cn_last)
         table_sn.append(sn_last)
         table_on.append(on_last)
         table_pn.append(pn_last)
         table_Un.append(qmc_param.U_qmc)
    return table_cn, table_sn, table_on, table_pn, table_Un
   
def save_different_tmax(p,p_qmc,res, filename, tmax_tab):
    if(res==None):
        print "saving is impossible: no results"
        return
    R = HDFArchive(filename, 'w')
    R['alpha'] = p.alpha
    R['Gamma'] = p.Gamma
    R['epsilon_d'] = p.epsilon_d
    R['GF_type'] = p.GF_type
    R['V'] = p.muL - p.muR
    R['muL'] = p.muL
    R['muR'] = p.muR
    R['beta'] = p.beta
    R['is_current'] = p.is_current
    R['tmax_tab'] = tmax_tab
    R['length_cycle'] = p_qmc.length_cycle
    R['n_cycles'] = p_qmc.n_cycles
    R['n_warmup_cycles'] = p_qmc.n_warmup_cycles
    R['cn'] = res[0]
    R['sn'] = res[1]
    R['on'] = res[2]
    R['pn'] = res[3]
    R['U_qmc'] = res[4]
    R['random_seed'] = p_qmc.random_seed
    del R
    
def QMC_different_seed(phys_param,
                       qmc_param,
                       tab_seed, 
                       if_adjust_U=True, 
                       verbosity=3, 
                       if_ph=False):
    """
    phys_param contains beta, Gamma, epsilon_d, muL, muR, GF_type, tmax, alpha
    qmc_params contains U, is_current, Last_order, n_cycles, n_warmup_cycles, length_cycle
    tab_tmax is a table of tmax
    """
    table_pn=[]
    table_sn=[]
    table_cn=[]
    table_on=[]
    table_Un=[]
    for i in range(len(tab_seed)):
         qmc_param.random_seed=tab_seed[i]
         print 'seed=',qmc_param.random_seed
         cn_last, sn_last, on_last, pn_last = QMC(phys_param, qmc_param, 
                                                  if_adjust_U, verbosity, if_ph)
         table_cn.append(cn_last)
         table_sn.append(sn_last)
         table_on.append(on_last)
         table_pn.append(pn_last)
         table_Un.append(qmc_param.U_qmc)
    return table_cn, table_sn, table_on, table_pn, table_Un
   
def save_different_seed(p,p_qmc,res, filename, tab_seed):
    if(res==None):
        print "saving is impossible: no results"
        return
    R = HDFArchive(filename, 'w')
    R['alpha'] = p.alpha
    R['Gamma'] = p.Gamma
    R['epsilon_d'] = p.epsilon_d
    R['GF_type'] = p.GF_type
    R['V'] = p.muL - p.muR
    R['muL'] = p.muL
    R['muR'] = p.muR
    R['beta'] = p.beta
    R['tmax'] = p.tmax
    R['is_current'] = p.is_current
    R['tab_seed'] = tab_seed
    R['length_cycle'] = p_qmc.length_cycle
    R['n_cycles'] = p_qmc.n_cycles
    R['n_warmup_cycles'] = p_qmc.n_warmup_cycles
    R['cn'] = res[0]
    R['sn'] = res[1]
    R['on'] = res[2]
    R['pn'] = res[3]
    R['U_qmc'] = res[4]
    del R
    
def print_info(h5_file):
    R = HDFArchive(h5_file, 'r')
    if ('alpha' in R): 
        print 'alpha =', R['alpha']
    if ('epsilon_d' in R): 
        print 'epsilon_d = ', R['epsilon_d']
    if ('Gamma' in R): 
        print 'Gamma = ', R['Gamma']
        
    if ('tmax_tab' in R):
        print 'tmax_tab = ', R['tmax_tab']
    elif ('tmax' in R):
        print 'tmax = ', R['tmax']
    else:
        print '!!! no tmax, no tmax_tab'
        
    if ('V_tab' in R):
      print 'V_tab = ',R['V_tab']
    elif ('V' in R): 
            print 'V = ', R['V']
    elif ('muL' in R and 'muR' in R):
            print 'muL = ', R['muL']
            print 'muR = ', R['muR']
        
    if ('beta' in R):
        print 'beta = ', R['beta']
    if ('is_current' in R):
        print 'is_current = ', R['is_current']
    if ('GF_type' in R):
        print 'GF_type = ', R['GF_type']
    if ('U_qmc' in R):
        print 'U_qmc = ', R['U_qmc']
    if ('n_cycles' in R): 
        print 'n_cycles = ', R['n_cycles']
    if ('n_warmup_cycles' in R):
        print 'n_warmup_cycles = ', R['n_warmup_cycles']
    if ('length_cycle' in R):
        print 'length_cycle = ', R['length_cycle'],
    if ('LastOrder' in R):
      print 'LastOrder = ', R['Nmax']
    if ('random_seed' in R):
      print 'random_seed = ', R['random_seed']
      