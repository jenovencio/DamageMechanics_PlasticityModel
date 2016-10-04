import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root
import matplotlib.pyplot as plt
from ipywidgets import widgets
from ipywidgets import *

%matplotlib inline
np.set_printoptions(precision=3, linewidth=300)



def Hookelaw(e,D,propmat):
    ''' Calculates de stress as
    sigma = (1-D)*E*e
    '''
    E = propmat.E
    sygma0 = propmat.sygma0
    M = propmat.M
    Kinf = propmat.Kinf
    b = propmat.b
    Rinf = propmat.Rinf
    S = propmat.S
    pd = propmat.pd
    
    s = (1.0 - D)*E*e
    
    return s
    
def yieldFunction(s,R,D,propmat):
    
    E = propmat.E
    sygma0 = propmat.sygma0
    M = propmat.M
    Kinf = propmat.Kinf
    b = propmat.b
    Rinf = propmat.Rinf
    S = propmat.S
    pd = propmat.pd
    S = propmat.S
    
    f = abs(s)/(1-D) - R - sygma0
    return f

def Heaviside(x):
    if x>0.0:
        return 1.0
    else:
        return 0.0
    

def elasticTrial(en1,an,propmat):
    ''' This functions get the total strain deformation at n+1 (en1) and the internal variables at n (an)
        and return the yield function trial, stress trial at n+1 (sn1) and the internal variables trial at n + 1
        
        where an = {epn,p,Rn,Xn,Dn}
    '''
    
    epn,pn,Rn,Dn = an
    
    # strain trial
    etrial = en1 - epn
    
    # stress trial
    strial = Hookelaw(etrial,Dn,propmat)
    
    # yield trial
    ftrial = yieldFunction(strial,Rn,Dn,propmat)
    
    an1trial = epn,pn,Rn,Dn
    
    return ftrial, strial, an1trial, etrial

class PropMat(object):
    def __init__(self, E, sygma0, M , Kinf, b, Rinf, S, pd, n):
        self.E = E
        self.sygma0 = sygma0
        self.M = M
        self.Kinf = Kinf
        self.b = b
        self.Rinf = Rinf
        self.S = S
        self.n = n
        self.pd = pd
        

def residualF(x0, propmat,dt):
    ''' The variables an must be created before call this function
    x0 = deltaL,sn1,Rn1,Xn1,Dn1
    '''
    
    # getting material properties
    E = propmat.E
    sygma0 = propmat.sygma0
    M = propmat.M
    Kinf = propmat.Kinf
    b = propmat.b
    Rinf = propmat.Rinf
    S = propmat.S
    n = propmat.n  
    pd = propmat.pd
    
    deltaL,sn1,Rn1,Dn1 = x0
    epn,pn,Rn,Dn = an
    fn1 = yieldFunction(sn1,Rn1,Dn1,propmat)
    en1 = etrial - (deltaL/(1.0-Dn1))*np.sign(sn1/(1.0-Dn1))
    Yn1 = sn1*sn1/(2.0*E*(1.0-Dn1)*(1.0-Dn1))
    pn1 = pn + deltaL/(1.0-Dn1)
    
    R1 = deltaL/(1.0 - Dn1) - ((fn1/Kinf)**M)*dt
    R2 = sn1 - (1.0 - Dn1)*E*en1
    R3 = Rn1 - Rn - b*(Rinf - Rn1)*deltaL
    R4 = Dn1 - Dn - ((Yn1/S)**n)*Heaviside(pn1 - pd)*deltaL/(1.0-Dn1)
    
    #print "log =", np.log((1.0-(fn1/Kinf)))
    F = np.array([R1,R2,R3,R4])
    return F

def jacobianJ(x0, propmat,dt):
    ''' This method calculates the jacobian matrix of the residual function
    return KT numpy matrix
    '''
    # getting material properties
    E = propmat.E
    sygma0 = propmat.sygma0
    M = propmat.M
    Kinf = propmat.Kinf
    b = propmat.b
    Rinf = propmat.Rinf
    S = propmat.S
    n = propmat.n                                              
    pd = propmat.pd
    
                                              
    #getting constitutive variables
    deltaL,sn1,Rn1,Dn1 = x0
    epn,pn,Rn,Dn = an
    pn1 = pn + deltaL/(1.0-Dn1)
    
    # yield function and auxiliary variables
    fn1 = yieldFunction(sn1/(1.0-Dn1),Rn1,Dn1,propmat)
    sinal = np.sign(sn1/(1.0-Dn1))
    Dn13=(1.0-Dn1)*(1.0-Dn1)*(1.0-Dn1) 
    HPn1 = Heaviside(pn1 - pd)
    Yn1 = sn1*sn1/(2.0*E*(1.0-Dn1)*(1.0-Dn1))
    en1 = etrial - (deltaL/(1.0-Dn1))*sinal
    
    # auxiliary derivatives    
    dYds = sn1/(E*(1.0- Dn1));
    dYdD = sn1*sn1/(E*Dn13);
    dNdf = (M/Kinf)*(fn1/Kinf)**(M-1.0); #derivative of Nortan law in relation to yieldFunction
    dfdD = sinal*sn1/((1.0 - Dn1)*(1.0 - Dn1)) ;# derivative of yieldFunction in relation do Dn1
    dR4dY =  -Heaviside(pn1 - pd)*n*(Yn1/S)**(n-1.0)*deltaL/(S*(1.0-Dn1));
    
    KT = np.zeros((4,4))                 
    # derivatives                  
    KT[0][0] = 1.0/(1.0 - Dn1)
    KT[0][1] = - dNdf*dt*sinal/(1.0-Dn1)                    
    KT[0][2] = dNdf*dt
    KT[0][3] =  deltaL/((1.0 - Dn1)*(1.0 - Dn1)) - dNdf*dt*dfdD 

    
    KT[1][0] = E*sinal
    KT[1][1] = 1.0
    KT[1][2] = 0.0
    KT[1][3] = E*etrial
    
                  
    KT[2][0] = -b*(Rinf - Rn1)
    KT[2][1] = 0.0
    KT[2][2] = 1.0 + b*deltaL
    KT[2][3] = 0.0
        
                  
    KT[3][0] = -((Yn1/S)**n)*Heaviside(pn1 - pd)/(1.0-Dn1)
    KT[3][1] = dR4dY*dYds
    KT[3][2] = 0.0
    KT[3][3] =  1.0 - ((Yn1/S)**n)*Heaviside(pn1 - pd)*deltaL/((1.0-Dn1)*(1.0-Dn1)) \
                - Heaviside(pn1 - pd)*deltaL*dR4dY*dYdD/(1.0-Dn1)
    
    
    return np.matrix(KT)


def loadcase(t,a,nlc):
    ''' This functions return the deformation at the time t
    where. Different nlc will return different load cases.
    t = current time
    rate = is the rate of increment
    nlc = is the number of the load case function
    '''
    if nlc == 1:
        e = a*t # defomation
        return e
    elif nlc == 2:
        e = np.sin(a*t) # defomation
        return e
    elif nlc == 3:
        e = abs(np.sin(a*t)) # defomation
        return e
    elif nlc == 4:
        e = abs(np.exp(a*t)*np.sin(a*t)) # defomation
        return e
    else:
        print("This load case is not implemented")
        return None
    
def testecase1(E, sygma0, M , Kinf, b, Rinf, S,pd, n ,rate,lc):
    global etrial, an
    # teste case 1
    en1 = 0.00
    epn = 0.0
    pn = 0.0
    Rn = 0.0
    Xn = 0.0
    Dn = 0.0
    an = epn,pn,Rn,Dn
    tinc = 0.001 # time step increment
    #lc = 1 # load case
    
    emax = 0.5 # maximum total strain 
    tend = emax/rate
    #tend = 20.0 # total simulation time
    
    t0 = 0 # initial time 
    #a = 0.1
    dt = tinc 
    data = []
    propmat = PropMat(E, sygma0, M , Kinf, b, Rinf, S, pd, n)
    nint = int(round(tend/tinc))  # number of iterations
    
    for i in range(nint):
        
        t = t0 + tinc  
        en1 = loadcase(t,rate,lc)
                      
        # check elastic trial state
        ftrial, strial, an1trial, etrial = elasticTrial(en1,an,propmat)

        #print("ftrial = %f" %ftrial)
        #print("strial = %f" %strial)
        if ftrial<0.0:
            sn1 = strial
            an1 = an1trial
            epn1,pn1,Rn1,Dn1 = an1
        else:
            deltaL = 0.0
            sn1 = strial
            Rn1 = Rn
            Dn1 =  Dn
            x0 = np.array([deltaL,sn1,Rn1,Dn1])
            #R = residualF(x0, propmat,dt)
            #xsol, info, ier, msg = fsolve(residualF,x0,full_output=1)
            J = lambda x0: jacobianJ(x0,propmat,dt) # Jacobian matrix      
            sol = root(lambda x0: residualF(x0,propmat,dt), x0,jac=J, method='lm', tol=1e-10)
            deltaL,sn1,Rn1,Dn1 = sol.x
            if sol.success == False:
                print(sol.success)
                
            pn1 = pn + deltaL/(1.0 - Dn1)
            epn1 = epn + deltaL*np.sign(sn1)
            sn = sn1
            Rn = Rn1
            an1 = epn1,pn1,Rn1,Dn1


        # store solution    
        #print np.array([en1,sn1,epn,pn1,Rn1,Xn1,Dn1])
        data.append([en1,sn1,epn1,pn1,Rn1,Dn1])

        #update internal variables
        an = an1
        pn = pn1
        epn = epn1
        
        # update time increment
        t0 = t
        
    #---------------------------------
    #plot results
    e = [x[0] for x in data]
    s = [x[1] for x in data]
    ep = [x[2] for x in data]
    Dn = [x[5] for x in data]
    Rn = [x[4] for x in data]
    

    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row',figsize=(8,8),dpi= 80)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(8,8),dpi= 80)
    f.tight_layout()
    plt.subplots_adjust(hspace = 0.3, wspace = 0.2)
    
    ax1.plot(e,s)    
    ax1.set_ylabel('Stress [MPa]')  
    ax1.set_xlabel('Total Strain [mm/mm]')  
    ax1.set_title('Stress vs total Strain')    

    
    ax2.plot(ep,s)    
    ax2.set_ylabel('Stress [MPa]')  
    ax2.set_xlabel('Accumulated plastic Strain [mm/mm]')  
    ax2.set_title('Stress vs Plastic Strain')    

    
    ax3.plot(ep,Dn)    
    ax3.set_ylabel('Damage ')  
    ax3.set_xlabel('Accumulated plastic Strain [mm/mm]')  
    ax3.set_title('Damage Evolution')   
    ax3.set_ylim(0., max([2.0*np.max(Dn),1e-3]))
    
    ax4.plot(ep,Rn)    
    ax4.set_ylabel('Isotropic Hardening ')  
    ax4.set_xlabel('Accumulated plastic Strain [mm/mm]')  
    ax4.set_title('Isotropic Hardening evolution')   
    
    plt.show()


#interact(testecase1,  E = (5e3,200.0e3,1e3), sygma0= (100,500,10), M= (2,10,1), 
#         Kinf= (50,800,50), b= (2,10,1), Rinf= (200,500,20),
#         S= (50,200,10), pd  = (0.005,0.5,0.01), n  = (0.5,2,0.5), rate = (0.0001,0.1,0.001), lc = (1,4,1)     


# Material properties
E = 72.0e3 # young modulus in MPA
sygma0 = 300.0 # initial yield stress in MPA
M = 8.2 
Kinf = 200.
b = 1.86
Rinf = 275.
S = 500.
n = 1.0
pd = 0.0
rate = 0.01
lc = 1
testecase1(E, sygma0, M , Kinf, b, Rinf, S, pd, n ,rate, lc)
