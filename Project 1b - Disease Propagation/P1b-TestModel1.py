
"""
Disease Propagation

What Parameters do we need to consider?
=======================================

Probabilistic Relation - Consecutive interactions?


||====================================================================||                                                                   
||        ,---.         |o              ,---.          |              ||
||        |    ,---.,---|.,---.,---.    |  _.,---.,---.|    ,---.     ||
||        |    |   ||   |||   ||   |    |   ||   |,---||    `---.     ||
||        `---'`---'`---'``   '`---|    `---'`---'`---^`---'`---'     ||
||                             `---'                                  ||
||====================================================================||

What we wish to achieve is to have a number of cities n = 1:M, which each have a population of N_n

We will be attempting to implement a version of the [(S)usceptible][(I)nfected] model (SI, SIS and more)
We will use the following notation to describe the populations in each city:
    S{n,n} = Susceptible living and staying in city n,                I{n,n} = Infected living and staying in City n
    S{m,n} = Susceptible commuting from city m to city n,  I{m,n} = Infected commuting from city m to city n
    N{n,n} = S{n,n}+I{n,n}
Finally, a single n notation Sn/In/Nn is specific to the currently present population in city n (Sum of Snn,Sn+1n,...Smn and I and N)
we then extract the "whole person" ratio of staying, and re-commuting after solving the rate equation for this city.

We can also include a "recovery" quotient, whereby a number "g" determines the rate of recovery of infected
     
The differential equation that we will solve per city is:
    
    dSn/dt = B(\sum^M_m=1{Smn})(\sum^M_m=1{Imn})/(\sum^M_m=1{Nmn}) + g(\sum^M_m=1{Imn}) 

"""
import scipy.linalg
import matplotlib

Cities = 3
B      = 0.5

class city(object):
    def __init__(self,N,I,comN):
        self.I   = I
        self.N   = N
        self.S   = N-I
        self.comN = comN
        self.comI = []
        self.comS = []
        for var in range(len(comN)):
            self.comI.append(I/(N)*comN[var])
            self.comS.append(comN[var]-self.comI[var])
        
N1 = 10000; N2  = 10000; N3 = 100000; 
C11 = 9980; C12 =    10; C13 = 10;
C21 =   10; C22 =  9980; C23 = 10;    
C31 =   10; C32 =    10; C33 = 99980;       
C1 = city(N1,100,[C11,C12,C13])
C2 = city(N2,  0,[C21,C22,C23])
C3 = city(N3,  0,[C31,C32,C33])


#for var in range(Cities):#
for var in range(100):
    dI1 = B*(C1.I-C1.comI[1]-C1.comI[2]+C2.comI[0]+C3.comI[0])*(C1.S-C1.comS[1]-C1.comS[2]+C2.comS[0]+C3.comS[0])/(C1.comN[0]+C2.comN[0]+C3.comN[0]) 
    rat = dI1/(C1.comN[0]+C2.comN[0]+C3.comN[0])
    C1.I = C1.I+C1.N*rat; C2.I = C2.I+C2.comN[0]*rat; C3.I = C3.I+C3.comN[0]*rat;

    C1 = city(N1,  C1.I,[C11,C12,C13])
    C2 = city(N2,  C2.I   ,[C21,C22,C23])
    C3 = city(N3,  C3.I   ,[C31,C32,C33])
    
    print('C1.I = ', str(C1.I))
