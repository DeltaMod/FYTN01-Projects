
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
import numpy as np 
"""
With numpy, you can use np.array([1,2,3]) and then you can multiply it as you normally couldn't with a list
numpy also allows you to use np.linspace(i,f,N), which is helpful.
"""
import scipy.linalg
import matplotlib.pyplot as plt
from random import randint

cities = 3
B      = 0.5



class city(object):
    def __init__(self,N,I,comN):
        self.I   = I
        self.N   = N
        self.S   = N-I
        self.comN = comN #Total Number of Commuters FROM the city self.comN[n] is transit from city self to city n 
        self.comI = []   #      Number of infected FROM the city self.comN[n] is transit from city self to city n
        self.comS = []
        if isinstance(comN,int):
            self.comI = I/(N)*comN
            self.comS = comN-self.comI
        if isinstance(comN,list):
            for var in range(len(comN)):
                self.comI.append((I/N)*comN[var])
                self.comS.append(comN[var]-self.comI[var])

# We initialise the cities, and from this can calculate the number of commuters 
# - We're just going to assume that 10% of each population commutes, and they commute based upon the ratio of the population of each city - so C[0]->C[1] = (C[0].N/10) * C[1].N/(C[0].N+C[1].N+C[2].N) 
# - When making the real model, we will of course find real values, but for now this will suffice :) 
Pop = []
for n in range(cities):
    Pop.append( randint(int(np.exp(15-1-n)),int(np.exp(15-n))))
# We then rationalise the number of commuters to each city
Comm = []
for n in range(cities):
    Comm.append([(Pop[n]/10) * Pop[var]/(sum(Pop)) for var in range(cities)])
    #Then, we find how many people stay in the city - this is just so that our sum later can be sum(all commuters) - (staying) so we only consider travellers
    Comm[n][n] = Pop[n]
    for var in range(len(Pop)):
        if var !=n:
            Comm[n][n] = Comm[n][n] - Comm[n][var]

#Now we generalise all cities into a single dictionary command, callable by C[index].classobjects 
C = {n:city(Pop[n],10,Comm[n]) for n in range(cities)}


Temp1 = []
dI    = []
#ICS is "infected city state", and will store [t, I]
ICS   = []
#We do not need to make SCS, since C[index] = N - ICS, but we will store it later 

for t in range(100):
    dI.append([B*C[n].I*C[n].S/C[n].N + sum([C[var].comN[n] for var in range(cities)]) - C[n].comN[n] for n in range(cities)])
     for var in range(cities):
        ICS
        City[var] = city(C[var].N,  C[var].I, [C[var].comN[n] for n in range(cities)])
        
print(str(t))
        
"""  
    dI1 = B*(C[0].I-C[0].comI[1]-C[0].comI[2]+C[1].comI[0]+C[2].comI[0])*(C[0].S-C[0].comS[1]-C[0].comS[2]+C[1].comS[0]+C[2].comS[0])/(C[0].comN[0]+C[1].comN[0]+C[2].comN[0]) 
    rat = dI1/(C[0].comN[0]+C[1].comN[0]+C[2].comN[0])
    C[0].I = C[0].I+C[0].N*rat; C[1].I = C[1].I+C[1].comN[0]*rat; C[2].I = C[2].I+C[2].comN[0]*rat;
    for var in range(cities):
        C[var] = city(C[var].N,  C[var].I, [C[var].comN[n] for n in range(cities)])
    
    print('C[0].I = ', str(C[0].I))
    Temp1.append(C[0].I) 
    
plt.plot(Temp1)
plt.show()
"""