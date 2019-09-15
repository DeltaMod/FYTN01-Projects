
"""
Disease Propagation

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
    
    dSn/dt = beta(\sum^M_m=1{Smn})(\sum^M_m=1{Imn})/(\sum^M_m=1{Nmn}) + g(\sum^M_m=1{Imn}) 

"""
import numpy as np 
"""
With numpy, you can use np.array([1,2,3]) and then you can multiply it as you normally couldn't with a list
numpy also allows you to use np.linspace(i,f,N), which is helpful.

You can also make an nxn empty matrix for when you know the final size of the array:
    np.array([[0.]*4]*5)
    Where the 0. defines a float, because we love floats
    
"""
#import scipy.linalg
import math
import matplotlib.pyplot as plt
from random import randint

"""
TODO: Quarantines after x% number of infected? 
"""

#================================#
#Additional simulation parameters#
#================================#
mode           = 'predef'  # predef|random     # Choose between randomly generating city population and infection values, and our preset  
cities         = 4                             # Total number of cities to run a simulation on - if mode = 'predef', this needs to be equal to 4
DayNightVar    = 'enable' # enable|disable    # Simulates day/night, and daily variation in infection rate (anything but enable disables)
WaningImmunity = 'enable'  # enable|disable    # Simulates waning immunity as an exponential function from 0 to eta 
eta            = [400 for n in range(cities)]  # Waning immunity term - re-susceptability rate (in hours)    
T              = 400                          # Hours to run simulation for

if (mode == 'predef' and cities !=4):
    print("\033[1;31;47m You need to set cities = 4 to run predef \n")

#This is an automatic city name generation system, it is only used if mode = 'random' - it produces cities like 0-ville, 1-town, 2-stad etc.


citymoniker    = ['stad','burg','ville','town','thorp'] 
citynames      = [str(n)+citymoniker[randint(1,len(citymoniker)-1)] for n in range(cities)]

# Modify infection parameters to your liking - we use: alpha = , beta = , gamma = theta = 0
if mode == 'predef':
    alpha = [0.001 for n in range(cities)]                # recovery probability  (sets 'Recovered')
    beta  = [0.05   for n in range(cities)]               # infection probability (sets 'Infected' )
    gamma = [0.0001 for n in range(cities)]                # vaccine probability   (sets 'Recovered')
    theta = [0.01 for n in range(cities)]               # death probability     (removes from N )

if mode == 'random':
    alpha = [randint(1,10)/1000 for n in range(cities)]  # recovery probability  (sets 'Recovered')
    beta  = [randint(30,40)/1000 for n in range(cities)] # infection probability (sets 'Infected' )
    gamma = [randint(1,20)/100 for n in range(cities)]   # vaccine probability   (sets 'Recovered')
    theta = [randint(1,5)/10000 for n in range(cities)]  # death probability     (removes from N )
    
class city(object):
    def __init__(self,N,I,R,comN):
        #Initialise total population, susceptible, infected, and recovered (as well as set t and n to zero
        self.N    = [N]              # Total Population
        self.I    = [I];             # Total Infected
        self.R    = [R];             # Total Recovered 
        self.S    = [N-I-R];         # Total Susceptible
        if (I+R > N or I<0 or R<0):
            print("\033[1;31;47m ERROR: Population imbalance!!!\n Please make sure that the sum of I and R  is smaller than N, and please don't enter any negative values! \n")
        self.t    = [0];             # Time elapsed
        self.n    = [len(self.N)-1]; # Unused
        self.D    = [0];             # Total Dead
        
        # We set up the number of commuters: Data access is self[n].com{N/S/I/R}[m][t] -> from city n, to city m at time t (where n > m is how many "commute to self")
        self.comN = [comN];          # Number of Commuters   FROM city self[n] commuting to city m 
        self.comI = []               # Number of Infected    FROM city self[n] commuting to city m
        self.comS = []               # Number of Susceptible FROM city self[n] commuting to city m   
        self.comR = []               # Number of Recovered   FROM city self[n] commuting to city m 
        
        # Initialise the delta terms - the values for n = 0 are set in dcalc, when calculating the first set of changed values
        self.dN   = []               # Change in N - should be zero if theta is zero
        self.dS   = []               # Change in S - is affected by: gamma, beta,  theta and dSr if waning immunity is enabled
        self.dI   = []               # Change in I - is affected by: alpha and theta
        self.dR   = []               # Change in R - is affected by: alpha, gamma and dSr if waning immunity is enabled
        
        #we also create a name class, just so that we can call it later
        self.name = []
        if isinstance(comN,int):     # Used if comN = 1x1 (single city simulation)
            self.comI.append((I/N)*comN)
            self.comS.append((self.S[-1]/N)*comN)
            self.comR.append(self.comN[-1] - (self.comI[-1]+self.comS[-1]))
        if isinstance(comN,list):    # Used if comN > 1x1 (N number of cities simulation)
            self.comI.append([(I/N)*comN[var]                  for var in range(cities)])
            self.comS.append([((N-I-R)/N)*comN[var]            for var in range(cities)])
            self.comR.append([(R/N)*comN[var]   for var in range(cities)])
    def dcalc(self,dS,dI,dR,dN):     # dcalc runs the addition step of the Euler method, it also recalculates the commuting population based on this
        self.dN.append(dN)
        self.dS.append(dS)
        self.dI.append(dI)
        self.dR.append(dR)
        self.t.append(len(self.N))              #
        self.n.append(len(self.N))              #This is just to keep track of how many n has elapsed, I guess it serves the same purpose as t right now
        self.N.append(self.N[-1] + dN )
        self.S.append(self.S[-1] + dS )
        self.I.append(self.I[-1] + dI )
        self.R.append(self.R[-1] + dR )
        self.D.append(self.N[0] - self.N[-1])
        self.comN.append(self.comN[-1])  # If we ever do mortality, this variable will need to be changed proportionally to dN
        self.comI.append([(self.I[-1]/self.N[-1]) * self.comN[self.n[-1]][var] for var in range(cities)])
        self.comS.append([(self.S[-1]/self.N[-1]) * self.comN[self.n[-1]][var] for var in range(cities)])
        self.comR.append([(self.R[-1]/self.N[-1]) * self.comN[self.n[-1]][var] for var in range(cities)])
        if self.t[-1] % 100  == 0: 
            print('Evaluated City ', self.name, 'for time t = ', str(self.t[-1]) )
            
    #class __add__(self,dI,dS,dR)

# - We initialise the cities, and from this can calculate the number of commuters 
# - We're just going to assume that 10% of each population commutes, and they commute based upon the ratio of the population of each city - so C[0]->C[1] = (C[0].N/10) * C[1].N/(C[0].N+C[1].N+C[2].N) 
# - When making the real model, we will of course find real values, but for now this will suffice :) 

Pop =  []
Comm = []
if mode == 'random':
    for n in range(cities):
        Pop.append( randint(int(np.exp(15-1-n/(cities/2))),int(np.exp(15-n/(cities/2)))))
        # We then rationalise the ratio of commuters to each city
    for n in range(cities):
        Comm.append([(randint(1,5)/100) * Pop[var]/(sum(Pop)) for var in range(cities)])
    
    #Then, we find how many people stay in the city - this is just so that our sum later can be sum(all commuters) - (staying) so we only consider travellers
    Comm[n][n] = 1 - sum( Comm[n][:])+Comm[n][n]
            
    # Now, we make random percentages of each city be infected
    InitI    = [randint(0,10) for n in range(cities)]

if mode == 'predef':
                # Format: PDCITIES[CityID][n], where n = 0 = name, n = 1 = population, n = 2 = initial infected (%)    
    PDCities = [['Marseille'  , 861635, 5],
                ['Montpellier', 277639, 0],
                ['Bezier'   ,   75999 , 0],
                ['Nimes'      , 150672, 0]]  #format: CityName - Population - Percentage Infected 
    NCOM     = 31600/14.384406911431498                 #Gives daily commuters as a function of our day/night cycle (commuters per hour)
    Pop      = [PDCities[n][1] for n in range(cities)]
    CF       = [Pop[n]/sum(Pop)for n in range(cities)]
    InitI    = [PDCities[n][2] for n in range(cities)]
    
    # The commuter values have been determined from french government sources.
    Comm     =  [[NCOM*CF[0]*Pop[0]/(Pop[0]*sum(Pop)), NCOM*CF[0]*Pop[1]/(Pop[0]*sum(Pop)), NCOM*CF[0]*Pop[2]/(Pop[0]*sum(Pop)), NCOM*CF[0]*Pop[3]/(Pop[0]*sum(Pop))],
                [ NCOM*CF[1]*Pop[0]/(Pop[0]*sum(Pop)), NCOM*CF[1]*Pop[1]/(Pop[0]*sum(Pop)), NCOM*CF[1]*Pop[2]/(Pop[0]*sum(Pop)), NCOM*CF[1]*Pop[3]/(Pop[0]*sum(Pop))],
                [ NCOM*CF[2]*Pop[0]/(Pop[0]*sum(Pop)), NCOM*CF[2]*Pop[1]/(Pop[0]*sum(Pop)), NCOM*CF[2]*Pop[2]/(Pop[0]*sum(Pop)), NCOM*CF[2]*Pop[3]/(Pop[0]*sum(Pop))],
                [ NCOM*CF[3]*Pop[0]/(Pop[0]*sum(Pop)), NCOM*CF[3]*Pop[1]/(Pop[0]*sum(Pop)), NCOM*CF[3]*Pop[2]/(Pop[0]*sum(Pop)), NCOM*CF[3]*Pop[3]/(Pop[0]*sum(Pop))]]
        
    
#Then, we find how many people stay in the city - this is just so that our sum later can be sum(all commuters) - (staying) so we only consider travellers
    for n in range(cities):
        Comm[n][n] = (1 - sum( Comm[n][:])+Comm[n][n])
# Note that, in the report, it is suggested that ComN[n][n] = 0, but since the sums cancel out, it doesn't matter if it's equal to N-sum(comN[n][m!=n])


#Now we generalise all cities into a single dictionary command, callable by C[index].classobjects 
C = {n:city(Pop[n],Pop[n]*InitI[n]/100,0,Comm[n]) for n in range(cities)}
#And make some fun names for them to have
if mode == 'random':
    for n in range(cities):
        C[n].name.append(citynames[n])
#Or use our predefined cities        
if mode == 'predef':
    for n in range(cities):
        C[n].name.append(PDCities[n][0])

#Since we want to run this simulation in hours, we need to invent a 24 hours "heat clock" which modifies the total number of commuters, and thus the infection probability        
timeang = np.linspace(0,2*np.pi,24)
DayAct  = [(np.sin(timeang[n])*np.sin(timeang[n]))+0.2/np.exp(np.cos(timeang[n])) for n in range(len(timeang))];  DayAct = DayAct/max(DayAct) #you can set all of this = 1 if you don't want a 24 hour clock to increase/decrease all variables    
if DayNightVar == 'enable':    
    DayAct[12:] = -DayAct[12:]
    WeekAct = [0.8,0.7,1,1,1,1,1]
else:
    DayAct[:] = 1;
    WeekAct = [1,1,1,1,1,1,1]
    

#Our current "waning immunity" assumes all dR(t) will become dSr at t+eta - meaning we get mass "susceptible" influxes - we intend to fix this by
# using a "moving pool" - One Rpool tracks every change in dR as Rpool = dR+Rpool such that Rpool[eta] should have a near 100% chance to cure the susceptible people added to the pool
# we will use a function f(0:eta) = e^(-(t-eta))



fSr = [(np.power(np.exp((np.linspace(0,eta[n],eta[n]))/eta[n])/np.exp(1) ,4))*np.linspace(0,1,eta[n]) for n in range(cities)]


dSr  = [[0 for t in range(T)] for n in range(cities)]       #dSr will hold recovered->susceptible populations 
Rpool =[[0 for t in range(eta[n])] for n in range(cities)]  #Moving Pool containing dR of non vaccinated people 
Rdpool =[[0 for t in range(eta[n])] for n in range(cities)] #tracks Rpool*fSr
for t in range(T):
    for n in range(cities):
        DWF = WeekAct[int(t/24)%7]*DayAct[t%24] #Keeps track of Day-Week-Factor 
        if WaningImmunity == 'enable':
            Rpool[n]  = Rpool[n][:-1]; Rpool[n].insert(0,abs(DWF)*alpha[n]*C[n].I[t])
            Rdpool[n] = [Rpool[n][m]*fSr[n][m] for m in range(eta[n])]
            dSr[n][t] = sum(Rdpool[n])
            for m in range(eta[n]):
                Rpool[n][m]  = Rpool[n][m] - Rdpool[n][m]
        else:
            dSr[n][t] = 0
        
        
            
        dScom =  DWF*sum([C[var].S[t]*C[var].comS[t][n] - C[n].S[t]*C[n].comS[t][var] for var in range(cities)]) 
        dS     = -abs(DWF)*beta[n]  * C[n].I[t]*C[n].S[t]/C[n].N[t] - abs(DWF)*gamma[n]*C[n].S[t] + dSr[n][t]  +dScom
              
        
        dIcom = DWF*sum([C[var].I[t]*C[var].comI[t][n] - C[n].I[t]*C[n].comI[t][var] for var in range(cities)]) 
        dI    =  abs(DWF)*beta[n]  * C[n].I[t]*C[n].S[t]/C[n].N[t] - abs(DWF)*alpha[n]*C[n].I[t] - theta[n] * C[n].I[t] +dIcom \
              #Here, we're calculating sum(Call(+self)->all - Cself->all(+self)) - which should be the same as sum(excluding self)
        
        dRcom = DWF*sum([C[var].R[t]*C[var].comR[t][n] - C[n].R[t]*C[n].comR[t][var] for var in range(cities)])
        dR    =  abs(DWF)*gamma[n] * C[n].S[t] + abs(DWF)*alpha[n]*C[n].I[t]  - dSr[n][t] + dRcom
              
        
        dN = -abs(DWF)*theta[n] * C[n].I[t] + dIcom + dScom + dRcom
        
        C[n].dcalc(dS,dI,dR,dN)

figsize = (10, 8)
plt.figure(1)
cols = 2 #We're making an nxn display, so we take the nearest sized matrix to go with it
rows = int(round(cities/2))
fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
axs = axs.flatten()
for n in range(cities):
    axs[n].plot(C[n].t, C[n].S, label='Susceptible in '       +C[n].name[0], color='blue')
    axs[n].plot(C[n].t, C[n].I, label='Infected in '          +C[n].name[0], color='red')
    axs[n].plot(C[n].t, C[n].R, label='Recovered in '         +C[n].name[0], color='green')
    if sum(theta) != 0:
        axs[n].plot(C[n].t, C[n].N, label='Alive in'              +C[n].name[0], color='orange')
        axs[n].plot(C[n].t, C[n].D, label='Dead in '              +C[n].name[0], color='grey')
    
    #axs[n].plot(C[n].t[1:], C[n].dR, label='dR'              +C[n].name[0], color='black')
    axs[n].set_title(                 'Disease-spreading in ' +C[n].name[0])
    #axs[n].plot(C[n].t, list(list(zip(*C[0].comR))[0]), label='Commuting to '       +C[n].name[0], color='black')
    axs[n].legend()

plt.grid(True)
plt.show()

