import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

alpha = 0.1
beta = 0.3
gamma = 0.9

state_0 = [10, 1, 0] #suscept, infect, recov

def SIR(state, t):
    d_suscept = -(beta*state[0]*state[1])/(sum(state))
    d_infect = (beta*state[0]*state[1])/(sum(state)) - gamma*state[1]
    d_recov = gamma*state[1]
    return([d_suscept, d_infect, d_recov])
t= np.linspace(0,10,100)
sol = scipy.integrate.odeint(SIR, state_0, t)
print(np.shape(sol[:,1]))
plt.plot(t, sol[:,1],label='infected' )
plt.plot(t, sol[:,0],label='suscept' )
plt.plot(t, sol[:,2],label='recov' )
plt.legend()
plt.show()
