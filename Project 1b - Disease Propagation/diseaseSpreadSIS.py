import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

beta = 0.1 #infectious rate
gamma = 0.1 #recovery rate


#Integration parameters 
t = np.linspace(0, 3500, 100000)

#Initial conditions
names = ['Paris', 'Hong-Kong', 'Malmo']
population = [10000000, 7000000, 300000]
infected_0 = [1000, 0, 0]

nb_cities = len(population)
# Travelling
trav = [[0.001]*nb_cities]*nb_cities #probability for an individual to travel from city n to city m

def multi_cities_SIS(infected, t):
    d_infected, susceptible = [[0]*nb_cities]*2
    for n in range(nb_cities):
        susceptible[n] = population[n] - infected[n]
        d_infected[n] = (beta*susceptible[n]*infected[n])/(population[n]) - gamma*infected[n]
        for m in range(nb_cities) :
            if m != n :
                d_infected[n] += trav[m][n]*infected[m] - trav[n][m]*infected[n] #exchange of infected ppl between cities
    return(d_infected)

sol = odeint(multi_cities_SIS, infected_0, t)

for n in range(nb_cities):
    plt.plot(t, sol[:,n], label=names[n])
plt.legend()
plt.show()


print('Final state :')
print(sol[-1])
