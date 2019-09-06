import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.integrate

#----------------PARAMETERS----------------
#disease parameters :
#TODO : base those on real litterature and not random estimations
alpha = 0.1
beta = 0.7
gamma = 0.1

#population parameters
#TODO : find some real cities to work from
# Note : the following three lists MUST have the same lenght (the number of cities)
cities_names = ['Onetown', 'Twoburg', 'Threeville', 'Fourstad']
susceptible_0 = [1000, 20, 300, 700] #initial susceptible ppl (population)
infected_0 = [100, 0, 0, 0] #initial infected ppl

#integration parameter
t= np.linspace(0,50,300) #integration range : start, end, number of points

#------------------------------------------

nb_cities = len(susceptible_0) #extract the number of cities from lenght of the list
recovered_0 = [0]*nb_cities #at first, no recovered ppl
state_0 = susceptible_0 + infected_0 + recovered_0 #pack the lists in one long list

w = [[0.01]*nb_cities]*nb_cities #probality for someone to go from one city to another.
# TODO : make it dependent on the size of the origin and destination cities, also, make sure total flow is null for each city
#so we don't get massive migrations like it seems to be the case in this model

#Quick run through the list to check that the cities are not leaking to many people :
for n in range(nb_cities):
    tot_flux = 0
    for m in range(nb_cities):
        if m!=n :
            tot_flux += w[m][n]*(susceptible_0[n]+infected_0[n]) - w[n][m]*(susceptible_0[m]+infected_0[m])
    if tot_flux != 0:
        print('Warning. With the current setting of w, the flow of people out of '+cities_names[n]+' is of '+str(tot_flux)+' every day.')


def SIR(state, t):
    ''' This function takes the current state and returns the derivative according to the SIR diff-eqs, i.e.
    y' = SIR(y,t).
    Input : state is a 1-D array, t is a float.
    Output : d_state is a 1-D array of the same length as state.
    '''
    susceptible = state[:nb_cities] #unpack "state" list into adequate variables
    infected = state[nb_cities:2*nb_cities]#unpack "state" list into adequate variables
    recovered = state[2*nb_cities:]#unpack "state" list into adequate variables

    d_suscept = [0]*nb_cities #initialize empty list
    d_infect = [0]*nb_cities#initialize empty list
    d_recov = [0]*nb_cities#initialize empty list

    for n in range(nb_cities):
        d_suscept[n] = -(beta*susceptible[n]*infected[n])/(susceptible[n]+infected[n]+recovered[n]) - gamma*susceptible[n] #SIR model for city n
        d_infect[n] = (beta*susceptible[n]*infected[n])/(susceptible[n]+infected[n]+recovered[n]) - alpha*infected[n]#SIR model for city n
        d_recov[n] = gamma*susceptible[n] + alpha*infected[n]#SIR model for city n

        for m in range(nb_cities):#adding the people coming and going from/to other cities :
            if m != n:
                d_suscept[n] += w[m][n]*susceptible[m] - w[n][m]*susceptible[n]
                d_infect[n] += w[m][n]*infected[m] - w[n][m]*infected[n]
                d_recov[n] += w[m][n]*recovered[m] - w[n][m]*recovered[n]
    return(d_suscept + d_infect + d_recov)#re-pack and return the list


sol = scipy.integrate.odeint(SIR, state_0, t)#computes the solution


sol_suscept = sol[:,:nb_cities] #unpacking solution list
sol_infect = sol[:,nb_cities:2*nb_cities]#unpacking solution list
sol_recov = sol[:,2*nb_cities:]#unpacking solution list

# plot the solution :
#for n in range(nb_cities):
#    plt.plot(t, sol_infect[:,n], label='infected in city n'+str(n))
#    plt.plot(t, sol_suscept[:,n], label='suscept in city n'+str(n))
#    plt.plot(t, sol_recov[:,n], label='recov in city n'+str(n))
#plt.legend()
#plt.show()

#for n in range(nb_cities):
#    plt.plot(t, sol_infect[:,n], label='infected in city n'+str(n))
#    plt.plot(t, sol_suscept[:,n], label='susceptible in city n'+str(n))
#    plt.plot(t, sol_recov[:,n], label='recovered in city n'+str(n))
#    plt.title('Disease-spreading in city n'+str(n))
#    plt.legend()
#    plt.show()

#print(sol[0]) #DEBUG
#print(sol[-1])#DEBUG

figsize = (10, 8)
cols = math.ceil(math.sqrt(nb_cities))
rows = math.ceil(math.sqrt(nb_cities))
#cols = 1
#rows = nb_cities 
fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
axs = axs.flatten()
for n in range(nb_cities):
    axs[n].plot(t, sol_infect[:,n], label='Infected in '+cities_names[n], color='red')
    axs[n].plot(t, sol_suscept[:,n], label='Susceptible in '+cities_names[n], color='blue')
    axs[n].plot(t, sol_recov[:,n], label='Recovered in '+cities_names[n], color='green')
    axs[n].set_title('Disease-spreading in '+cities_names[n])
    axs[n].legend()
plt.show()
