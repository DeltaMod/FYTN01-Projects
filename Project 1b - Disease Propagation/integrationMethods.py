import numpy as np
import scipy.integrate


def Euler(func, y0, t):
    lenVect = len(y0)
    lenTime = len(t)
    sol = np.array([[0.]*lenVect]*lenTime)
    sol[0] = y0
    for i in range(1,lenTime):
        dy = func(sol[i-1], t[i-1])     
        sol[i] = sol[i-1] + dy*(t[i]-t[i-1])
    return(sol)

def RungeKutta4(func, y0, t):
    lenVect = len(y0)
    lenTime = len(t)
    sol = np.array([[0.]*lenVect]*lenTime)
    sol[0] = y0
    for i in range(1,lenTime):
        h = t[i]-t[i-1]
        k1 = h*func(sol[i-1], t[i-1])
        k2 = h*func(sol[i-1]+k1/2., t[i-1]+h/2.)
        k3 = h*func(sol[i-1]+k2/2., t[i-1]+h/2.)
        k4 = h*func(sol[i-1]+k3, t[i-1]+h)
        sol[i] = sol[i-1] + (1./6.)*(k1 + 2*k2 +2*k3 +k4)
    return(sol)

odeint = RungeKutta4 #define default solving method
#odeint = Euler #define default solving method
