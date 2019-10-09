"""
Project 3
"""
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import cv2 #run pip install opencv-python
import time 
from random import randint
from mpl_toolkits.mplot3d import Axes3D


NWalk = 20
NDIM = 15
Steps = 50
PGDIM = [25,25,25] #Plagground Dimensions [x,y,z]
plt.rcParams['figure.dpi']   = 150



class walkergen(object):
    def __init__(self,DIM,TYPE):
        LB = int(DIM[0]/4)
        UB = int(3*DIM[0]/4)
        self.x = [randint(LB,UB)]  
        self.y = [randint(LB,UB)]  
        self.z = [randint(LB,UB)]
        self.coord = [[self.x[-1],self.y[-1],self.z[-1]]]
        if TYPE == 'Aggro':
            self.aggro = True if randint(0,10)>8 else False # 
    def walkermove(self,DIM):
        X = self.x[-1]; Y = self.y[-1]; Z = self.z[-1] 
        X = self.x[-1]+randint(-1,1)
        while X not in range(DIM[0]):
            X = self.x[-1]+randint(-1,1)
        
        Y = self.y[-1]+randint(-1,1)
        while Y not in range(DIM[1]):
            Y = self.y[-1]+randint(-1,1)
        Z = self.z[-1]+randint(-1,1)
        while Z not in range(DIM[2]):
            Z = self.z[-1]+randint(-1,1) 
        self.x.append(X)
        self.y.append(Y)
        self.z.append(Z)
        self.coord.append([X,Y,Z])
W = {n: walkergen(PGDIM,'Aggro') for n in range(NWalk)}

def walkerplot(WLOC):
    Locs = []
    for m in range(NWalk):
        Locs.append(np.array(WLOC[m].coord))
    return(Locs)
    #for n in range(len(WLOC)):
    #    Cube[n] = [[Cube[n][0]-0.5,Cube[n][0]+0.5],[Cube[n][1]-0.5,Cube[n][1]+0.5],[Cube[n][2]-0.5,Cube[n][2]+0.5]]
PG = []
for n in range(Steps):
    Wc = W.copy()
    Locs = walkerplot(Wc)
    TempM = np.zeros((PGDIM[0],PGDIM[1],PGDIM[2]))
    for n in range(len(Locs)):
        TempM[Locs[n][-1][0],Locs[n][-1][1],Locs[n][-1][2]] = 1
    PG.append(TempM)
    for n in range(NWalk):
        W[n].walkermove(PGDIM)



plt.ion()
fig = plt.figure(1)
for n in range(Steps):
    fig.clf()
    ax = fig.gca(projection='3d')
    ax.voxels(PG[n], facecolors='red', edgecolor='k',alpha = 0.7)
    plt.pause(0.5)

plt.show()