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
from IPython import display


NWALK  = 200   #Number of Walkers
DIMX = 100; DIMY = 100; DIMZ = 100   #Dimension of Area Considered 
NSTEPS = 1000
PGDIM = [DIMX,DIMY,DIMZ] #Plagground Dimensions [x,y,z]
plt.rcParams['figure.dpi']   = 150

#%% Testing out using lists of lists instead for this, such that W[n,0]

def walkergen2(N,DIM,TYPE):
    WGen = [None]*N
    LB = int(DIM[0]/10)
    UB = int(3*DIM[0]/10)
    for m in range(N):
        X = randint(LB,UB) 
        Y = randint(LB,UB) 
        Z = randint(LB,UB)    
        WGen[m] = [[m,[X,Y,Z],[DIM[0],DIM[1],DIM[2]],'alive']]
    return(WGen)
def walkermove(self):
     for m in range(len(self)):
        #Random Movement from Current Location
        X = self[m][-1][1][0]; Y = self[m][-1][1][1]; Z = self[m][-1][1][2]
        X = self[m][-1][1][0]+randint(-1,1)
        while X not in range(self[m][-1][2][0]):
            X = self[m][-1][1][0]+randint(-1,1)
        
        Y = self[m][-1][1][1]+randint(-1,1)
        while Y not in range(self[m][-1][2][1]):
            Y = self[m][-1][1][1]+randint(-1,1)
        Z = self[m][-1][1][2]+randint(-1,1)
        
        while Z not in range(self[m][-1][2][2]):
            Z = self[m][-1][1][2]+randint(-1,1)          
        self[m].append([m,[X,Y,Z],[self[m][-1][2][0],self[m][-1][2][1],self[m][-1][2][2]],'alive'])

def localive(self):
    STATUS = [W[n][-1][3]for n in range(len(W))]                    # Polls status of walkers in the current step
    ALIVE  = [i for i in range(len(STATUS)) if STATUS[i]=='alive']  # Gets index of alive walkers
    Locs   = [W[ALIVE[n]][-1][1] for n in range(len(ALIVE))]
    return(Locs)
    
#A[walkerID][StepNumber][ID/Coord/Dim/Status][AdDim]        

W = walkergen2(NWALK,PGDIM,'basic')
Locs = []
for m in range(NSTEPS):     
    walkermove(W)
    Locs.append(localive(W))

plt.ion()
fig = plt.figure(1)

for n in range(NSTEPS):
    CurrLoc = np.asarray([[Locs[n][m][0],Locs[n][m][1],Locs[n][m][2]] for m in range(len(Locs[n]))])
    fig.clf()
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(0, PGDIM[0])
    ax.set_ylim3d(0,PGDIM[0])
    ax.set_zlim3d(0,PGDIM[0])
    ax.scatter3D(CurrLoc[:,0],CurrLoc[:,1],CurrLoc[:,2])
    plt.pause(0.01)
#%%        
            
            
class walkergen(object):
    def __init__(self,DIM,TYPE,INDEX):
        LB = int(DIM[0]/4)
        UB = int(3*DIM[0]/4)
        self.DIM = DIM
        self.x = [randint(LB,UB)]  
        self.y = [randint(LB,UB)]  
        self.z = [randint(LB,UB)]
        self.coord = [[self.x[-1],self.y[-1],self.z[-1]]]
        self.alive = True
        self.TYPE  = TYPE
        self.inx   = INDEX
        if TYPE == 'Aggro':
            self.aggro = True if randint(0,10)>8 else False #
        elif TYPE == 'Explode':
            self.explode = True
        elif TYPE == 'Gather':
            self.gather = True
            self.dtfood = [0]
            
    def walkermove(self,OLD):
        if self.alive == True:
            #%% Random Movement from Current Location
            X = self.x[-1]; Y = self.y[-1]; Z = self.z[-1] 
            X = self.x[-1]+randint(-1,1)
            while X not in range(self.DIM[0]):
                X = self.x[-1]+randint(-1,1)
            
            Y = self.y[-1]+randint(-1,1)
            while Y not in range(self.DIM[1]):
                Y = self.y[-1]+randint(-1,1)
            Z = self.z[-1]+randint(-1,1)
            while Z not in range(self.DIM[2]):
                Z = self.z[-1]+randint(-1,1) 
            
            if [X,Y,Z] in OLD:
                IND = OLD.index([X,Y,Z])
                if self.inx != IND:
                    self.alive = False
                    W[IND].alive = False
                    
            else:        
                self.x.append(X)
                self.y.append(Y)
                self.z.append(Z)
                self.coord.append([X,Y,Z])
W = {n: walkergen(PGDIM,'Aggro',n) for n in range(NWALK)}

def walkerplot(WLOC):
    Locs = []
    for m in range(NWALK):
        Locs.append(np.array(WLOC[m].coord))
    return(Locs)
    #for n in range(len(WLOC)):
    #    Cube[n] = [[Cube[n][0]-0.5,Cube[n][0]+0.5],[Cube[n][1]-0.5,Cube[n][1]+0.5],[Cube[n][2]-0.5,Cube[n][2]+0.5]]
PG = []
for n in range(NSTEPS):
    Wc = W.copy()
    Locs = walkerplot(Wc)
    TempM = np.zeros((PGDIM[0],PGDIM[1],PGDIM[2]))
    for n in range(len(Locs)):
        TempM[Locs[n][-1][0],Locs[n][-1][1],Locs[n][-1][2]] = 1
    PG.append(TempM)
    OldW = [W[n].coord[-1].copy() for n in range(NWALK)]
    for n in range(NWALK): 
        W[n].walkermove(OldW)



plt.ion()
fig = plt.figure(1)
for n in range(NSTEPS):
    fig.clf()
    ax = fig.gca(projection='3d')
    ax.voxels(PG[n], facecolors='red', edgecolor='k',alpha = 0.7)
    plt.pause(0.01)

plt.show()