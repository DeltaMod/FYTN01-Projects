"""
Project 3
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 #run pip install opencv-python
import time 
from random import randint
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from IPython import display


NWALK  = 50   #Number of Walkers
DIMX = 5; DIMY = 5; DIMZ = 5   #Dimension of Area Considered 
NSTEPS = 50
PGDIM = [DIMX,DIMY,DIMZ] #Plagground Dimensions [x,y,z]
plt.rcParams['figure.dpi']   = 150

#%% Testing out using lists of lists instead for this, such that W[n,0]

def walkergen(N,DIM,TYPE):
    WGen = [None]*N
    LB = int(2*DIM[0]/10)
    UB = int(8*DIM[0]/10)
    for m in range(N):
        X    = randint(LB,UB) 
        Y    = randint(LB,UB) 
        Z    = randint(LB,UB)
        HNTR = bool(randint(0,10))
        if HNTR ==False:
            WTYPE = 'aggro'
        else:
            WTYPE = 'passive'
        WGen[m] = [m,[X,Y,Z],[DIM[0],DIM[1],DIM[2]],'alive',WTYPE]
    WStep = []
    WStep.append([WGen[n] for n in range(len(WGen))])
    return(WStep)

def walkermove(self):
    WMov = []
    for m in range(len(self[-1])):
        if self[-1][m][3] == 'alive':
            #Random Movement from Current Location
            X = self[-1][m][1][0]+randint(-1,1)
            while X not in range(self[-1][m][2][0]): #Re-calculate if X is outside of bounds
                X = self[-1][m][1][0]+randint(-1,1)
            
            Y = self[-1][m][1][1]+randint(-1,1)
            while Y not in range(self[-1][m][2][1]): #Re-calculate if Y is outside of bounds
                Y = self[-1][m][1][1]+randint(-1,1)
            
            Z = self[-1][m][1][2]+randint(-1,1)
            while Z not in range(self[-1][m][2][2]): #Re-calculate if Z is outside of bounds
                Z = self[-1][m][1][2]+randint(-1,1)
            WMov.append([m,[X,Y,Z],self[-1][m][2],self[-1][m][3],self[-1][m][4]])
        else:
            WMov.append(self[-1][m])
    self.append([WMov[n] for n in range(len(WMov))])
        
    
def walkersplode(self,LCTN,LID):
    A,IND = np.unique((LCTN[-1]),axis=0,return_index = True)
    for m in range(len(self[-1])):
        if m not in LID[-1][IND]:
            if self[-1][m][3] == 'alive':
                self[-1][m][3] = 'dead'
                
#%%
def NearestNeighbour(LAGG,AGGIND,LPASS,PASSIND,nselec):
    NSUM =  []
    delt = []
    NLOC = []
    for n in range(len(AGGIND[nselec])):
        delt.append(abs(LAGG[nselec][n]-LPASS[nselec]))
    for n in range(len(delt)):
        NSUM.append([sum(delt[n][m]) for m in  range(len(delt[n]))])
    NLOC = LPASS[nselec][PASSIND[nselec][np.argmin(NSUM)]] 
    return NLOC
    #return delt
A = NearestNeighbour(AggLoc,AggIND,PasLoc,PasIND,0)
#%%
    
                
                
                
        
def localive(self):
    STATUS = [W[-1][n][3]for n in range(len(W[-1]))]                            # Polls status of walkers in the current step
    ALIVE  = [i for i in range(len(STATUS)) if STATUS[i]                 =='alive']    # Gets index of alive walkers
    AGGRO  = [ALIVE[i] for i in range(len(ALIVE)) if W[-1][ALIVE[i]][4]  =='aggro']    # Gets index of aggro walkers
    PASSV  = [ALIVE[i] for i in range(len(ALIVE)) if W[-1][ALIVE[i]][4]  =='passive']    # Gets index of passive walkers
    DEAD   = [i for i in range(len(STATUS)) if STATUS[i]                 =='dead' ]    # Gets index of dead walkers
    LCTN   = [W[-1][ALIVE[n]][1] for n in range(len(ALIVE))]
    LOCD   = [W[-1][DEAD[n]][1] for n in range(len(DEAD))]
    LAGG   = [W[-1][AGGRO[n]][1] for n in range(len(AGGRO))]
    LPAS   = [W[-1][PASSV[n]][1] for n in range(len(PASSV))]
    return(ALIVE,LCTN,LOCD,AGGRO,LAGG,PASSV,LPAS)
    
#A[walkerID][StepNumber][ID/Coord/Dim/Status][AdDim]        

W = walkergen(NWALK,PGDIM,'basic')
AlvLoc = [] #Locations of Alive Walkers
AlvIND = [] #Location Index of Alive Walkers
DthLoc = [] #Locations of Dead Walkers
AggIND = [] #Location Index of aggro walkers 
AggLoc = [] #Location of aggro walkers
PasIND = [] #Location Index of passive walkers
PasLoc = [] #Location of PassiveWalkers
for m in range(NSTEPS):     
    LocLivAll = localive(W)
    AlvLoc.append(np.array(LocLivAll[1]))
    AlvIND.append(np.array(LocLivAll[0]))
    DthLoc.append(np.array(LocLivAll[2]))
    AggIND.append(np.array(LocLivAll[3]))
    AggLoc.append(np.array(LocLivAll[4]))
    PasIND.append(np.array(LocLivAll[5]))
    PasLoc.append(np.array(LocLivAll[6]))
    walkermove(W)
    walkersplode(W,AlvLoc,AlvIND)

    
#%%
fig = plt.figure(1)
plt.ion()
DW = [NWALK-(NWALK-len(AlvLoc[n])) for n in range(len(AlvLoc))]
for n in range(NSTEPS):
    fig.clf()
    fig.suptitle('Evolution of random walkers')
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(0, PGDIM[0])
    ax.set_ylim3d(0,PGDIM[0])
    ax.set_zlim3d(0,PGDIM[0])
    if len(PasLoc[n])!=0:
        ax.scatter3D(PasLoc[n][:,0],PasLoc[n][:,1],PasLoc[n][:,2],color='b',alpha=0.8)
    if len(AggLoc[n])!=0:
        ax.scatter3D(AggLoc[n][:,0],AggLoc[n][:,1],AggLoc[n][:,2],color='r',alpha=0.8)
    if len(DthLoc[n])!=0:
         ax.scatter3D(DthLoc[n][:,0],DthLoc[n][:,1],DthLoc[n][:,2],color='k',alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax = fig.add_subplot(1,2,2)
    ax.plot(DW[1:n])  
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel('Number of walkers')
    ax.set_xlabel('Number of iterations')
    plt.pause(0.001)
#%% 

   
          
"""            
Legacy code: This old version uses a class system that isn't exactly the best - and the voxel based plotting method is slow.
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
"""
