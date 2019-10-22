"""
   ___             _         __    ____                                                          
  / _ \_______    (____ ____/ /_  |_  /  ____                                                    
 / ___/ __/ _ \  / / -_/ __/ __/ _/_ <  /___/                                                    
/_/  /_/  \_____/ /\__/\__/\__/ /____/                                                           
   ______    |___/ __          __  _       ___      __     __  _                                 
  / __/ /____ ____/ / ___ ____/ /_(_____  / _ \___ / ___ _/ /_(____  ___  ___                    
 _\ \/ __/ _ / __/ _ / _ `(_-/ __/ / __/ / , _/ -_/ / _ `/ __/ / _ \/ _ \(_-<                    
/___/\__/\___\__/_//_\_,_/___\__/_/\__/ /_/|_|\__/_/\_,_/\__/_/\___/_//_/___/                    
 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_(_)

Making walkers blow up on contact, or play a "friendly game of hide and seek"
                                                                                                 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.stats import chisquare
import tqdm

plt.rcParams['figure.dpi']   = 150

#Parameters used for the stable population
#NWALK = 100   #Number of Walkers
#DIMX  = 23; DIMY = 23; DIMZ = 23#Dimension of Area Considered 
#NSTEPS     = 10000
#PGDIM      = [DIMX,DIMY,DIMZ] #Plagground Dimensions [x,y,z]
#WalkerType = 'Aggro' #Aggro|Exploding
#BIRTHS     = True  #Births or no Births
#AGGRNG     = 4       #Aggro Gen RNG - 1:AGGRNG+1 chance to make hunter 
#HR         = 9       #Hunting Radius
#PR         = 2       #Passive Radius
#BR         = 20      #Passive Birth Rate (equiv to 1:BR)
#ABR        = 60     #Aggro Birth chance (equiv to 1:ABR) - This can only happen within 2 days of eating
#ASTRV      = 16      #Rate of aggressive starvation - If aggro does not eat in  ASTRV days, it dies
#PSTRV      = BR*3  #Rate of passive "starvation" - On average, each passive cell should reproduce twice in its lifetime
#ANIMATE    = False    #Animates results - set to false to simply get the population plot
#AXLIM      = True    #If true - sets axis constraints for entire lattice, if false - sets constraints only to the action
#MAXPAS     = NWALK*5 #Maximum sustainiable population - This is just to make sure the simulation does not get out of hand
#DIMPLOT    = False  #Plots only distribution

"""
   _____            __     __  _             ___                          __             
  / __(_)_ _  __ __/ /__ _/ /_(_)__  ___    / _ \___ ________ ___ _  ___ / /____ _______ 
 _\ \/ /  ' \/ // / / _ `/ __/ / _ \/ _ \  / ___/ _ `/ __/ _ `/  ' \/ -_) __/ -_) __(_-< 
/___/_/_/_/_/\_,_/_/\_,_/\__/_/\___/_//_/ /_/   \_,_/_/  \_,_/_/_/_/\__/\__/\__/_/ /___/ 
 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
(_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_)

Modify these in any way - the code should be robust to handle most, if not all, possible combinations that aren't zero.
Current parameters should be decent enough to allow for a hunter-runner plot to function
"""
NWALK      = 100               #Number of Walkers
DIMX       = 25                #Lattice x-dim
DIMY       = 25                #Lattice y-dim             
DIMZ       = 25                #Lattice z-dim 
PGDIM      = [DIMX,DIMY,DIMZ]  #Plagground Dimensions [x,y,z]
NSTEPS     = 100              #Total Number of steps taken
WalkerType = 'Aggro'           #Aggro|Exploding
BIRTHS     = True              #Births or no Births
AGGRNG     = 4                 #Aggro Gen RNG - 1:AGGRNG+1 chance to make hunter 
HR         = 9                 #Hunting Radius
PR         = 2                 #Passive Radius
BR         = 20                #Passive Birth Rate (equiv to 1:BR)
ABR        = 60                #Aggro Birth chance (equiv to 1:ABR) - This can only happen within 2 days of eating
ASTRV      = 16                #Rate of aggressive starvation - If aggro does not eat in  ASTRV days, it dies
PSTRV      = BR*3              #Rate of passive "starvation" - On average, each passive cell should reproduce twice in its lifetime
ANIMATE    = True              #Animates results - set to false to simply get the population plot - Note, time to complete simulation 
                               # is IDENTICAL between these modes, it simply takes a long time before it finishes plotting
AXLIM      = True              #If true - sets axis constraints for entire lattice, if false - sets constraints only to the action
MAXPAS     = NWALK*5           #Maximum sustainiable population - This is just to make sure the simulation does not get out of hand
DIMPLOT    = True              #Plots only distribution
DTHPLOT    = False             #Plots locations of dead walkers - may get cluttered
#%% Testing out using lists of lists instead for this, such that W[n,0]

def walkergen(N,DIM,TYPE,HRNG):
    WGen = [None]*N
    #These parameters control volume within the lattice that walkers are generated
    LBM = 3/10; UBM = 7/10;  #Passive/exploding parameters
    LBX = int(LBM*DIM[0]);LBY = int(LBM*DIM[1]);LBZ = int(LBM*DIM[2])
    UBX = int(UBM*DIM[0]);UBY = int(UBM*DIM[1]);UBZ = int(UBM*DIM[2])
    LBMh = 2/10; UBMh = 8/10; #Aggressive walker parameters
    LBXh = int(LBMh*DIM[0]);LBYh = int(LBMh*DIM[1]);LBZh = int(LBMh*DIM[2])
    UBXh = int(UBMh*DIM[0]);UBYh = int(UBMh*DIM[1]);UBZh = int(UBMh*DIM[2])
    for m in range(N):
        HNTR = bool(randint(0,HRNG))
        if TYPE=='Exploding':
            HNTR = True
        if HNTR ==False:
            X    = randint(LBXh,UBXh) 
            Y    = randint(LBYh,UBYh) 
            Z    = randint(LBZh,UBZh)
            WTYPE = 'aggro'
        else:
            X    = randint(LBX,UBX) 
            Y    = randint(LBY,UBY) 
            Z    = randint(LBZ,UBZ)
            WTYPE = 'passive'
        WGen[m] = [m,[X,Y,Z],[DIM[0],DIM[1],DIM[2]],'alive',WTYPE,[],0]
    WStep = []
    WStep.append([WGen[n] for n in range(len(WGen))])
    return(WStep)
    
def walkeradd(self,BIRA,LPAS,MAXPOP):
    if len(LPAS[-1])<MAXPOP:
        for n in range(len(self[-1])):
            if self[-1][n][4] == 'passive' and self[-1][n][3] == 'alive':
                BRNG = bool(int(randint(0,BIRA)/BIRA))
                if BRNG == True:
                    coordmod = [self[-1][n][1][0]+randint(-1,1),self[-1][n][1][1]+randint(-1,1),self[-1][n][1][2]+randint(-1,1)]
                    self[-1].append([len(self[-1])+1,coordmod,self[-1][n][2],'alive','passive',[],0])

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
            WMov.append([m,[X,Y,Z],self[-1][m][2],self[-1][m][3],self[-1][m][4],[],self[-1][m][6]+1])
        else:
            WMov.append(self[-1][m])
    self.append([WMov[n] for n in range(len(WMov))])
 
def walkerhunt(self,HR,PR):
    WMov = []
    for m in range(len(self[-1])):
        if self[-1][m][3] == 'alive':
            #Random Movement from Current Location
            MoveDir = self[-1][m][5][1] - self[-1][m][5][0] 
            if self[-1][m][4] == 'passive' and np.sum(abs(MoveDir))<PR:
                for n in range(3):
                    if MoveDir[n] > 0:
                        MoveDir[n] = -1
                    elif MoveDir[n] == 0:
                        MoveDir[n] = 0
                    elif MoveDir[n] < 0:
                        MoveDir[n] = 1
                rngmod = round(randint(0,3)/3)
                X = self[-1][m][1][0]+MoveDir[0]*rngmod
                while X not in range(self[-1][m][2][0]): #Re-calculate if X is outside of bounds
                    X = self[-1][m][1][0]+randint(-1,1)
                rngmod = round(randint(0,3)/3)
                Y = self[-1][m][1][1]+MoveDir[1]*rngmod
                while Y not in range(self[-1][m][2][1]): #Re-calculate if Y is outside of bounds
                    Y = self[-1][m][1][1]+randint(-1,1)
                rngmod = round(randint(0,3)/3)
                Z = self[-1][m][1][2]+MoveDir[2]*rngmod
                while Z not in range(self[-1][m][2][2]): #Re-calculate if Z is outside of bounds
                    Z = self[-1][m][1][2]+randint(-1,1)
                WMov.append([m,[X,Y,Z],self[-1][m][2],self[-1][m][3],self[-1][m][4],self[-1][m][5],self[-1][m][6]+1])
                
            elif self[-1][m][4] == 'aggro' and np.sum(abs(MoveDir))<HR:
                for n in range(3):
                    if MoveDir[n] > 0:
                        MoveDir[n] = 1
                    elif MoveDir[n] == 0:
                        MoveDir[n] = 0
                    elif MoveDir[n] < 0:
                        MoveDir[n] = -1
                rngmod = round(randint(5,10)/5)   
                X = self[-1][m][1][0]+MoveDir[0]*rngmod
                while X not in range(self[-1][m][2][0]): #Re-calculate if X is outside of bounds
                    X = self[-1][m][1][0]+randint(-1,1)
                
                rngmod = round(randint(5,10)/5)  
                Y = self[-1][m][1][1]+MoveDir[1]*rngmod
                while Y not in range(self[-1][m][2][1]): #Re-calculate if Y is outside of bounds
                    Y = self[-1][m][1][1]+randint(-1,1)
                
                rngmod = round(randint(5,10)/5)  
                Z = self[-1][m][1][2]+MoveDir[2]*rngmod
                while Z not in range(self[-1][m][2][2]): #Re-calculate if Z is outside of bounds
                    Z = self[-1][m][1][2]+randint(-1,1)
                WMov.append([m,[X,Y,Z],self[-1][m][2],self[-1][m][3],self[-1][m][4],self[-1][m][5],self[-1][m][6]+1])
            else:
                X = self[-1][m][1][0]+randint(-1,1)
                while X not in range(self[-1][m][2][0]): #Re-calculate if X is outside of bounds
                    X = self[-1][m][1][0]+randint(-1,1)
            
                Y = self[-1][m][1][1]+randint(-1,1)
                while Y not in range(self[-1][m][2][1]): #Re-calculate if Y is outside of bounds
                    Y = self[-1][m][1][1]+randint(-1,1)
            
                Z = self[-1][m][1][2]+randint(-1,1)
                while Z not in range(self[-1][m][2][2]): #Re-calculate if Z is outside of bounds
                    Z = self[-1][m][1][2]+randint(-1,1)
                WMov.append([m,[X,Y,Z],self[-1][m][2],self[-1][m][3],self[-1][m][4],[],self[-1][m][6]+1])
                
        else:
            WMov.append(self[-1][m])
            
    self.append([WMov[n] for n in range(len(WMov))])       
    
def walkersplode(self,LCTN,LID):
    A,IND = np.unique((LCTN[-1]),axis=0,return_index = True)
    for m in range(len(self[-1])):
        if m not in LID[-1][IND]:
            if self[-1][m][3] == 'alive':
                self[-1][m][3] = 'dead'
                
def walkerkill(self,PIND,PLOC,AIND,ALOC):
    for m in range(len(ALOC[-1])):
        IDLOC = [[i,m] for i in range(len(PLOC[-1])) if PLOC[-1][i][0]  == ALOC[-1][m][0] and PLOC[-1][i][1]  == ALOC[-1][m][1] and PLOC[-1][i][2]  == ALOC[-1][m][2] ]    # Gets index of aggro walkers
        
        for n in range(len(IDLOC)):    
            W[-1][PIND[-1][IDLOC[n][0]]][3] = 'dead'
            W[-1][AIND[-1][IDLOC[n][1]]][6] = 0   #set days since last eaten

def aggrostarve(self,AIND,STARVD):
    for n in range(len(AIND[-1])):
        if self[-1][AIND[-1][n]][6] > STARVD:
            self[-1][AIND[-1][n]][3] = 'dead'

def passivestarve(self,PIND,STARVD,PASMAX):
    CROWDING = bool(randint(0,int((PASMAX/(len(PIND[-1])+1))) ))
    for n in range(len(PIND[-1])):
        if self[-1][PIND[-1][n]][6] > STARVD:
            self[-1][PIND[-1][n]][3] = 'dead'
        elif CROWDING == False and self[-1][PIND[-1][n]][6]<2:
            self[-1][PIND[-1][n]][3] = 'dead'
          
def aggrobirth(self,AIND,BIRA):
    for n in range(len(AIND[-1])):
        if self[-1][AIND[-1][n]][6] < 2:
            BRNG = bool(int(randint(0,BIRA)/BIRA))
            if BRNG == True:
                coordmod = [self[-1][AIND[-1][n]][1][0]+randint(-1,1),self[-1][AIND[-1][n]][1][1]+randint(-1,1),self[-1][AIND[-1][n]][1][2]+randint(-1,1)]
                self[-1].append([len(self[-1])+1,coordmod,self[-1][AIND[-1][n]][2],'alive','aggro',[],0])
#%%
    
                
                
                
        
def localive(WLKR):
    STATUS = [WLKR[-1][n][3]for n in range(len(WLKR[-1]))]                                   # Polls status of walkers in the current step
    ALIVE  = [i for i in range(len(STATUS)) if STATUS[i]                 =='alive']    # Gets index of alive walkers
    AGGRO  = [ALIVE[i] for i in range(len(ALIVE)) if WLKR[-1][ALIVE[i]][4]  =='aggro']    # Gets index of aggro walkers
    PASSV  = [ALIVE[i] for i in range(len(ALIVE)) if WLKR[-1][ALIVE[i]][4]  =='passive']  # Gets index of passive walkers
    DEAD   = [i for i in range(len(STATUS)) if STATUS[i]                 =='dead' ]    # Gets index of dead walkers
    LCTN   = [WLKR[-1][ALIVE[n]][1] for n in range(len(ALIVE))]
    LOCD   = [WLKR[-1][DEAD[n]][1] for n in range(len(DEAD))]
    LAGG   = [WLKR[-1][AGGRO[n]][1] for n in range(len(AGGRO))]
    LPAS   = [WLKR[-1][PASSV[n]][1] for n in range(len(PASSV))]
    return(ALIVE,LCTN,LOCD,AGGRO,LAGG,PASSV,LPAS)
    
#A[walkerID][StepNumber][ID/Coord/Dim/Status][AdDim]        
#%% 
    
def NearestPassive(LAGG,LPASS,nselec):
    NSUM  = []
    delt  = []
    NLOC  = []
    APVec = []
    if len(LPASS[nselec])!= 0:
        for n in range(len(LAGG[nselec])):
            delt.append(abs(LAGG[nselec][n]-LPASS[nselec]))
        for n in range(len(delt)):
            NSUM.append([sum(delt[n][m]) for m in  range(len(LPASS[nselec]))])
            NLOC.append(np.array(LPASS[nselec][np.argmin(NSUM[n])]))
            APVec.append([LAGG[nselec][n],NLOC[n]])
    else:
        for n in range(len(LAGG[nselec])):
            APVec.append([np.array((LAGG[nselec][n])),np.array((-1000, -1000, -1000))])      
    return APVec

def NearestAggro(LAGG,LPASS,nselec):
    NSUM  = []
    delt  = []
    NLOC  = []
    PAVec = []
    if len(LAGG[nselec])!=0:
        for n in range(len(LPASS[nselec])):
            delt.append(abs(LPASS[nselec][n]-LAGG[nselec]))
        for n in range(len(delt)):
            NSUM.append([sum(delt[n][m]) for m in  range(len(LAGG[nselec]))])
            NLOC.append(np.array(LAGG[nselec][np.argmin(NSUM[n])]))
            PAVec.append([LPASS[nselec][n],NLOC[n]])
    else:
        for n in range(len(LPASS[nselec])):
            PAVec.append([np.array((LPASS[nselec][n])),np.array((-1000, -1000, -1000))])
    
    return PAVec

#%% This entire loop controls the simulation, but all changes should be made above
W = walkergen(NWALK,PGDIM,WalkerType,AGGRNG)
AlvLoc = [] #Locations of Alive Walkers
AlvIND = [] #Location Index of Alive Walkers
DthLoc = [] #Locations of Dead Walkers
AggIND = [] #Location Index of aggro walkers 
AggLoc = [] #Location of aggro walkers
PasIND = [] #Location Index of passive walkers
PasLoc = [] #Location of PassiveWalkers
ATPVec = [] #Vector from Agg->Pass
PTAVec = [] #Vector from Pass->Agg
pbstep = 100 #ProgressBarSteps
with tqdm.tqdm(total=int(pbstep)) as pbar:
    for m in range(NSTEPS):
        if BIRTHS == True and m!=0:      
            walkeradd(W,BR,PasIND,MAXPAS)
        LocLivAll = localive(W)
        AlvLoc.append(np.array(LocLivAll[1]))
        AlvIND.append(np.array(LocLivAll[0]))
        DthLoc.append(np.array(LocLivAll[2]))
        AggIND.append(np.array(LocLivAll[3]))
        AggLoc.append(np.array(LocLivAll[4]))
        PasIND.append(np.array(LocLivAll[5]))
        PasLoc.append(np.array(LocLivAll[6]))
        
        if WalkerType == 'Aggro':
            if len(PasLoc[-1])!=0:
                ATPVec.append(NearestPassive(AggLoc,PasLoc,-1))
                PTAVec.append(NearestAggro(AggLoc,PasLoc,-1))
                for n in range(len(AggIND[-1])):
                    W[-1][AggIND[-1][n]][5] = ATPVec[-1][n]
                for n in range(len(PasIND[-1])):
                    W[-1][PasIND[-1][n]][5] = PTAVec[-1][n]
                
                walkerhunt(W,HR,PR)
                walkerkill(W,PasIND,PasLoc,AggIND,AggLoc)
                aggrostarve(W,AggIND,ASTRV)
                aggrobirth(W,AggIND,ABR)
                passivestarve(W,PasIND,PSTRV,MAXPAS)
                
            else: 
                walkermove(W)
                aggrostarve(W,AggIND,50)
                aggrobirth(W,AggIND,ABR)
                passivestarve(W,PasIND,PSTRV,MAXPAS)
        elif WalkerType == 'Exploding':
            walkermove(W)
            walkersplode(W,AlvLoc,AlvIND)
        if m%int(NSTEPS/pbstep)==0:
            pbar.update(1)
            pbar.set_description('Hunters: '+str(len(AggIND[-1]))+', Runners: '+str(len(PasIND[-1])))
        
pbar.close()

#%% This entire block is for iterative (or non-iterative plotting) in addition to some more data analysis stuff
DW  =  [NWALK-(NWALK-len(AlvLoc[n])) for n in range(len(AlvLoc))]
DAgg = [len(AggIND[n]) for n in range(len(AggIND))]
DPas = [len(PasIND[n]) for n in range(len(PasIND))]
DIST3= []
for n in range(3):
    DIST = []
    for m in range(NSTEPS):
        if len(AlvLoc[m]) !=0:
            DIST.append([(AlvLoc[m][:,n] ==i).sum() for i in range(PGDIM[n])])
    DIST3.append(DIST)
if ANIMATE == True:
    fig = plt.figure(1)
    plt.ion()
    for n in range(NSTEPS):
        fig.clf()
        fig.suptitle('Evolution of random walkers')
        if DIMPLOT == True:
            cmrun = plt.get_cmap("winter")
            cmhun = plt.get_cmap("autumn")
            cmded = plt.get_cmap("Greys")
            ax = fig.add_subplot(1,2,1,projection='3d')
            ax = fig.gca(projection='3d')
            if AXLIM == True:
                ax.set_xlim3d(0, PGDIM[0])
                ax.set_ylim3d(0,PGDIM[1])
                ax.set_zlim3d(0,PGDIM[2])
            if DTHPLOT == True:
                if len(DthLoc[n])!=0:
                    ax.scatter3D(DthLoc[n][:,0],DthLoc[n][:,1],DthLoc[n][:,2],c=DthLoc[n][:,2],cmap=cmded,alpha=0.2)
            if len(PasLoc[n])!=0:
                ax.scatter3D(PasLoc[n][:,0],PasLoc[n][:,1],PasLoc[n][:,2],c=PasLoc[n][:,2],cmap=cmrun,alpha=0.8)
            if len(AggLoc[n])!=0:
                ax.scatter3D(AggLoc[n][:,0],AggLoc[n][:,1],AggLoc[n][:,2],c=AggLoc[n][:,2],cmap=cmhun,alpha=0.8)
            
            if len(ATPVec) > n:
                if len(ATPVec[n])!=0:
                    
                    for m in range(len(ATPVec[n])):
                        if sum(ATPVec[n][m][1])!=-3000:
                            if sum(abs(ATPVec[n][m][1]-ATPVec[n][m][0]))<HR:
                                linecolour = 'orange'
                                
                            elif sum(abs(ATPVec[n][m][1]-ATPVec[n][m][0]))>HR:
                                linecolour = 'green'
                                
                                ax.plot3D([ATPVec[n][m][0][0],ATPVec[n][m][1][0]],[ATPVec[n][m][0][1],ATPVec[n][m][1][1]],[ATPVec[n][m][0][2],ATPVec[n][m][1][2]],color = linecolour )
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax = fig.add_subplot(1,2,2)
        else:
            ax = fig.gca()
        if AXLIM == True:
            ax.axis((0,NSTEPS,0,max(DW)))
        ax.plot(DW[1:n],color='green')
        ax.plot(DAgg[1:n],color='red')
        ax.plot(DPas[1:n],color='blue')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel('Number of walkers')
        ax.set_xlabel('Number of iterations')
        plt.pause(0.01)
else:
#%%
    fig = plt.figure(1)
    plt.ion()
    fig.clf()
    fig.suptitle('Evolution of random walkers')
    if DIMPLOT == True:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax = fig.gca(projection='3d')
        cmrun = plt.get_cmap("winter")
        cmhun = plt.get_cmap("autumn")
        cmded = plt.get_cmap("Greys")
        if AXLIM == True:
            ax.set_xlim3d(0, PGDIM[0])
            ax.set_ylim3d(0,PGDIM[1])
            ax.set_zlim3d(0,PGDIM[2])
        if len(DthLoc[-1])!=0:
             ax.scatter3D(DthLoc[-1][:,0],DthLoc[-1][:,1],DthLoc[-1][:,2],c=DthLoc[-1][:,2],cmap=cmded,alpha=0.01)
        if len(PasLoc[-1])!=0:
            ax.scatter3D(PasLoc[-1][:,0],PasLoc[-1][:,1],PasLoc[-1][:,2],c=PasLoc[-1][:,2],cmap=cmrun,alpha=0.8)
        if len(AggLoc[-1])!=0:
            ax.scatter3D(AggLoc[-1][:,0],AggLoc[-1][:,1],AggLoc[-1][:,2],c=AggLoc[-1][:,2],cmap=cmhun,alpha=0.8)
        if len(ATPVec) > n:
            if len(ATPVec[-1])!=0:
                
                for m in range(len(ATPVec[-1])):
                    if sum(ATPVec[-1][m][1])!=-3000:
                        if sum(abs(ATPVec[-1][m][1]-ATPVec[-1][m][0]))<HR:
                            linecolour = 'orange'
                        elif sum(abs(ATPVec[-1][m][1]-ATPVec[-1][m][0]))>HR:
                            linecolour = 'green'
                        ax.plot3D([ATPVec[-1][m][0][0],ATPVec[-1][m][1][0]],[ATPVec[-1][m][0][1],ATPVec[-1][m][1][1]],[ATPVec[-1][m][0][2],ATPVec[-1][m][1][2]],color = linecolour )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax = fig.add_subplot(1,2,2)
    else:
            ax = fig.gca()
    if AXLIM == True:
        ax.axis((0,NSTEPS,0,max(DW)))
    ax.plot(DW,color='green',label='Total')
    if WalkerType == 'Aggro':
        ax.plot(DAgg,color='red',label='Hunters')
        ax.plot(DPas,color='blue',label='Runners')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel('Number of walkers')
    ax.set_xlabel('Number of iterations')
    plt.pause(0.01)
    ax.legend(loc='upper right')
    ax.grid()
    
#%%  Plotting histograms at any point in time m  - currently set to only NSTEPS
EXTRAFIG = False
if EXTRAFIG == True:
    fig = plt.figure(2)
    for m in range(NSTEPS):
        if m%(NSTEPS-1)==0:
            #RDATA = [DIST3[0][m][n]+DIST3[1][m][n]+DIST3[2][m][n] for n in range(len(DIST3[2][m]))]
            bindim     = np.linspace(1,max(PGDIM),int(max(PGDIM))) 
            histx,BINx = np.histogram(AlvLoc[m][:,0],bins = bindim)
            histy,BINy = np.histogram(AlvLoc[m][:,1],bins = bindim)
            histz,BINz = np.histogram(AlvLoc[m][:,2],bins = bindim)
            hist = histx+histy+histz
            if len(hist)>20:
                SmoD = signal.savgol_filter(hist,round(len(hist)/2)*2-3,5)
            plt.ion()    
            plt.clf()
            plt.grid()
            bins  = np.linspace(1,max(PGDIM),len(BINx))
            plt.bar(bins[:-1],hist,width = bins[1]-bins[0],edgecolor='orangered',facecolor='coral') 
            if len(hist)>20:
                plt.plot(bins[:-1],SmoD)    
            plt.pause(0.01)
            plt.xlabel('Distance from Origo')
            plt.ylabel('Number of Walkers')
    
    #%% For the love of god, don't try to run the animation - it used 10 GB of my ram at 20000 iterations
    #Either way, this plots the traces of all walkers - the commented tex animates it - don't do that unless you want to
    ONLYRUNIFBEEFCOMPUTER = False
    PWA = []
    if ONLYRUNIFBEEFCOMPUTER == True:
        for m in range(NWALK):
            PWO =[np.array([W[n][m][1][0],W[n][m][1][1],W[n][m][1][2]]) for n in range(NSTEPS)]
            PWA.append(np.stack(PWO))
            fig = plt.figure(3)
            
#                for n in range(NSTEPS):
#                    plt.clf()
#                    for m in range(NWALK):
#                        ax = fig.gca(projection='3d')
#                        plt.ion()
#                        ax.plot3D(PWA[m][0:n,0],PWA[m][0:n,1],PWA[m][0:n,2],linewidth=1)
#                
#                    plt.pause(0.01)
        ax = fig.gca(projection='3d')
        plt.ion()
        for m in range(NWALK):
            ax.plot3D(PWA[m][:,0],PWA[m][:,1],PWA[m][:,2],linewidth=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        ax.set_xlim3d(0, PGDIM[0])
        ax.set_ylim3d(0,PGDIM[1])
        ax.set_zlim3d(0,PGDIM[2])
        
    #%% Plot the real space locations of all passive and agressive walkers

    fig = plt.figure(4)
    ax = fig.gca(projection='3d')
    m = NSTEPS-1
    cmpas = plt.get_cmap("winter")
    cmhun = plt.get_cmap("autumn")
    ax.scatter3D(PasLoc[m][:,0],PasLoc[m][:,1],PasLoc[m][:,2],alpha=0.8,c=PasLoc[m][:,2],cmap=cmpas)   
    if WalkerType=='Aggro':    
        ax.scatter3D(AggLoc[m][:,0],AggLoc[m][:,1],AggLoc[m][:,2],alpha=0.8,c=AggLoc[m][:,2],cmap=cmhun)   
    ax.set_xlim3d(0, PGDIM[0])            
    ax.set_ylim3d(0,PGDIM[1])
    ax.set_zlim3d(0,PGDIM[2]) 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    #%% Plot densities of each measured simulation
    
    fig = plt.figure(5)
    DWDen  =  [(NWALK-(NWALK-len(AlvLoc[n])))/(PGDIM[0]*PGDIM[1]*PGDIM[2]) for n in range(len(AlvLoc))]
    ax = fig.gca()
    
    if AXLIM == True:
        ax.axis((0,NSTEPS,0,max(DWDen)))
    ax.plot(DWDen,label='N = '+str(NWALK))
    ax.set_ylabel('Density of walkers [walkers/units^3]')
    ax.set_xlabel('Number of iterations')
    ax.grid()
    ax.legend(loc='upper right')
    
    
    
    #%% Compare density distributions and Chi Squared test
    
    fig = plt.figure(6)
    plt.clf()
    for NSIZE in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,5000]:
        if NSIZE == 5000:
            ESTFctn = np.array([NSIZE*(t*0.004)**-1/(PGDIM[0]*PGDIM[1]*PGDIM[2]) for t in range(1,NSTEPS)])
            ax = fig.gca()
            ax.axis((1,NSTEPS,0,max(DWDen)))
            ax.plot(ESTFctn,label='Estimated')  
            ax.plot(DWDen,label='Measured')
    ax.set_ylabel('Density of walkers [walkers/units^3]')
    ax.set_xlabel('Number of iterations')
    ax.legend()    
    ax.grid()
    ChiSQUARE = chisquare(ESTFctn[9999:],DWDen[10000:])
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
