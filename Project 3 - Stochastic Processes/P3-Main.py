"""
Project 3
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import randint
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.stats import norm
import tqdm
plt.rcParams['figure.dpi']   = 150

#Good parameters
#NWALK = 500   #Number of Walkers
#DIMX  = 100; DIMY = 100; DIMZ = 100#Dimension of Area Considered 
#NSTEPS     = 3000
#PGDIM      = [DIMX,DIMY,DIMZ] #Plagground Dimensions [x,y,z]
#WalkerType = 'Aggro' #Aggro|Exploding
#BIRTHS     = 'True'  #Births or no Births
#AGGRNG     = 6       #Aggro Gen RNG - 1:AGGRNG+1 chance to make hunter 
#HR         = 100     #Hunting Radius
#PR         = 50      #Passive Radius
#BR         = 200     #Passive Birth Rate (equiv to 1:BR)
#ABR        = 10      #Aggro Birth chance (equiv to 1:ABR) - This can only happen within 2 days of eating
#ASTRV      = 50      #Rate of aggressive starvation - If aggro does not eat in  ASTRV days, it dies
#PSTRV      = BR*1.5  #Rate of passive "starvation" - On average, each passive cell should reproduce twice in its lifetime
#ANIMATE    = False   #Animates results - set to false to simply get the population plot
#AXLIM      = True    #If true - sets axis constraints for entire lattice, if false - sets constraints only to the action
#MAXPAS     = NWALK*5 #Maximum sustainiable population - This is just to make sure the simulation does not get out of hand


NWALK = 20   #Number of Walkers
DIMX  = 250; DIMY = 250; DIMZ = 250#Dimension of Area Considered 
NSTEPS     = 500
PGDIM      = [DIMX,DIMY,DIMZ] #Plagground Dimensions [x,y,z]
WalkerType = 'Exploding' #Aggro|Exploding
BIRTHS     = False  #Births or no Births
AGGRNG     = 99999999       #Aggro Gen RNG - 1:AGGRNG+1 chance to make hunter 
HR         = 5       #Hunting Radius
PR         = 2       #Passive Radius
BR         = 50      #Passive Birth Rate (equiv to 1:BR)
ABR        = 50     #Aggro Birth chance (equiv to 1:ABR) - This can only happen within 2 days of eating
ASTRV      = 16      #Rate of aggressive starvation - If aggro does not eat in  ASTRV days, it dies
PSTRV      = BR*3  #Rate of passive "starvation" - On average, each passive cell should reproduce twice in its lifetime
ANIMATE    = False    #Animates results - set to false to simply get the population plot
AXLIM      = True    #If true - sets axis constraints for entire lattice, if false - sets constraints only to the action
MAXPAS     = NWALK*5 #Maximum sustainiable population - This is just to make sure the simulation does not get out of hand
DIMPLOT    = True  #Plots only distribution
#%% Testing out using lists of lists instead for this, such that W[n,0]

def walkergen(N,DIM,TYPE,HRNG):
    WGen = [None]*N
    LBM = 4/10; UBM = 6/10; 
    LBX = int(LBM*DIM[0]);LBY = int(LBM*DIM[1]);LBZ = int(LBM*DIM[2])
    UBX = int(UBM*DIM[0]);UBY = int(UBM*DIM[1]);UBZ = int(UBM*DIM[2])
    for m in range(N):
        HNTR = bool(randint(0,HRNG))
        if TYPE=='Exploding':
            HNTR = True
        if HNTR ==False:
            X    = randint(1,DIM[0]-1) 
            Y    = randint(1,DIM[1]-1) 
            Z    = randint(1,DIM[2]-1)
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

    #return delt
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
            #walkersplode(W,AlvLoc,AlvIND)
        if m%int(NSTEPS/pbstep)==0:
            pbar.update(1)
            pbar.set_description('Hunters: '+str(len(AggIND[-1]))+', Runners: '+str(len(PasIND[-1])))
        
pbar.close()
#%% Colourfader code
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
#%%
DW  =  [NWALK-(NWALK-len(AlvLoc[n])) for n in range(len(AlvLoc))]
DAgg = [len(AggIND[n]) for n in range(len(AggIND))]
DPas = [len(PasIND[n]) for n in range(len(PasIND))]
DIST3= []
for n in range(3):
    DIST = []
    for m in range(NSTEPS):
        DIST.append([(AlvLoc[m][:,n] ==i).sum() for i in range(PGDIM[n])])
    DIST3.append(DIST)
if ANIMATE == True:
    fig = plt.figure(1)
    plt.ion()
    for n in range(NSTEPS):
        fig.clf()
        fig.suptitle('Evolution of random walkers')
        if DIMPLOT == True:
            ax = fig.add_subplot(1,2,1,projection='3d')
            ax = fig.gca(projection='3d')
            if AXLIM == True:
                ax.set_xlim3d(0, PGDIM[0])
                ax.set_ylim3d(0,PGDIM[1])
                ax.set_zlim3d(0,PGDIM[2])
            if len(PasLoc[n])!=0:
                ax.scatter3D(PasLoc[n][:,0],PasLoc[n][:,1],PasLoc[n][:,2],color='b',alpha=0.8)
            if len(AggLoc[n])!=0:
                ax.scatter3D(AggLoc[n][:,0],AggLoc[n][:,1],AggLoc[n][:,2],color='r',alpha=0.8)
            if len(DthLoc[n])!=0:
                 ax.scatter3D(DthLoc[n][:,0],DthLoc[n][:,1],DthLoc[n][:,2],color='k',alpha=0.2)
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
    fig = plt.figure(1)
    plt.ion()
    fig.clf()
    fig.suptitle('Evolution of random walkers')
    if DIMPLOT == True:
        ax = fig.add_subplot(1,2,1,projection='3d')
        ax = fig.gca(projection='3d')
        if AXLIM == True:
            ax.set_xlim3d(0, PGDIM[0])
            ax.set_ylim3d(0,PGDIM[1])
            ax.set_zlim3d(0,PGDIM[2])
        if len(PasLoc[-1])!=0:
            ax.scatter3D(PasLoc[-1][:,0],PasLoc[-1][:,1],PasLoc[-1][:,2],color='b',alpha=0.8)
        if len(AggLoc[-1])!=0:
            ax.scatter3D(AggLoc[-1][:,0],AggLoc[-1][:,1],AggLoc[-1][:,2],color='r',alpha=0.8)
        if len(DthLoc[-1])!=0:
             ax.scatter3D(DthLoc[-1][:,0],DthLoc[-1][:,1],DthLoc[-1][:,2],color='k',alpha=0.2)
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
    ax.legend()
    ax.grid()
#%% 
fig = plt.figure(2)
for m in range(NSTEPS):
    if m%(NSTEPS-1)==0:
        
        RDATA = [DIST3[0][m][n]+DIST3[1][m][n]+DIST3[2][m][n] for n in range(len(DIST3[2][m]))]
        SmoD = signal.savgol_filter(RDATA,PGDIM[0]-99,4)
        plt.ion()    
        plt.clf()
        plt.grid()
        plt.plot(RDATA) 
        plt.plot(SmoD)    
        plt.pause(0.01)


#%%
ONLYRUNIFBEEFCOMPUTER = False
PWA = []
for m in range(NWALK):
    PWO =[np.array([W[n][m][1][0],W[n][m][1][1],W[n][m][1][2]]) for n in range(NSTEPS)]
    PWA.append(np.stack(PWO))
    fig = plt.figure(3)
    if ONLYRUNIFBEEFCOMPUTER == True:
        for n in range(NSTEPS):
            plt.clf()
            for m in range(NWALK):
                ax = fig.gca(projection='3d')
                plt.ion()
                ax.plot3D(PWA[m][0:n,0],PWA[m][0:n,1],PWA[m][0:n,2],linewidth=1)
        
            plt.pause(0.01)
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
#%%
"""     
Extra Plots
   
fig = plt.figure(4)
ax = fig.gca(projection='3d')
ax.scatter3D(PasLoc[-1][:,0],PasLoc[-1][:,1],PasLoc[-1][:,2],color='b',alpha=0.8)     
ax.set_xlim3d(0, PGDIM[0])            
ax.set_ylim3d(0,PGDIM[1])
ax.set_zlim3d(0,PGDIM[2]) 
"""
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
