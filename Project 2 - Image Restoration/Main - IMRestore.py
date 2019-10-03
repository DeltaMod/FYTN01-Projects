"""
                                                                                                                      
 ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______               
|______||______||______||______||______||______||______||______||______||______||______||______||______|              
     __          ______               __               __        _______                           __                 
    |  |        |   __ \.----..-----.|__|.-----..----.|  |_     |_     _|.--.--.--..-----.        |  |                
    |  |        |    __/|   _||  _  ||  ||  -__||  __||   _|      |   |  |  |  |  ||  _  |        |  |                
    |  |        |___|   |__|  |_____||  ||_____||____||____|      |___|  |________||_____|        |  |                
    |__|                            |___|                                                         |__|                
 ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______  ______               
|______||______||______||______||______||______||______||______||______||______||______||______||______|           
   
                                                                                                                      
                                                                                                          
1 - Download Greyscale image (We converted a colour .jpeg to a greyscale .bmp file)
2. Create a mask (dimensions M × N) (We created a two colour (black and alpha) mask on top of the image in GIMP)
3. Implement a code which solves numerically the Laplace equation for the missing region. Denote by Irestored(x, y), the restored image.
4. To quantify how successful the image restoration was introduce a “discrepancy score”, a Chi squared test, between the graffiti-sprayed regions from the original image and the
restored images. For instance,

Chi^2 = 1/n sum_{x,y}(np.abs())
nPx,y[Irestored(x, y) − I(x, y)]2
2 (1)
Here, the sum is over the pixels in the missing region and
2
1
n − 1X
x,y
[I(x, y) − Imean]2 (2)
where Imean is the mean of the original data of the missing region. Investigate
how 2 depends on the size and shape of the missing region. Are some images
more difficult than others?
Advanced version (for higher grade)
Try to improve upon the method above. For instance, you could try to include a
force term into Laplace equation, use an anisotropic or spatially-varying diffusion
constant, change the boundary condition etc. Or, you may want to consult one of
the methods available in the literature (see [3]-[6]). Or, even better, come up with
some method of your own!
"""
from matplotlib.pyplot import imread
from scipy.signal import fftconvolve as FFTCONV2
import matplotlib.pyplot as plt
import numpy as np
import cv2 #run pip install opencv-python
"""
    ____                  __     __  __                                        __  _                     __                   __
   /  _/___  ____  __  __/ /_   / /_/ /_  ___     ____  ____  ___  _________ _/ /_(_)___  ____  _____   / /_  ___  ________  / /
   / // __ \/ __ \/ / / / __/  / __/ __ \/ _ \   / __ \/ __ \/ _ \/ ___/ __ `/ __/ / __ \/ __ \/ ___/  / __ \/ _ \/ ___/ _ \/ / 
 _/ // / / / /_/ / /_/ / /_   / /_/ / / /  __/  / /_/ / /_/ /  __/ /  / /_/ / /_/ / /_/ / / / (__  )  / / / /  __/ /  /  __/_/  
/___/_/ /_/ .___/\__,_/\__/   \__/_/ /_/\___/   \____/ .___/\___/_/   \__,_/\__/_/\____/_/ /_/____/  /_/ /_/\___/_/   \___(_)   
         /_/                                        /_/                                                                        
The format is: OPERATIONS = ['FILTER','MODE',Repetitions] - Add as many as you want in the given formatm, just add another row and you're good to go!
"""

showIterations = False

#OPERATIONS = [['Gauss-3x3','FFTConvolveCut',4],
#              ['Gauss-5x5','FFTConvolveCut',4],]

OPERATIONS = [['Gauss-3x3','PixelsOnly',20]]



"""
MODE available:
    'PixelsOnly'     - Standard for loop to do a mask-defined pixel-by-pixel image convolution
    'FullImage'      - Applies a filter over the ENTIRE IMAGE (SLOW)
    'FFTConvolve'    - Converts image into frequency domain, then multiplies it with the image kernel (Faster than FullImage) 
    'FFTConvolveCut' - Cuts out an area around the mask, does FFTCONVOLVE, then transforms it back and adds the change to the image
    'MedianFilter':  - Doesn't need a "FILTER" - it does what it says on the tin

FILTER available:
    'EdgeLapl-3x3'   - Laplacian edge finder, 3x3 
    'IEdgeLapl-3x3'  - Inverse laplacian edge finder, 3x3
    'Gauss-3x3'      - Gaussian kernel, 3x3
    'Gauss-5x5'      - Gaussian kernel, 5x5
    'Unsharp-5x5'    - Unsharp filter, 5x5
    'BoxBlur-3x3'    - Average value filter, 3x3
    'Sharpen-3x3'    - Sharpen filter (STRONG), 3x3
    


I = IMHandler(original,mask)
I.IMFLTR(s_coord,'Gauss-5x5' ,'FFTConvolveCut',7)

"""


plt.rcParams['figure.dpi']   = 200
plt.rcParams['axes.grid'] = False

#Read in and determine the dimensions of the image
original = imread('EM_ScreamGS_lowres.bmp')[:,:,0] #Available Images: EM_ScreamGS_lowres.bmp
Dim   = original.shape
rowrange = Dim[0]; colrange = Dim[1]

#Read in and plot the mask
mask = imread('mask_lowres2.bmp')[:,:,0]/255 #Available Masks: mask_lowres|mask_lowres2|mask_lowres3|mask_lowres4|mask_noise
plt.figure()
plt.imshow(original, cmap='gray')
plt.title('Original Image')
plt.show()

## Pre-plot damaged image
plt.figure()
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.show()

## Since we're using convolusion kernels, we're expecting to see a reduction in the size  of the image for each iteration unless we do something about it.
#We take the image, mirror it in all axes around itself, and let the convolution happen only in the "image plane" where edge quality is preserved

maskI    = original*mask  #We make the original damaged image
#For the mirroring, we need three types - corner cuts (both), vertical cuts (column pixels only), vertical cuts (row pixels only)
mirrIB   = np.array([[maskI[row+1,col+1] for col in range(colrange-2)] for row in range(rowrange-2)])   #Then take the mirror plane to be [start+1:end-1]
mirrIV   = np.array([[maskI[row,col+1]   for col in range(colrange-2)] for row in range(rowrange  )])   #Vertical cut = column trimming
mirrIH   = np.array([[maskI[row+1,col]   for col in range(colrange  )] for row in range(rowrange-2)])   #Horizontal cut = row trimming

mirrDim = mirrIB.shape 
mirr    = mirrDim[0];  micr= mirrDim[1] #MIrror-Row-Range and MIrror-Column-Range - 
#We only need mirrIB.shape, because it's the only offset we need to know to for loop all of the damaged image for our recreation

#Then, the damaged image is the full 3x3 matrix of correctly mirrored images, as to allow the kernel to sample corners and edges with no modification
#We use np.flip(mirrI,axis = 1) for Horizontal, and np.flip(axis = 0) for vertical - finally np.flip(mirrIB,axis=None) for both
#In order, row by row from top to bottom, we get:  both|vert|both:hor|ORIGINAL
TBR    = np.hstack((np.flip(mirrIB,axis = None), np.flip(mirrIH,0),np.flip(mirrIB,axis = None)))
MR     = np.hstack((np.flip(mirrIV,-1),maskI,np.flip(mirrIV,-1)))
dmgI   = np.vstack((TBR,MR,TBR))
  
plt.figure()
plt.imshow(dmgI, cmap='gray')
plt.title('Damaged image')
plt.show()

IMRes = maskI #Initialise size of "restored image"

#We make a list of kernels, callable by their names - 
KernelNames = ['EdgeLapl-3x3', 'IEdgeLapl-3x3','Gauss-3x3', 'Gauss-5x5','Unsharp-5x5','BoxBlur-3x3','Sharpen-3x3']
KM = []

#Edge Laplacian = 'EdgeLapl-3x3':
KM.append(np.array([       [0,  1, 0],
                           [1, -4, 1], 
                           [0,  1, 0]]))
    
#Inverse Edge Laplacian = 'IEdgeLapl-3x3':                    
KM.append(np.array([       [-1, -1, -1],
                           [-1,  8, -1], 
                           [-1, -1, -1]]))
#Gaussian Blur 3x3 = 'Gauss-3x3':
KM.append(1/16*np.array([  [1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])) 
#Gaussian Blur 5x5 = 'Gauss-5x5':
KM.append(1/256*np.array([ [1,  4,  6,  4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1,  4,  6,  4, 1]])) 
#Unsharp Mask 5x5 == 'Unsharp-5x5':
KM.append(-1/256*np.array([[1,  4,    6,  4, 1],
                           [4, 16,   24, 16, 4],
                           [6, 24, -476, 24, 6],
                           [4, 16,   24, 16, 4],
                           [1,  4,    6,  4, 1]])) 
#Box Blur 3x3 = 'BoxBlur-3x3'
KM.append(1/9*np.array([   [1,  1, 1],
                           [1,  1, 1], 
                           [1,  1, 1]]))
#Sharpen 3x3  = 'Sharpen-3x3'
KM.append(1/9*np.array([       [-1,  -1, -1],
                           [-1,   9, -1], 
                           [-1,  -1, -1]]))
#We then stick all of these into a neat dictionary for later use!
Kernel = {KernelNames[m]:KM[m] for m in range(len(KernelNames))}


#%% Used to extract the coordinates of the mask (this can be set to detect noise that isn't zero)
def s_transform(mask):
    s_map = []
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y] < 1:
                s_map.append([x+mirr,y+micr])
    return(s_map)
s_coord = s_transform(maskI)

#%% Our IMHandler class allows for any nomber of consecutive codes to be run
class IMHandler(object):
    def __init__(self,IMG,DPIX): #Initialise the damaged image
        self.IMG  = [IMG]       # Original Image
        self.DPIX = [DPIX]      # Damage Pixels applied to main image
        try:
            self.DMGI = [IMG*DPIX]
        except:
              print("\033[1;31;47m ERROR: IMG and DPIX dimensions do not match - please make an N*M mask the same size as the source image! \n")
        self.MASK = []
        self.FLTR = []
        self.MODE = []
        self.REP  = []
        self.IMRES= [IMG]
        self.MSKCU= []
    def IMFLTR(self,MASK,FLTR,MODE,REP):
        Dim = np.shape(self.DMGI[-1])
        rowrange = Dim[0]; colrange = Dim[1]
        LO = Kernel[FLTR]
        dLO = int(np.floor(len(LO)/2)); uLO = (len(LO) - dLO); 

        #try:
        #For image convolutions to work, we need 
        #For the mirroring, we need three types - corner cuts (both), vertical cuts (column pixels only), vertical cuts (row pixels only)
        mirrIB   = np.array([[self.DMGI[-1][row+1,col+1] for col in range(colrange-2)] for row in range(rowrange-2)]) #Then take the mirror plane to be [start+1:end-1]
        mirrIV   = np.array([[self.DMGI[-1][row,col+1]   for col in range(colrange-2)] for row in range(rowrange  )])   #Vertical cut = column trimming
        mirrIH   = np.array([[self.DMGI[-1][row+1,col]   for col in range(colrange  )] for row in range(rowrange-2)])   #Horizontal cut = row trimming
        mirrDim = mirrIB.shape 
        mirr    = mirrDim[0];  micr= mirrDim[1] #MIrror-Row-Range and MIrror-Column-Range - 
        #We only need mirrIB.shape, because it's the only offset we need to know to for loop all of the damaged image for our recreation
        
        #Then, the damaged image is the full 3x3 matrix of correctly mirrored images, as to allow the kernel to sample corners and edges with no modification
        #We use np.flip(mirrI,axis = 1) for Horizontal, and np.flip(axis = 0) for vertical - finally np.flip(mirrIB,axis=None) for both
        #In order, row by row from top to bottom, we get:  both|vert|both:hor|ORIGINAL
        TBR    = np.hstack((np.flip(mirrIB,axis = None), np.flip(mirrIH,0),np.flip(mirrIB,axis = None)))
        MR     = np.hstack((np.flip(mirrIV,-1),self.DMGI[-1],np.flip(mirrIV,-1)))
        dmgI   = np.vstack((TBR,MR,TBR))
        for repeats in range(REP):  
            IMRES  = self.IMRES[-1].copy()
            if MODE == 'FullImage':
                for row in range(mirr,mirr+rowrange):
                    for col in range(micr,micr+colrange):
                        IMRES[row-mirr][col-micr] = sum(sum(dmgI[row-dLO:row+uLO,col-dLO:col+uLO]*LO))
            if MODE == 'FFTConvolve':
                CONV = FFTCONV2(dmgI,LO,'same') #Note: Since FFTCONV2 mode = 'same" outputs an NxM-1 sized matrix, we can use it for the centre image without fault since we recerate it every time
                IMRES = CONV[mirr:mirr+rowrange,micr:micr+colrange]
                
            if MODE == 'FFTConvolveCut':
                MaskCut = np.zeros((dmgI.shape[0],dmgI.shape[1]))
                for pix in range(len(MASK)):
                    row = MASK[pix][0]; col = MASK[pix][1]
                    MaskCut[row-3*dLO:row+3*uLO,col-3*dLO:col+3*uLO] = dmgI[row-3*dLO:row+3*uLO,col-3*dLO:col+3*uLO]
                CONV = FFTCONV2(MaskCut,LO,'same') #Note: Since FFTCONV2 mode = 'same" outputs an NxM-1 sized matrix, we can use it for the centre image without fault since we recerate it every time
                for pix in range(len(MASK)):
                    row = MASK[pix][0]; col = MASK[pix][1]
                    dmgI[row,col] = CONV[row,col]
                IMRES = dmgI[mirr:mirr+rowrange,micr:micr+colrange]
                self.MSKCU.append(MaskCut)
                
            if MODE == 'MedianFilter':
                for row in range(mirr,mirr+rowrange):
                    for col in range(micr,micr+colrange):    
                        IMRES[row-mirr][col-micr] = np.median(dmgI[row-dLO:row+uLO,col-dLO:col+uLO])
                        
            if MODE == 'PixelsOnly':
                for pix in range(len(MASK)):
                    row = MASK[pix][0]; col = MASK[pix][1]
                    IMRES[row - mirr][col-micr] = sum(sum(dmgI[row-dLO:row+uLO,col-dLO:col+uLO]*LO))
            self.DMGI.append(IMRES)
            self.IMRES.append(IMRES)
            mirrIB   = np.array([[IMRES[row+1,col+1] for col in range(colrange-2)] for row in range(rowrange-2)]) #Then take the mirror plane to be [start+1:end-1]
            mirrIV   = np.array([[IMRES[row,col+1]   for col in range(colrange-2)] for row in range(rowrange  )])   #Vertical cut = column trimming
            mirrIH   = np.array([[IMRES[row+1,col]   for col in range(colrange  )] for row in range(rowrange-2)])   #Horizontal cut = row trimming
            TBR    = np.hstack((np.flip(mirrIB,axis = None), np.flip(mirrIH,0),np.flip(mirrIB,axis = None)))
            MR     = np.hstack((np.flip(mirrIV,-1),IMRES,np.flip(mirrIV,-1)))
            dmgI   = np.vstack((TBR,MR,TBR))
        #except:
         #   print("\033[1;31;47m ERROR: Something went wrong - you need to input correct values for this to work! \n")

I = IMHandler(original,mask)
for m in range(len(OPERATIONS)):
    I.IMFLTR(s_coord,OPERATIONS[m][0],OPERATIONS[m][1],OPERATIONS[m][2])

#%%
cols = 1

plt.figure()
plt.imshow(I.DMGI[0],cmap = 'gray')
plt.title('Image before restoration')
plt.show()

if showIterations :
    for m in range(1, len(I.IMRES)):
        plt.figure()
        plt.imshow(I.IMRES[m],cmap='gray')
        plt.title('Repair Iteration '+str(m))
        plt.show()
else :
    plt.figure()
    plt.imshow(I.IMRES[-1],cmap='gray')
    plt.title('Repair Iteration '+str(len(I.IMRES)))
    plt.show()


#%% Error Calculation (compare original to filtered image)

IMDiff = (-1-abs(original - I.IMRES[-1]))
plt.figure()
plt.imshow(IMDiff,cmap='gray')
plt.title('Difference Between Original and Repaired')
plt.show()

SCORE = np.sum(np.abs(I.IMRES[-1]-original))/np.count_nonzero(maskI[-1]-original)/(np.sum(np.abs(maskI - original))/np.count_nonzero(maskI-original))
print('\n The numbers are in! \n The average pixel difference in the graffiti`d image is: ' +str((np.sum(np.abs(maskI - original))/np.count_nonzero(maskI-original))))
print('\n The average pixel difference in the repaired image is: ' +str(np.sum(np.abs(I.IMRES[-1]-original))/np.count_nonzero(maskI[-1]-original)))
print('\n The ratio of average pixel difference between the mask is = '+str(SCORE)+' \n A lower number is better!')

if SCORE>1:
    print("\033[1;31;47m Are you trying to mess the image up or something? The you're not supposed to be aiming for a high score :(")

"""
Legacy Code 3 - #%% This filter does not work - We'd need a better method of identifying where the dead pixels are, and we likely need to evaulate it each iteration to determine if we are done.
experimentalfilter = 0 
if experimentalfilter == 1:
    DIFFIMGx = np.power(np.flip(np.rot90(np.diff([dmgI[:,m] for m in range(dmgI.shape[1])]),3),-1),2)
    DIFFIMGy = np.power(np.diff([dmgI[m,:] for m in range(dmgI.shape[0])]),2)
    DIFFIMG =  DIFFIMGx[mirr-1:mirr+rowrange,micr-1:micr+colrange]+DIFFIMGy[mirr-1:mirr+rowrange,micr-1:micr+colrange]
    DIFFIMG[DIFFIMG>0.6] = 1
    DIFFIMG[DIFFIMG<=0.6] = 0
    maskI = DIFFIMG
    plt.imshow(DIFFIMG,cmap = 'gray')


Legacy Code 2 - Has been moved into a class, still works as intended 

for repeats in range(5):
    if SearchMode == 'FullImage':
        for row in range(mirr,mirr+rowrange):
            for col in range(micr,micr+colrange):
                IMRes[row-mirr][col-micr] = sum(sum(dmgI[row-dLO:row+uLO,col-dLO:col+uLO]*LO))
    
    if SearchMode == 'PixelsOnly':
        for pix in range(len(s_coord)):
            row = s_coord[pix][0]; col = s_coord[pix][1]
            IMRes[row - mirr][col-micr] = sum(sum(dmgI[row-dLO:row+uLO,col-dLO:col+uLO]*LO))
    
    mirrIB   = np.array([[IMRes[row+1,col+1] for col in range(colrange-2)] for row in range(rowrange-2)]) #Then take the mirror plane to be [start+1:end-1]
    mirrIV   = np.array([[IMRes[row,col+1]   for col in range(colrange-2)] for row in range(rowrange  )])   #Vertical cut = column trimming
    mirrIH   = np.array([[IMRes[row+1,col]   for col in range(colrange  )] for row in range(rowrange-2)])   #Horizontal cut = row trimming
    TBR    = np.hstack((np.flip(mirrIB,axis = None), np.flip(mirrIH,0),np.flip(mirrIB,axis = None)))
    MR     = np.hstack((np.flip(mirrIV,-1),IMRes,np.flip(mirrIV,-1)))
    dmgI   = np.vstack((TBR,MR,TBR))

Legacy Code 1, which still uses corner and edge conditions - obsolete to the max, considering what we have now
This code does not use a mirror matrix, and must use the mask image alone, which is NOT called dmgI anymore - to change





Old correction Kernels:
    CornerCorrection for laplacian edge finder
    CC =  np.array([[1,   1, 1],
                    [1, 0.5, 1],
                    [1,   1, 1]])
    LC = np.array([[1,  1,   1],
                 [1, 0.75, 1],
                 [1, 1,   1]])
    Corner/EdgeCorrection for laplacian inverse edge finder
    CC =  np.array([[1,   1,  1],
                    [1, 5/8,  1],
                    [1,   1,  1]])
    LC = np.array([[1,  1,   1],
                 [1, 5/8, 1],
                 [1, 1,   1]])
for row in range(rowrange):
    for col in range(colrange):
        
        #First we check corner cases, because they need a CornerCorrection (CC) matrix applied to each appropriate matrix
        if dmgI[row][col] == 0:
            if row == 0:
                if col == 0:                #TL Corner - Want BR matrix only
                    IMRes[row][col] = sum(sum(dmgI[row:row+2,col:col+2]*LO[1:3,1:3]*CC[1:3,1:3]))
                elif col == colrange:       #TR Corner - Want BL matrix only
                    IMRes[row][col] = sum(sum(dmgI[row:row+2,col-1:col+1]*LO[1:3,0:2]*CC[1:3,0:2]))
                else:                       #Top Line - Want Bottom matrix only
                    IMRes[row][col] = sum(sum(dmgI[row:row+2,col-1:col+2]*LO[1:3,0:3]*LC[1:3,0:3]))
            elif row == rowrange:
                if col == 0:                 #BL Corner - Want TR matrix only 
                    IMRes[row][col] = sum(sum(dmgI[row-1:row+1,col:col+2]*LO[0:2,1:3]*CC[0:2,1:3]))
                elif col == colrange:        #BR Corner- Want TL matrix only
                    IMRes[row][col] = sum(sum(dmgI[row-1:row+1,col-1:col+1]*LO[0:2,0:2]*CC[0:2,0:2])) 
                else:                        #Bottom Line - Want Top Matrix only
                    IMRes[row][col] = sum(sum(dmgI[row-1:row+1,col-1:col+2]*LO[1:3,0:3]*LC[1:3,0:3]))
            elif col == 0 and row != 0 and row != rowrange:         #Left Line - Want Right Matrix only
                IMRes[row][col] = sum(sum(dmgI[row-1:row+2,col:col+2]*LO[0:3,1:3]*LC[0:3,1:3]))
            elif col == colrange and row != 0 and row != rowrange:  #Right Line - Want Left Matrix only
                IMRes[row][col] = sum(sum(dmgI[row-1:row+2,col-1:col+1]*LO[1:3,0:3]*LC[0:3,0:2]))   
            else:
                IMRes[row][col] = sum(sum(dmgI[row-1:row+2,col-1:col+2]*LO[0:3,0:3]))   
        else:
            IMRes[row][col] = dmgI[row][col]

plt.figure()
plt.imshow(IMRes, cmap='gray')
plt.title('Restored Image?')
plt.show()


"""


#%%Short code to show off convolutions:
PLOTTER = 1
if PLOTTER == 1:
    imexp = 'MaskAndDiff'
    A = I.DMGI[1]
    B = I.IMG[0]
    
    A[0,:] = B[0,:]
    A[:,0] = B[:,0]
    C = KM[0]
    m = 0
    D = C[0:3,0:3]*B[m:m+3,m:m+3]
    
    E = [np.sum(D)]
    if imexp == 'BadIm':
        BadIm = []
        for m in range(OPERATIONS[0][2]):
            IMT = I.IMRES[m].copy()
            IMT[0:m,:] = 0;  IMT[:,colrange-m:colrange] = 0
            IMT[:,0:m] = 0;  IMT[rowrange-m:rowrange,:] = 0
            BadIm.append(IMT[:,:])
            cmap = plt.cm.gray
            norm = plt.Normalize(vmin=np.min(BadIm[m]), vmax=np.max(BadIm[m]))
            image = cmap(norm(BadIm[m]))
            plt.imsave('ImageFrames/ConvProb'+str(m)+'.png', image)
    
    #We define a colourmap to keep the same pixel value range for all plots
    cmap    = plt.cm.gray
    norm    = plt.Normalize(vmin=0, vmax=256)
    normdif = plt.Normalize(vmin=0, vmax=256) 
    
    
    ResIm  = []; ResIm.append(I.IMG[0]*I.DPIX[0]); imageres = cmap(norm(ResIm[0]))
    ImDiff = []
    IMCOMPARISON = []
    ImDiff.append(255*np.ones((rowrange,colrange)))
    
    for pix in range(len(s_coord)):
            row = s_coord[pix][0]-mirr; col = s_coord[pix][1]-micr
            ImDiff[0][row][col] = 255-np.abs(I.DPIX[0][row][col]*I.IMG[0][row,col]-I.IMG[0][row,col])
    imagediff= cmap(normdif(ImDiff[0]))
    plt.imsave('ImageFrames/'+OPERATIONS[0][0]+'Res'+str(0)+'.png', imageres)
    plt.imsave('ImageFrames/'+OPERATIONS[0][0]+'Dif'+str(0)+'.png', imagediff)
    if imexp == 'MaskAndDiff':
        for m in range(1,OPERATIONS[0][2]):
            IMT      = I.IMRES[m].copy()
            EMPTY = 255*np.ones((rowrange,colrange))
            EMPTY[0,0] = 255
            for pix in range(len(s_coord)):
                row = s_coord[pix][0]-mirr; col = s_coord[pix][1]-micr
                EMPTY[row,col] = 255-np.abs(int(IMT[row,col])-int(I.IMG[0][row,col]))
            ImDiff.append(EMPTY)
            ResIm.append(IMT[:,:])
        for m in range(0,OPERATIONS[0][2]):
            HorStack = np.hstack((ResIm[m],ImDiff[m]))
            VertPad = 255*np.ones((50,HorStack.shape[1]))
            IMCOMPARISON.append(np.vstack((VertPad,HorStack)))
            imagecat = cmap(norm(IMCOMPARISON[m]))
            imageres = cmap(norm(ResIm[m]))
            imagediff= cmap(normdif(ImDiff[m]))
            plt.imsave('ImageFrames/'+OPERATIONS[0][0]+'Res'+str(m)+'.png', imageres)
            plt.imsave('ImageFrames/'+OPERATIONS[0][0]+'Dif'+str(m)+'.png', imagediff)
            plt.imsave('ImageFrames/'+OPERATIONS[0][0]+'ResDif'+str(m)+'.png', imagecat)
        for m in range(0,OPERATIONS[0][2]):
            image = cv2.imread('ImageFrames/'+OPERATIONS[0][0]+'ResDif'+str(m)+'.png')  
            texted_image =cv2.putText(img=np.copy(image), text="Repair Iteration"+str(m), org=(20,20),fontFace=3, fontScale=1, color=(0,0,0), thickness=3)
            plt.imshow(texted_image)
            plt.show()

# map the normalized data to colors
# image is now RGBA (512x512x4) 

# save the image
