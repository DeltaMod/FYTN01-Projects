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
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi']   = 200
plt.rcParams['axes.grid'] = False

original = imread('EM_ScreamGS_lowres.bmp')[:,:,0]
Dim   = original.shape
rowrange = Dim[0]; colrange = Dim[1]

mask = imread('mask_lowres2.bmp')[:,:,0]/255
plt.figure()
plt.imshow(original, cmap='gray')
plt.title('Original Image')
plt.show()

##
plt.figure()
plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.show()

##
maskI    = original*mask                                                                             #We make the original image
#For the mirroring, we need three types - corner cuts (both), vertical cuts (column pixels only), vertical cuts (row pixels only)
mirrIB   = np.array([[maskI[row+1,col+1] for col in range(colrange-2)] for row in range(rowrange-2)]) #Then take the mirror plane to be [start+1:end-1]
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

h = 1 #Lattice Parameter 
KernelMode = 'Gauss-5x5'     # EdgeLapl|IEdgeLapl|Gauss-3x3|Gauss-5x5|Unsharp-5x5
SearchMode = 'PixelsOnly'    # FullImage|PixelsOnly
if KernelMode == 'EdgeLapl':
    LO =  np.array([[0,  1, 0],
                    [1, -4, 1], #Edge Laplacian
                    [0,  1, 0]])
    
elif KernelMode == 'IEdgeLapl':                    
    LO =  np.array([ [-1, -1, -1],
                     [-1,  8, -1], #Inverse Edge Laplacian
                     [-1, -1, -1]])

elif KernelMode == 'Gauss-3x3':
    LO =  1/16*np.array([ [1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) #Gaussian Blur 3x3
elif KernelMode == 'Gauss-5x5':
    LO =  1/256*np.array([ [1,  4,  6,  4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1,  4,  6,  4, 1]]) #Gaussian Blur 5x5
elif KernelMode == 'Unsharp-5x5':
    LO = -1/256*np.array([ [1,  4,    6,  4, 1],
                          [4, 16,   24, 16, 4],
                          [6, 24, -476, 24, 6],
                          [4, 16,   24, 16, 4],
                          [1,  4,    6,  4, 1]]) #Gaussian Blur 5x5
dLO = int(np.floor(len(LO)/2)); uLO = (len(LO) - dLO); 
def s_transform(mask):
    s_map = []
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y] == 0:
                s_map.append([x+mirr,y+micr])
    return(s_map)
s_coord = s_transform(maskI)

for repeats in range(10):
    if SearchMode == 'FullImage':
        for row in range(mirr,mirr+rowrange):s
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

"""

Legacy Code, which still uses corner and edge conditions - obsolete to the max, considering what we have now
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
"""      


plt.figure()
plt.imshow(IMRes, cmap='gray')
plt.title('Restored Image?')
plt.show()
