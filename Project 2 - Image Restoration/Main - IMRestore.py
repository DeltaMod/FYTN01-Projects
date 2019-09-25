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
plt.rcParams['figure.dpi']   = 150
plt.rcParams['axes.grid'] = False

original = imread('EM_ScreamGS.bmp')[:,:,0]
Dim   = original.shape
rowrange = Dim[0]; colrange = Dim[1]

mask = imread('mask2.bmp')[:,:,0]/255
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
dmgI = original*mask
plt.figure()
plt.imshow(dmgI, cmap='gray')
plt.title('Damaged image')
plt.show()

IMRes = np.zeros([rowrange,colrange])

h = 1 #Lattice Parameter 
LO =  np.array([[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]])
CC =  np.array([[1, 1,   1],
                 [1, 0.5, 1],
                 [1, 1,   1]])
LC = np.array([[1,  1,   1],
                 [1, 0.75, 1],
                 [1, 1,   1]])
for row in range(rowrange-1):
    for col in range(colrange-1):
        
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
