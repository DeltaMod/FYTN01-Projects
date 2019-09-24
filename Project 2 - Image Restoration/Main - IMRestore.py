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

import numpy as np 
import math
import matplotlib.pyplot as plt
from random import randint
import scipy.misc.imread as imr
plt.rcParams['axes.grid'] = True
"""