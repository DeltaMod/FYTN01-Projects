from matplotlib.pyplot import imread
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

original = imread('EM_ScreamGS_lowres.bmp')[:,:,0]
mask = imread('mask_lowres2.bmp')[:,:,0]
damaged_img = (original)*(mask/255)

plt.figure()
plt.imshow(original, cmap='gray')
plt.title('Original Image')
plt.show()

##
plt.figure()
plt.imshow(mask, cmap='gray_r')
plt.title('Mask')
plt.show()

##
plt.figure()
plt.imshow(damaged_img, cmap='gray')
plt.title('Damaged image')
plt.show()

##
def s_transform(mask):
    s_map = []
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x,y] == 0:
                s_map.append([x,y])
    return(s_map)

s_coord = s_transform(mask)
A = np.zeros([len(s_coord)]*2, dtype=np.int8)
B = np.zeros(len(s_coord))
for s in range(len(s_coord)):
    x, y = s_coord[s]
    A[s,s] = -4
    if [x-1, y] in s_coord :
        A[s, s_coord.index([x-1,y])] = 1
    elif x-1 >= 0 :
        B[s] -= damaged_img[x-1,y]

    if [x, y-1] in s_coord :
        A[s, s_coord.index([x,y-1])] = 1
    elif y-1 >= 0 :
        B[s] -= damaged_img[x,y-1]

    if [x+1, y] in s_coord :
        A[s, s_coord.index([x+1,y])] = 1
    elif x+1 < damaged_img.shape[0] :
        B[s] -= damaged_img[x+1,y]

    if [x, y+1] in s_coord :
        A[s, s_coord.index([x,y+1])] = 1
    elif y+1 < damaged_img.shape[1] :
        B[s] -= damaged_img[x,y+1]


##
sol = np.linalg.solve(A,B)

##
repaired_img = deepcopy(damaged_img)
for s in range(len(sol)):
    repaired_img[s_coord[s][0], s_coord[s][1]] = sol[s]

##
plt.figure()
plt.imshow(repaired_img, cmap='gray')
plt.title('Repaired Image')
plt.show()
