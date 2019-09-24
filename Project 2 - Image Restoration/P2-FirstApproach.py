from matplotlib.pyplot import imread
import matplotlib.pyplot as plt

original = imread('EM_ScreamGS.bmp')[:,:,0]
mask = imread('mask2.bmp')[:,:,0]

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
damaged_img = (255-original)*(mask/255)
plt.figure()
plt.imshow(damaged_img, cmap='gray_r')
plt.title('Damaged image')
plt.show()
