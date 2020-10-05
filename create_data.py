import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
from dlia_tools.random_image_generator import AdditiveGaussianNoise, draw_ring

# **** input data generator parameters
img_rows, img_cols = 32, 32 # dimension des images
img_channels = 1 # nombre de canaux par image
gauss_n_std = 40 # écart type du bruit blanc qui sera ajouté
nb_obj_l = 1 # nombre minimal d'anneaux par image
nb_obj_h = 4 # nombre maximal d'anneaux par image
#r1_disk_l = 2
#r1_disk_h = 4
r1_ring_l = 4 # rayon extérieur minimal
r1_ring_h = 8 # rayon extérieur maximal
rad_ratio= 0.5 # rapport rayon intérieur/rayon extérieur
gray_l = 20 # niveau de gris minimum
gray_h = 200 # niveau de gris maximum
norm = 255  # constante de normalisation
img_number=11100 # nombre total d'images dans la base (train:10000 + val:1000 + test:100)

data=np.zeros((img_number, nb_obj_h, 5), dtype=int)
for i in range(img_number):
    n_rings = np.random.randint(nb_obj_l, nb_obj_h+1)
    for j in range(n_rings):
        centre_x=np.random.randint(0, img_rows)
        centre_y=np.random.randint(0, img_cols)
        r1=np.random.randint(r1_ring_l, r1_ring_h)
        v=np.random.randint(gray_l, gray_h)
        data[i][j]=np.array([1,centre_x, centre_y, r1, v])
np.savez(os.path.join('data','data'),data=data)

noise = AdditiveGaussianNoise(gauss_n_std)
img_gt=np.zeros((img_number,img_rows,img_cols,1)) # images binaires
img_gray=np.zeros((img_number,img_rows,img_cols,1)) # images à niveaux de gris
img_noise=np.zeros((img_number,img_rows,img_cols,1)) # images bruitées
for i in range(img_number):
    im1=img_gray[i,:,:,0]
    im2=img_gt[i,:,:,0]
    n_rings = data[i, :, 0].sum()
    for j in range(n_rings):
        if data[i][j][0]==1:
            x=data[i][j][1]
            y=data[i][j][2]
            r1=data[i][j][3]
            r2=int(r1*rad_ratio)
            v=data[i][j][4]
            draw_ring(im1,x,y,r1,r2,v)
            draw_ring(im2,x,y,r1,r2,1)
    #img_noise[i,:,:,0]=addGaussianNoise(im1,gauss_n_std)
    img_noise[i,:,:,0]=im1
    noise(img_noise[i,:,:,0])

np.savez(os.path.join('data','images'),img_gt=img_gt,img_noise=img_noise)

plt.rcParams['figure.figsize'] = (10.0, 8.0)
index = np.random.randint(img_gt.shape[0])
plt.subplot(1, 3, 1)
plt.imshow(img_gt[index, :, :, 0])
plt.title("Truth")
plt.subplot(1, 3, 2)
plt.imshow(img_gray[index, :, :, 0])
plt.title("Gray image")
plt.subplot(1, 3, 3)
plt.imshow(img_noise[index, :, :, 0])
plt.title("Noisy image (input)")
plt.show()

