import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
os.environ["CUDA_VISIBLE_DEVICES"] ='1'
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
# local package
from dlia_tools.u_net import u_net
from dlia_tools.keras_image2image import DeadLeavesWithSegmGenerator
from dlia_tools.keras_custom_loss import jaccard2_loss
from dlia_tools.random_image_generator import AdditiveGaussianNoise, draw_ring #, addGaussianNoise
from dlia_tools.random_image_generator import ROG_disks, ROG_rings, RandomPosGenUniform, RandomIntGenUniform
from dlia_tools.eval import jaccard

data=np.load(os.path.join('data','data.npz'))['data'] # array of rings' parameters for each im: presence, center, radia, gray lev.
img_gt=np.load(os.path.join('data','images.npz'))['img_gt'] # gt
img_noise=np.load(os.path.join('data','images.npz'))['img_noise'] # noisy images
img_number = data.shape[0]
img_rows = img_gt.shape[1]
img_cols = img_gt.shape[2]
img_channels = img_gt.shape[3]
n_train = 10000
n_val = 1000
n_test = 100

# architecture params
nb_filters_0 = 8
sigma_noise = 0.01
# ****  deep learning model
shape = (img_rows, img_cols, img_channels)
model = u_net(shape, nb_filters_0, sigma_noise=sigma_noise)
opt_name = 'sgd'  # choices:adadelta; sgd, rmsprop, adagrad, adam
loss_func = jaccard2_loss  # mse, mae, binary_crossentropy, jaccard2_loss
if opt_name == "sgd":
    opt = SGD(lr=0.1)
elif opt_name == "rmsprop":
    opt = RMSprop()
elif opt_name == "adagrad":
    opt = Adagrad()
elif opt_name == "adadelta":
    opt = Adadelta()
elif opt_name == "adam":
    opt = Adam(lr=1e-5)
else:
    raise NameError("Wrong optimizer name")
model.compile(loss=loss_func, optimizer=opt)

X_test=img_noise[n_train+n_val:n_train+n_val+n_test,:,:,:]
Y_test=img_gt[n_train+n_val:n_train+n_val+n_test,:,:,:]
jaccard_log = open(os.path.join('logs','jaccard.log'),'w')
'''
# Control model
model.load_weights(os.path.join('weights','model_control.h5'))
Y_pred = model.predict(X_test)
jaccard_log.write('Jaccard on test set for control model = ' + str(jaccard(Y_test, Y_pred)) + '\n')
print('Jaccard on test set for control model = ', jaccard(Y_test, Y_pred))
'''

total_run = int(input('How many sequential runs?: '))
initial_X_test = np.zeros((100,32,32,1))
initial_X_test[:,:,:,:] = X_test[:,:,:,:]
gauss_n_std = 40
noise = AdditiveGaussianNoise(gauss_n_std)
look = 7
for oubli in range(0,105,5):
	model.load_weights(os.path.join('weights','model_oubli_'+str(oubli)+'.h5'))
	for iterate in range (0,total_run):	
		Y_pred = model.predict(X_test)
		jaccard_log.write('Jaccard on test set for oubli ' + str(oubli)+ ' on run ' + str(iterate) + ' ' + str(jaccard(Y_test, Y_pred))+ '\n')
		print('Jaccard on test set (Y) for oubli '+str(oubli) + ' on run ' + str(iterate) + ' ' + str(jaccard(Y_test, Y_pred)))
		X_test[:,:,:,:] = Y_pred*255
		for image in range(len(X_test)):
			noise(X_test[image,:,:,0])
		'''if oubli >= 90:
			plt.subplot(1,2,1)
			plt.imshow(Y_pred[look])
			plt.subplot(1,2,2)
			plt.imshow(X_test[look])
			plt.show()'''
	X_test[:,:,:,:] = initial_X_test

'''
for taille in range(r1_ring_l,r1_ring_h):
  model.load_weights(os.path.join('weights','model_taille_'+str(taille)+'.h5'))
  Y_pred = model.predict(X_test)
  jaccard_log.write('Jaccard on test set for taille '+str(taille)+' = ' + str(jaccard(Y_test, Y_pred))+ '\n')
  print('Jaccard on test set for taille '+str(taille)+' = ', jaccard(Y_test, Y_pred))

# Random taille
model.load_weights(os.path.join('weights','model_taille_random.h5'))
Y_pred = model.predict(X_test)
jaccard_log.write('Jaccard on test set for random taille = ' + str(jaccard(Y_test, Y_pred)) + '\n')
print('Jaccard on test set for random taille = ', jaccard(Y_test, Y_pred))

# Random deplacement
model.load_weights(os.path.join('weights','model_deplacement.h5'))
Y_pred = model.predict(X_test)
jaccard_log.write('Jaccard on test set for deplacement = ' + str(jaccard(Y_test, Y_pred)) + '\n')
print('Jaccard on test set for deplacement = ', jaccard(Y_test, Y_pred))
'''
jaccard_log.close()
