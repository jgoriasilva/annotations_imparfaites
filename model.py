import tensorflow as tf
print('Using Tensorflow version', tf.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
#from tensorflow.compat.v1.keras.backend import set_session
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
#sess = tf.compat.v1.Session(config=config)
#set_session(sess)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

# matplotlib default values
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# local package
from dlia_tools.u_net import u_net
from dlia_tools.keras_image2image import DeadLeavesWithSegmGenerator
from dlia_tools.keras_custom_loss import jaccard2_loss
from dlia_tools.random_image_generator import AdditiveGaussianNoise, draw_ring #, addGaussianNoise
from dlia_tools.random_image_generator import ROG_disks, ROG_rings, RandomPosGenUniform, RandomIntGenUniform
from dlia_tools.eval import jaccard

# Load data
data=np.load(os.path.join('data','data.npz'))['data'] # array of rings' parameters for each im: presence, center, radia, gray lev.
img_gt=np.load(os.path.join('data','images.npz'))['img_gt'] # gt
img_noise=np.load(os.path.join('data','images.npz'))['img_noise'] # noisy images
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
img_number = data.shape[0]
img_rows = img_gt.shape[1]
img_cols = img_gt.shape[2]
img_channels = img_gt.shape[3]

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

'''
if input('Save initial weights [y/n]? ') == 'y':
	model.save_weights(os.path.join('weights','start_weights.h5'))
if input('Print model [y/n]? ') == 'y':
	print(model.summary())
'''

# fit params
batch_size = 128
nb_epoch = 4000 # nb. max d'époques
patience = 20 # nb max. d'époques sans amélioration de la validation loss
n_train = 10000
n_val = 1000
n_test = 100

# database for X
X_train=img_noise[:n_train,:,:,:]
X_val=img_noise[n_train:n_train+n_val,:,:,:]
X_test=img_noise[n_train+n_val:n_train+n_val+n_test,:,:,:]

train_type = 'oubli'
# train_type = input('Train type [control/oubli]: ')
if train_type == 'control': 
	Y_train=img_gt[:n_train,:,:,:]
	Y_val=img_gt[n_train:n_train+n_val,:,:,:]
	Y_test=img_gt[n_train+n_val:n_train+n_val+n_test,:,:,:]
	model.load_weights(os.path.join('weights','start_weights.h5'))
	# Early stopping
	es= EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto', restore_best_weights=True)
	# Save training metrics regularly
	csv_logger = CSVLogger(os.path.join('logs','training_control.log'))
	verbose = 1
	history = model.fit(X_train, Y_train,
   	    	            batch_size=batch_size,
    	                epochs=nb_epoch,
                	    validation_data=(X_val, Y_val),
                    	shuffle=True,
                    	verbose=verbose,
                    	callbacks=[es, csv_logger])
	print("Best validation loss: %.5f" % (np.min(history.history['val_loss'])))
	print("at: %d" % np.argmin(history.history['val_loss']))
	model.save_weights(os.path.join('weights','model_control.h5'))
	print("Saved model to disk")
	
	# Training curve
	plt.rcParams['figure.figsize'] = (10.0, 8.0)
	plt.plot(history.epoch, history.history['loss'], label='train')
	plt.plot(history.epoch, history.history['val_loss'], label='val')
	plt.title('Training performance')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend()
	plt.ylim(0.0, 0.9)
	plt.savefig(os.path.join('images','training_control.png'))

elif train_type == 'oubli':
	distortion_log = open(os.path.join('logs','distortion_oubli.log'),'w')
	for proba_oversight in range(0, 100, 5):
		proba_oversight /= 100
		# Generate the images
		img_imp_gt=np.zeros((img_number,img_rows,img_cols,1))
		for i in range(img_number):
			im3=img_imp_gt[i,:,:,0]
			n_rings = data[i, :, 0].sum()
			for j in range(n_rings):
				if data[i][j][0]==1:
					x=data[i][j][1]
					y=data[i][j][2]
					r1=data[i][j][3]
					r2=int(r1*rad_ratio)
					v=data[i][j][4]
					if np.random.rand() > proba_oversight :
						draw_ring(im3,x,y,r1,r2,1)
		# Labels
		Y_train=img_imp_gt[:n_train,:,:,:]
		Y_val=img_imp_gt[n_train:n_train+n_val,:,:,:]
		Y_test=img_gt[n_train+n_val:n_train+n_val+n_test,:,:,:]

		# Load weights
		model.load_weights(os.path.join('weights','start_weights.h5'))

  		# Training
  		# Save training metrics regularly
		param_str = str(int(100*proba_oversight))
		csv_logger = CSVLogger(os.path.join('logs','training_log_oubli'+param_str+'.log'))
  		# Early stopping
		es= EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto', restore_best_weights=True)
		verbose = 2
		history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=nb_epoch,
                      validation_data=(X_val, Y_val),
                      shuffle=True,
                      verbose=verbose,
                      callbacks=[es, csv_logger])
  
  		# serialize weights to HDF5
		model.save_weights(os.path.join('weights','model_oubli_'+param_str+'.h5'))
		print('Saved model oubli '+param_str+' to disk')
		
		# Training curve
		plt.rcParams['figure.figsize'] = (10.0, 8.0)
		plt.plot(history.epoch, history.history['loss'], label='train')
		plt.plot(history.epoch, history.history['val_loss'], label='val')
		plt.title('Training performance')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend()
		plt.ylim(0.0, 0.9)
		plt.savefig(os.path.join('images','training_curve_oubli'+param_str+'.png'))
		plt.cla()	
		plt.clf()
		plt.close()
		
 	
		# Distortion between gt and labels 	
		distortion_log.write('Distortion between ground truth and labels for oubli ' + str(proba_oversight) + ' : ' + str(jaccard(img_gt, img_imp_gt)) + '\n')
	distortion_log.close()
