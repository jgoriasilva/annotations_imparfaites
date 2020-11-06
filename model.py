import tensorflow as tf
print('Using Tensorflow version', tf.__version__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='3'
#from tensorflow.compat.v1.keras.backend import set_session
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
#sess = tf.compat.v1.Session(config=config)
#set_session(sess)

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
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

train_type = 'mean'
# train_type = input('Train type [control/oubli/taille/mean/deplace]: ')
if train_type == 'control': 
	Y_train=img_gt[:n_train,:,:,:]
	Y_val=img_gt[n_train:n_train+n_val,:,:,:]
	Y_test=img_gt[n_train+n_val:n_train+n_val+n_test,:,:,:]
	model.load_weights(os.path.join('weights','start_weights.h5'))
	# Early stopping
	es= EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto', restore_best_weights=True)
	# Save training metrics regularly
	csv_logger = CSVLogger(os.path.join('logs','training','training_control.log'))
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
	plt.savefig(os.path.join('images','training','training_control.png'))

elif train_type == 'oubli':
	distortion_log = open(os.path.join('logs','distortion','distortion_oubli.log'),'w')
	jaccard_log = open(os.path.join('logs','jaccard','jaccard_oubli.log'),'w')
	
	modifications = 'y'
	# modifications = input('make modifications to the outputs? ')
	runs_train = 2
	# runs_train = int(input('how many training runs? '))
	for proba_oversight in range(0, 100, 5):
		proba_oversight /= 100
		oubli_str = str(int(100*proba_oversight))
		runs_str = str(runs_train)
		patience = 20
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

		plt.figure()
		plt.title('Inputs, predictions of the network and ground truth for test set, forgetting probability of {}%'.format(proba_oversight))
		for i in range(5):
			plt.subplot(5,6,i*6+1).title.set_text('input')
			plt.imshow(X_train[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+2).title.set_text('imperfect label')
			plt.imshow(Y_train[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+3).title.set_text('ground truth')
			plt.imshow(img_gt[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+4).title.set_text('input')
			plt.imshow(X_val[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+5).title.set_text('imperfect label')
			plt.imshow(Y_val[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+6).title.set_text('ground truth')
			plt.imshow(img_gt[n_train+i])
			plt.axis('off')	
		plt.savefig(os.path.join('images','oubli','initial_oubli_'+oubli_str+'.png'))
		plt.clf()
		plt.close()
	
		for run in range(runs_train):
			continue_train = 0
			
			K.clear_session()
			model = u_net(shape, nb_filters_0, sigma_noise=sigma_noise)
			model.compile(loss=loss_func, optimizer=opt)
			print('oubli {}, run {}, patience {}'.format(proba_oversight,run,patience))
			# Load weights
			model.load_weights(os.path.join('weights','start_weights.h5'))
  			# Training
  			# Save training metrics regularly
			csv_logger = CSVLogger(os.path.join('logs','training','oubli','training_log_oubli_'+oubli_str+'_'+str(run)+'.log'))
	  		# Early stopping
			es= EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='auto', restore_best_weights=True)
			verbose = 2
			history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=nb_epoch,
                      validation_data=(X_val, Y_val),
                      shuffle=True,
                      verbose=verbose,
                      callbacks=[es, csv_logger])
			
			jaccard_log.write('Jaccard on test set for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_test,model.predict(X_test)))+'\n') 
			print('Jaccard on test set for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_test,model.predict(X_test)))) 
			
			plt.figure()
			for i in range(5):
				plt.subplot(5,6,i*6+1).title.set_text('input')
				plt.imshow(X_test[i*4])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+2).title.set_text('prediction')
				plt.imshow(model.predict(X_test)[i*4])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+3).title.set_text('truth')
				plt.imshow(Y_test[i*4])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+4).title.set_text('input')
				plt.imshow(X_test[(i+1)*18])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+5).title.set_text('prediction')
				plt.imshow(model.predict(X_test)[(i+1)*18])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+6).title.set_text('truth')
				plt.imshow(Y_test[(i+1)*18])
				plt.axis('off')	
			plt.savefig(os.path.join('images','oubli','oubli_'+oubli_str+'_run_'+str(run)+'.png'))
			plt.clf()
			plt.close()
			
			'''
			for image in model.predict(X_test):
				black = 0
				gray = 0
				for row in image:
					for pixel in row:
						if pixel[:]*256 < 20:
							black += 1
						if pixel[:]*256 < 240:
							gray += 1
				
				rate = float(black)/float(gray)
				if(rate < 0.80):
					continue_train = 1
					break
			
			if(continue_train == 0):
				break
			'''
			'''
			jaccard_train_before = jaccard(Y_train,img_imp_gt[:n_train,:,:,:])
			jaccard_val_before = jaccard(Y_val,img_imp_gt[n_train:n_train+n_val,:,:,:])
			jaccard_log.write('Jaccard on train set before seuil for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:]))+'\n') 
			print('Jaccard on train set before seuil for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:]))) 
			'''
			
			if modifications == 'y':
				Y_train = np.maximum(model.predict(X_train), img_imp_gt[:n_train,:,:,:])
				Y_train_tmp = np.zeros((n_train,img_rows,img_cols,img_channels))
				Y_train_tmp = Y_train[:,:,:,:]
				for img in Y_train:
					threshold = filters.threshold_otsu(img)
					img[img >= threshold] = 1
					img[img < threshold] = 0
								
				Y_val = np.maximum(model.predict(X_val), img_imp_gt[n_train:n_train+n_val,:,:,:])
				Y_val_tmp = np.zeros((n_train,img_rows,img_cols,img_channels))
				Y_val_tmp = Y_val[:,:,:,:]
				for img in Y_val:
					threshold = filters.threshold_otsu(img)
					img[img >= threshold] = 1
					img[img < threshold] = 0

			elif modifications == 'n':
				Y_train = np.zeros((n_train,img_rows,img_cols,img_channels))
				Y_train = model.predict(X_train)[:,:,:,:]
			
				Y_val = np.zeros((n_val,img_rows,img_cols,img_channels))
				Y_val = model.predict(X_val)[:,:,:,:]
			
			'''
			jaccard_train_after = jaccard(Y_train,img_imp_gt[:n_train,:,:,:])
			jaccard_val_after = jaccard(Y_val,img_imp_gt[n_train:n_train+n_val,:,:,:])
			jaccard_log.write('Jaccard on train set after seuil for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:]))+'\n') 
			print('Jaccard on train set after seuil for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:])))

			if jaccard_train_after < jaccard_train_before and jaccard_val_after < jaccard_val_before:
				Y_train = Y_train_tmp
				Y_val = Y_val_tmp
				quit_train = 1
			
			'''
			
			# patience -= 2
		
		# serialize weights to HDF5
		model.save_weights(os.path.join('weights','oubli','model_oubli_'+oubli_str+'.h5'))
		print('Saved model oubli '+oubli_str+' to disk')
		'''
		# Training curve
		plt.rcParams['figure.figsize'] = (10.0, 8.0)
		plt.plot(history.epoch, history.history['loss'], label='train')
		plt.plot(history.epoch, history.history['val_loss'], label='val')
		plt.title('Training performance')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend()
		plt.ylim(0.0, 0.9)
		plt.savefig(os.path.join('images','training','training_curve_oubli'+param_str+'.png'))
		plt.cla()	
		plt.clf()
		plt.close()
		'''
 	
		# Distortion between gt and labels 	
		distortion_log.write('Jaccard between ground truth and labels for oubli ' + oubli_str + ' : ' + str(jaccard(img_gt, img_imp_gt)) + '\n')
	distortion_log.close()
	jaccard_log.close()

elif train_type == 'taille':	
	distortion_log = open(os.path.join('logs','distortion','distortion_taille.log'),'w')
	jaccard_log = open(os.path.join('logs','jaccard','jaccard_taille.log'),'w')
	
	for taille in range(20, 110,5):
		taille /= 10
		taille_str = str(taille)
		
		patience = 20
	
		# Generate the images
		img_imp_gt=np.zeros((img_number,img_rows,img_cols,1))
		for i in range(img_number):
			im3=img_imp_gt[i,:,:,0]
			n_rings = data[i, :, 0].sum()
			for j in range(n_rings):
				if data[i][j][0]==1:
					x=data[i][j][1]
					y=data[i][j][2]
					r1=taille
					r2=int(r1*rad_ratio)
					v=data[i][j][4]
					draw_ring(im3,x,y,r1,r2,1)

		run = 0
		
		print('taille {}, run {}, patience {}'.format(taille,run,patience))
		
		# Labels
		Y_train=img_imp_gt[:n_train,:,:,:]
		Y_val=img_imp_gt[n_train:n_train+n_val,:,:,:]
		Y_test=img_gt[n_train+n_val:n_train+n_val+n_test,:,:,:]
		
		plt.figure()
		for i in range(5):
			plt.subplot(5,6,i*6+1).title.set_text('input')
			plt.imshow(X_train[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+2).title.set_text('imperfect label')
			plt.imshow(Y_train[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+3).title.set_text('ground truth')
			plt.imshow(img_gt[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+4).title.set_text('input')
			plt.imshow(X_val[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+5).title.set_text('imperfect label')
			plt.imshow(Y_val[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+6).title.set_text('ground truth')
			plt.imshow(img_gt[n_train+i])
			plt.axis('off')	
		plt.savefig(os.path.join('images','taille','initial_taille_'+taille_str+'.png'))
		plt.clf()
		plt.close()

		K.clear_session()
		model = u_net(shape, nb_filters_0, sigma_noise=sigma_noise)
		model.compile(loss=loss_func, optimizer=opt)

		# Load weights
		model.load_weights(os.path.join('weights','start_weights.h5'))
  		
		# Training
  		# Save training metrics regularly
		csv_logger = CSVLogger(os.path.join('logs','training','taille','training_log_taille_'+taille_str+'.log'))
  		# Early stopping
		es= EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='auto', restore_best_weights=True)
		verbose = 2
		history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=nb_epoch,
                      validation_data=(X_val, Y_val),
                      shuffle=True,
                      verbose=verbose,
                      callbacks=[es, csv_logger])
  		# serialize weights to HDF5
		model.save_weights(os.path.join('weights','taille','model_taille_'+taille_str+'.h5'))
		print('Saved model taille '+taille_str+' to disk')

		jaccard_log.write('Jaccard on test set for oubli '+taille_str+' run '+str(run)+' '+str(jaccard(Y_test,model.predict(X_test)))+'\n') 
		print('Jaccard on test set for oubli '+taille_str+' run '+str(run)+' '+str(jaccard(Y_test,model.predict(X_test)))) 
		
		plt.figure()
		for i in range(5):
			plt.subplot(5,6,i*6+1).title.set_text('input')
			plt.imshow(X_test[i*4])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+2).title.set_text('prediction')
			plt.imshow(model.predict(X_test)[i*4])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+3).title.set_text('truth')
			plt.imshow(Y_test[i*4])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+4).title.set_text('input')
			plt.imshow(X_test[(i+1)*18])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+5).title.set_text('prediction')
			plt.imshow(model.predict(X_test)[(i+1)*18])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+6).title.set_text('truth')
			plt.imshow(Y_test[(i+1)*18])
			plt.axis('off')	
		plt.savefig(os.path.join('images','taille','taille_'+taille_str+'_run_'+str(run)+'.png'))
		plt.clf()
		plt.close()
		
		'''	
		# Training curve
		plt.rcParams['figure.figsize'] = (10.0, 8.0)
		plt.plot(history.epoch, history.history['loss'], label='train')
		plt.plot(history.epoch, history.history['val_loss'], label='val')
		plt.title('Training performance')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend()
		plt.ylim(0.0, 0.9)
		plt.savefig(os.path.join('images','training','training_curve_taille'+param_str+'.png'))
		plt.cla()	
		plt.clf()
		plt.close()
		'''
 		
		# Distortion between gt and labels 	
		distortion_log.write('Jaccard between ground truth and labels for taille ' + taille_str + ' : ' + str(jaccard(img_gt, img_imp_gt)) + '\n')
	distortion_log.close()
	jaccard_log.close()

elif train_type == 'mean':	
	distortion_log = open(os.path.join('logs','distortion','distortion_mean.log'),'w')
	jaccard_log = open(os.path.join('logs','jaccard','jaccard_mean.log'),'w')
	patience = 5
	runs_train = 3
	modifications = 'n'
	
	for mean in [4,5,6,7]:
		for std in range(10,25,5):
			std /= 10
		
			# Generate the images
			img_imp_gt=np.zeros((img_number,img_rows,img_cols,1))
			for i in range(img_number):
				im3=img_imp_gt[i,:,:,0]
				n_rings = data[i, :, 0].sum()
				for j in range(n_rings):
					if data[i][j][0]==1:
						x=data[i][j][1]
						y=data[i][j][2]
						r1 = int(round(np.random.normal(loc=mean, scale=std)))
						r2=int(r1*rad_ratio)
						v=data[i][j][4]
						draw_ring(im3,x,y,r1,r2,1)
			
			# Labels
			Y_train=img_imp_gt[:n_train,:,:,:]
			Y_val=img_imp_gt[n_train:n_train+n_val,:,:,:]
			Y_test=img_gt[n_train+n_val:n_train+n_val+n_test,:,:,:]
			
			plt.figure()
			for i in range(5):
				plt.subplot(5,6,i*6+1).title.set_text('input')
				plt.imshow(X_train[i])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+2).title.set_text('imperfect label')
				plt.imshow(Y_train[i])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+3).title.set_text('ground truth')
				plt.imshow(img_gt[i])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+4).title.set_text('input')
				plt.imshow(X_val[i])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+5).title.set_text('imperfect label')
				plt.imshow(Y_val[i])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+6).title.set_text('ground truth')
				plt.imshow(img_gt[n_train+i])
				plt.axis('off')	
			plt.savefig(os.path.join('images','mean','initial_mean_{}_std_{}.png'.format(mean,std)))
			plt.clf()
			plt.close()

			for run in range(runs_train):
				print('mean {}, std {}, run {}, patience {}'.format(mean,std,run,patience))

				K.clear_session()
				model = u_net(shape, nb_filters_0, sigma_noise=sigma_noise)
				model.compile(loss=loss_func, optimizer=opt)

				# Load weights
				model.load_weights(os.path.join('weights','start_weights.h5'))
				
				# Training
				# Save training metrics regularly
				csv_logger = CSVLogger(os.path.join('logs','training','mean','training_log_mean_{}_std_{}.log'.format(mean,std)))
				# Early stopping
				es= EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='auto', restore_best_weights=True)
				verbose = 2
				history = model.fit(X_train, Y_train,
							batch_size=batch_size,
							epochs=nb_epoch,
							validation_data=(X_val, Y_val),
							shuffle=True,
							verbose=verbose,
							callbacks=[es, csv_logger])

				jaccard_log.write('Jaccard on test set for mean {} std {} run {} = {}\n'.format(mean,std,run,jaccard(Y_test,model.predict(X_test)))) 
				print('Jaccard on test set for mean {} std {} run {} = {}'.format(mean,std,run,jaccard(Y_test,model.predict(X_test)))) 
				
				plt.figure()
				for i in range(5):
					plt.subplot(5,6,i*6+1).title.set_text('input')
					plt.imshow(X_test[i*4])
					plt.axis('off')
				for i in range(5):
					plt.subplot(5,6,i*6+2).title.set_text('prediction')
					plt.imshow(model.predict(X_test)[i*4])
					plt.axis('off')
				for i in range(5):
					plt.subplot(5,6,i*6+3).title.set_text('truth')
					plt.imshow(Y_test[i*4])
					plt.axis('off')
				for i in range(5):
					plt.subplot(5,6,i*6+4).title.set_text('input')
					plt.imshow(X_test[(i+1)*18])
					plt.axis('off')
				for i in range(5):
					plt.subplot(5,6,i*6+5).title.set_text('prediction')
					plt.imshow(model.predict(X_test)[(i+1)*18])
					plt.axis('off')
				for i in range(5):
					plt.subplot(5,6,i*6+6).title.set_text('truth')
					plt.imshow(Y_test[(i+1)*18])
					plt.axis('off')	
				plt.savefig(os.path.join('images','mean','mean_{}_std_{}_run_{}.png'.format(mean,std,run)))
				plt.clf()
				plt.close()

				if modifications == 'y':
					Y_train = model.predict(X_train)
					Y_train_tmp = np.zeros((n_train,img_rows,img_cols,img_channels))
					Y_train_tmp = Y_train[:,:,:,:]
					for img in Y_train:
						threshold = filters.threshold_otsu(img)
						img[img >= threshold] = 1
						img[img < threshold] = 0
									
					Y_val = model.predict(X_val)
					Y_val_tmp = np.zeros((n_train,img_rows,img_cols,img_channels))
					Y_val_tmp = Y_val[:,:,:,:]
					for img in Y_val:
						threshold = filters.threshold_otsu(img)
						img[img >= threshold] = 1
						img[img < threshold] = 0

				elif modifications == 'n':
					Y_train = np.zeros((n_train,img_rows,img_cols,img_channels))
					Y_train = model.predict(X_train)[:,:,:,:]
				
					Y_val = np.zeros((n_val,img_rows,img_cols,img_channels))
					Y_val = model.predict(X_val)[:,:,:,:]
				
				'''
				jaccard_train_after = jaccard(Y_train,img_imp_gt[:n_train,:,:,:])
				jaccard_val_after = jaccard(Y_val,img_imp_gt[n_train:n_train+n_val,:,:,:])
				jaccard_log.write('Jaccard on train set after seuil for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:]))+'\n') 
				print('Jaccard on train set after seuil for oubli '+oubli_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:])))

				if jaccard_train_after < jaccard_train_before and jaccard_val_after < jaccard_val_before:
					Y_train = Y_train_tmp
					Y_val = Y_val_tmp
					quit_train = 1
				
				'''
				
				# patience -= 2
			
			# serialize weights to HDF5
			model.save_weights(os.path.join('weights','mean','model_mean_{}_std_{}.h5'.format(mean,std)))
			print('Saved model mean {} std {} to disk'.format(mean,std))

			'''	
			# Training curve
			plt.rcParams['figure.figsize'] = (10.0, 8.0)
			plt.plot(history.epoch, history.history['loss'], label='train')
			plt.plot(history.epoch, history.history['val_loss'], label='val')
			plt.title('Training performance')
			plt.ylabel('loss')
			plt.xlabel('epoch')
			plt.legend()
			plt.ylim(0.0, 0.9)
			plt.savefig(os.path.join('images','training','training_curve_mean'+param_str+'.png'))
			plt.cla()	
			plt.clf()
			plt.close()
			'''
			
			# Distortion between gt and labels 	
		distortion_log.write('Jaccard between ground truth and labels for mean {} std {}: {}\n'.format(mean,std,jaccard(img_gt, img_imp_gt)))
	distortion_log.close()
	jaccard_log.close()

elif train_type == 'deplace':
	distortion_log = open(os.path.join('logs','distortion','distortion_deplace.log'),'w')
	jaccard_log = open(os.path.join('logs','jaccard','jaccard_deplace.log'),'w')
	
	modifications = 'y'
	# modifications = input('make modifications to the outputs? ')
	runs_train = 1
	# runs_train = int(input('how many training runs? '))
	for deplace in range(1, 11):
		deplace_str = str(deplace)
		# runs_str = str(runs_train)
		patience = 20
		# Generate the images
		img_imp_gt=np.zeros((img_number,img_rows,img_cols,1))
		for i in range(img_number):
			im3=img_imp_gt[i,:,:,0]
			n_rings = data[i, :, 0].sum()
			for j in range(n_rings):
				if data[i][j][0]==1:
					x=data[i][j][1]+np.random.randint(low=0-deplace,high=deplace)
					y=data[i][j][2]+np.random.randint(low=0-deplace,high=deplace)
					r1=data[i][j][3]
					r2=int(r1*rad_ratio)
					v=data[i][j][4]
					draw_ring(im3,x,y,r1,r2,1)
		# Labels
		Y_train=img_imp_gt[:n_train,:,:,:]
		Y_val=img_imp_gt[n_train:n_train+n_val,:,:,:]
		Y_test=img_gt[n_train+n_val:n_train+n_val+n_test,:,:,:]

		plt.figure()
		for i in range(5):
			plt.subplot(5,6,i*6+1).title.set_text('input')
			plt.imshow(X_train[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+2).title.set_text('imperfect label')
			plt.imshow(Y_train[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+3).title.set_text('ground truth')
			plt.imshow(img_gt[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+4).title.set_text('input')
			plt.imshow(X_val[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+5).title.set_text('imperfect label')
			plt.imshow(Y_val[i])
			plt.axis('off')
		for i in range(5):
			plt.subplot(5,6,i*6+6).title.set_text('ground truth')
			plt.imshow(img_gt[n_train+i])
			plt.axis('off')	
		plt.savefig(os.path.join('images','deplace','initial_deplace_'+deplace_str+'.png'))
		plt.clf()
		plt.close()
	
		for run in range(runs_train):
			continue_train = 0
			
			K.clear_session()
			model = u_net(shape, nb_filters_0, sigma_noise=sigma_noise)
			model.compile(loss=loss_func, optimizer=opt)
			print('deplace {}, run {}, patience {}'.format(deplace,run,patience))
			# Load weights
			model.load_weights(os.path.join('weights','start_weights.h5'))
  			# Training
  			# Save training metrics regularly
			csv_logger = CSVLogger(os.path.join('logs','training','deplace','training_log_deplace_'+deplace_str+'_'+str(run)+'.log'))
	  		# Early stopping
			es= EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='auto', restore_best_weights=True)
			verbose = 2
			history = model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      epochs=nb_epoch,
                      validation_data=(X_val, Y_val),
                      shuffle=True,
                      verbose=verbose,
                      callbacks=[es, csv_logger])
			
			jaccard_log.write('Jaccard on test set for deplace '+deplace_str+' run '+str(run)+' '+str(jaccard(Y_test,model.predict(X_test)))+'\n') 
			print('Jaccard on test set for deplace '+deplace_str+' run '+str(run)+' '+str(jaccard(Y_test,model.predict(X_test)))) 

			plt.figure()
			for i in range(5):
				plt.subplot(5,6,i*6+1).title.set_text('input')
				plt.imshow(X_test[i*4])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+2).title.set_text('prediction')
				plt.imshow(model.predict(X_test)[i*4])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+3).title.set_text('truth')
				plt.imshow(Y_test[i*4])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+4).title.set_text('input')
				plt.imshow(X_test[(i+1)*18])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+5).title.set_text('prediction')
				plt.imshow(model.predict(X_test)[(i+1)*18])
				plt.axis('off')
			for i in range(5):
				plt.subplot(5,6,i*6+6).title.set_text('truth')
				plt.imshow(Y_test[(i+1)*18])
				plt.axis('off')	
			plt.savefig(os.path.join('images','deplace','deplace_'+deplace_str+'_run_'+str(run)+'.png'))
			plt.clf()
			plt.close()

			'''
			for image in model.predict(X_test):
				black = 0
				gray = 0
				for row in image:
					for pixel in row:
						if pixel[:]*256 < 20:
							black += 1
						if pixel[:]*256 < 240:
							gray += 1
				
				rate = float(black)/float(gray)
				if(rate < 0.80):
					continue_train = 1
					break
			
			if(continue_train == 0):
				break
			'''
			'''
			jaccard_train_before = jaccard(Y_train,img_imp_gt[:n_train,:,:,:])
			jaccard_val_before = jaccard(Y_val,img_imp_gt[n_train:n_train+n_val,:,:,:])
			jaccard_log.write('Jaccard on train set before seuil for deplace '+deplace_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:]))+'\n') 
			print('Jaccard on train set before seuil for deplace '+deplace_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:]))) 
			'''
			'''
			if modifications == 'y':
				Y_train = np.maximum(model.predict(X_train), img_imp_gt[:n_train,:,:,:])
				Y_train_tmp = np.zeros((n_train,img_rows,img_cols,img_channels))
				Y_train_tmp = Y_train[:,:,:,:]
				for img in Y_train:
					threshold = filters.threshold_otsu(img)
					img[img >= threshold] = 1
					img[img < threshold] = 0
								
				Y_val = np.maximum(model.predict(X_val), img_imp_gt[n_train:n_train+n_val,:,:,:])
				Y_val_tmp = np.zeros((n_train,img_rows,img_cols,img_channels))
				Y_val_tmp = Y_val[:,:,:,:]
				for img in Y_val:
					threshold = filters.threshold_otsu(img)
					img[img >= threshold] = 1
					img[img < threshold] = 0

			elif modifications == 'n':
				Y_train = np.zeros((n_train,img_rows,img_cols,img_channels))
				Y_train = model.predict(X_train)[:,:,:,:]
			
				Y_val = np.zeros((n_val,img_rows,img_cols,img_channels))
				Y_val = model.predict(X_val)[:,:,:,:]
			'''
			'''
			jaccard_train_after = jaccard(Y_train,img_imp_gt[:n_train,:,:,:])
			jaccard_val_after = jaccard(Y_val,img_imp_gt[n_train:n_train+n_val,:,:,:])
			jaccard_log.write('Jaccard on train set after seuil for deplace '+deplace_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:]))+'\n') 
			print('Jaccard on train set after seuil for deplace '+deplace_str+' run '+str(run)+' '+str(jaccard(Y_train,img_imp_gt[:n_train,:,:,:])))

			if jaccard_train_after < jaccard_train_before and jaccard_val_after < jaccard_val_before:
				Y_train = Y_train_tmp
				Y_val = Y_val_tmp
				quit_train = 1
			
			'''
			
			# patience -= 2
		
		# serialize weights to HDF5
		model.save_weights(os.path.join('weights','deplace','model_deplace_'+deplace_str+'.h5'))
		print('Saved model deplace '+deplace_str+' to disk')
		'''
		# Training curve
		plt.rcParams['figure.figsize'] = (10.0, 8.0)
		plt.plot(history.epoch, history.history['loss'], label='train')
		plt.plot(history.epoch, history.history['val_loss'], label='val')
		plt.title('Training performance')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend()
		plt.ylim(0.0, 0.9)
		plt.savefig(os.path.join('images','training','training_curve_deplace'+param_str+'.png'))
		plt.cla()	
		plt.clf()
		plt.close()
		'''
 	
		# Distortion between gt and labels 	
		distortion_log.write('Jaccard between ground truth and labels for deplace ' + deplace_str + ' : ' + str(jaccard(img_gt, img_imp_gt)) + '\n')
	distortion_log.close()
	jaccard_log.close()
