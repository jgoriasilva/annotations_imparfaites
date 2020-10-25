import numpy as np
import matplotlib.pyplot as plt
import os

train_type = input('train type? ')
distortion_log = open(os.path.join('logs','distortion','distortion_'+train_type+'.log'),'r')
jaccard_log = open(os.path.join('logs','jaccard','jaccard_'+train_type+'.log'),'r')
distortion_data = []
jaccard_data = []

for line in distortion_log:
	distortion_data.append(float(line.split(' ')[-1]))

'''
if train_type == 'oubli':
	x = [item for item in distortion_data for _ in range(total_run)]
	x = np.array(x)
else:
	x = np.array(distortion_data)
'''

if train_type == 'oubli':
	for line in jaccard_log:
		line = line.split(' ')
		if(line[2] == 'test'):
			jaccard_data.append([float(line[-4])/5, int(line[-2]), float(line[-1])])
	data = np.array(jaccard_data)

	data_0 = []
	data_1 = []
	data_2 = []
	data_3 = []

	for line in data:
		line[0] = distortion_data[int(line[0])]
		if line[1] == 0:
			data_0.append([line[0],line[2]])
		if line[1] == 1:
			data_1.append([line[0],line[2]])
		if line[1] == 2:
			data_2.append([line[0],line[2]])
		if line[1] == 3:
			data_3.append([line[0],line[2]])

	data_0 = np.array(data_0)
	data_1 = np.array(data_1)
	data_2 = np.array(data_2)
	data_3 = np.array(data_3)

elif train_type == 'taille':
	for line in jaccard_log:
		jaccard_data.append(float(line.split(' ')[-1]))
	data = np.array([distortion_data,jaccard_data])

plt.figure()
if train_type == 'oubli':
	plt.plot(data_0[:,0],data_0[:,1],'o',label='run 0')
	plt.plot(data_1[:,0],data_1[:,1],'o',label='run 1')
	plt.plot(data_2[:,0],data_2[:,1],'o',label='run 2')
	plt.plot(data_3[:,0],data_3[:,1],'o',label='run 3')
	plt.legend()
elif train_type == 'taille':
	plt.plot(data[0,:],data[1,:],'o',label='network outputs')
plt.xlabel('Jaccard between ground truth labels and labels given to the network')
plt.ylabel('Jaccard between ground truth labels and outputs of the network')
plt.title('Relation between imperfect labeling and network performance')
plt.plot([0,0.6],[0,0.6], label='y = x')
plt.xlim(0)
plt.ylim(0)
plt.grid()
plt.savefig(os.path.join('images','graph_'+train_type+'.png'))
plt.show()


'''
# Scatter plot
plt.figure()
if train_type == 'oubli':
	for run in range(0,total_run,3):
		plt.plot(x[run:len(x):total_run], y[run:len(y):total_run], 'o', label=str(run))
else:
	plt.plot(x, y, 'o')

# Curve fitting
deg = int(input('Degree of polynomial fit: '))
coef = np.polyfit(x, y, deg)
poly = np.poly1d(coef)

# r2 score
yhat = poly(x)
ybar = sum(y)/len(y)
SST = sum((y - ybar)**2)
SSreg = sum((yhat - ybar)**2)
r2 = SSreg/SST

if deg == 1:
	plt.plot(x, poly(x), label='y={:.2f}x+{:.2f}'.format(coef[0], coef[1]) + '; r2 = {:.2f}'.format(r2))
elif deg == 2:
	plt.plot(x, poly(x), label='y={:.2f}x^2+{:.2f}x+{:.2f}'.format(coef[0], coef[1], coef[2]) + '; r2 = {:.2f}'.format(r2))

'''


