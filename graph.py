import numpy as np
import matplotlib.pyplot as plt
import os

distortion_log = open(os.path.join('logs','distortion_oubli.log'),'r')
jaccard_log = open(os.path.join('logs','jaccard.log'),'r')
distortion_data = []
jaccard_data = []
total_run = int(input('How many sequential runs? '))

for line in distortion_log:
	distortion_data.append(float(line.split(' ')[-1]))
x = [item for item in distortion_data for _ in range(total_run)]
x = np.array(x)

for line in jaccard_log:
	jaccard_data.append(float(line.split(' ')[-1]))
y = np.array(jaccard_data)

# Scatter plot
plt.figure()
for run in range(total_run):
	plt.plot(x[run:len(x):total_run], y[run:len(y):total_run], 'o', label='network outputs')
'''
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

plt.xlabel('Jaccard between ground truth labels and labels given to the network')
plt.ylabel('Jaccard between ground truth labels and outputs of the network')
plt.title('Relation between imperfect labeling and network performance')
plt.plot([0,1],[0,1],label='y = x')
plt.legend()
plt.grid()
plt.savefig(os.path.join('images','graph_oubli.png'))
#plt.show()
