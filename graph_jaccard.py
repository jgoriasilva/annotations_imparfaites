import numpy as np
import matplotlib.pyplot as plt
import os

jaccard_log = open(os.path.join('logs','jaccard','jaccard_oubli.log'),'r')
jaccard_data = []
total_runs = int(input('how many sequential runs? '))
x = range(total_runs)
y_before = []
y_after = []

for line in jaccard_log:
	line = line.split(' ')
	if line[2] == 'test':
		continue
	elif line[4] == 'before':
		y_before.append(float(line[-1]))	
	elif line[4] == 'after':
		y_after.append(float(line[-1]))	

for oubli in range(0,100,5):
	plt.figure(oubli)
	plt.title('Evolution of jaccard on train set for oubli {}'.format(oubli))
	plt.scatter(x,y_before[int(oubli/5*total_runs):int((oubli/5+1)*total_runs)],label='before')
	plt.scatter(x,y_after[int(oubli/5*total_runs):int((oubli/5+1)*total_runs)],label='after')
	plt.xlabel('run number')
	plt.ylabel('jaccard between y_train and ground truth label')
	plt.xticks(range(total_runs))
	plt.legend()
	# plt.grid()
	plt.savefig(os.path.join('images','jaccard','before_after_{}.png'.format(oubli)))
	plt.cla()
	plt.close()


	
		
