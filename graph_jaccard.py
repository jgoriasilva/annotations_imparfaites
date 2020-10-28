import numpy as np
import matplotlib.pyplot as plt
import os

train_type = 'oubli'
distortion_log = open(os.path.join('logs','distortion','distortion_'+train_type+'.log'),'r')
jaccard_log = open(os.path.join('logs','jaccard','jaccard_'+train_type+'.log'),'r')
distortion_data = []
jaccard_data_before = []
jaccard_data_after = []

for line in distortion_log:
    distortion_data.append(float(line.split(' ')[-1]))

if train_type == 'oubli':
    for line in jaccard_log:
        line = line.split(' ')
        if(line[4] == 'before'):
            jaccard_data_before.append([float(line[-4])/5, float(line[-1])])
        if(line[4] == 'after'):
            jaccard_data_after.append([float(line[-4])/5, float(line[-1])])
    data_before = np.array(jaccard_data_before)
    data_after = np.array(jaccard_data_after)

    data_0 = []
    data_1 = []

    for line in data_before:
        line[0] = distortion_data[int(line[0])]
        data_0.append([line[0],line[1]])
    for line in data_after:
        line[0] = distortion_data[int(line[0])]
        data_1.append([line[0],line[1]])

    data_0 = np.array(data_0)
    data_1 = np.array(data_1)

plt.figure()
if train_type == 'oubli':
    # plt.plot([0,1],[0,1],'--', label='y = x')
    plt.plot(data_0[:,0],data_0[:,1],'o',label='before threshold')
    plt.plot(data_1[:,0],data_1[:,1],'o',label='after threshold')
    # plt.plot(data_2[:,0],data_2[:,1],'o',label='run 2')
    # plt.plot(data_3[:,0],data_3[:,1],'o',label='run 3')
plt.xlabel('Jaccard between ground truth labels and labels initially given to the network')
plt.ylabel('Jaccard between ground truth labels and labels given to the network')
plt.title('Impacts of thresholding over the quality of the labels')
plt.xlim(0)
plt.ylim(0)
plt.grid()
plt.legend()
plt.savefig(os.path.join('images','graph_before_after_'+train_type+'.png'))
plt.show()
                            
