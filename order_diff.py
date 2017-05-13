import numpy as np
import scipy as sci
import glob
#from  hybridmodules import match
from  pylab import plot,show
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
### import match functions ordered by lev and extrapolation order
N2_dirs = glob.glob('Data/Hybrids/BBH_SKS_d20_q5_sA_0_0_-0.900_sB_0_0_0/SEOBNRv1N2_Lev*num_w_d1_flow50.txt')
N3_dirs = glob.glob('Data/Hybrids/BBH_SKS_d20_q5_sA_0_0_-0.900_sB_0_0_0/SEOBNRv1N3_Lev*num_w_d1_flow50.txt')
N4_dirs = glob.glob('Data/Hybrids/BBH_SKS_d20_q5_sA_0_0_-0.900_sB_0_0_0/SEOBNRv1N4_Lev*num_w_d1_flow50.txt')
time_N2 = []
time_N3 = []
time_N4 = []
N2_y = []
N3_y = []
N4_y = []
for val in N2_dirs:
        data1 = np.loadtxt(val,usecols = [0])
        data2 = np.loadtxt(val,usecols = [1])
        N2_y.append(data1)
        time_N2.append(data2)
for val in N3_dirs:
        data1 = np.loadtxt(val,usecols = [0])
        data2 = np.loadtxt(val,usecols = [1])
        N3_y.append(data1)
	time_N3.append(data2)
for val in N4_dirs:
	data1 = np.loadtxt(val,usecols = [0])
	data2 = np.loadtxt(val,usecols = [1])
	N4_y.append(data1)
	time_N4.append(data2)
N2_y = np.array(N2_y)
N3_y = np.array(N3_y)
N4_y = np.array(N4_y)
time_N2 = np.array(time_N2)
time_N3 = np.array(time_N3)
time_N4 = np.array(time_N4)
times = (time_N2,time_N3,time_N4)
# Fix a lev then calculate the differences in N changes
diff_N2_N3 = []
diff_N2_N4 = []
diff_N3_N4 = []
### Calculates differences between extrapolation order given lev
for i,val in enumerate(N2_y):
	if len(N2_y[i]) < len(N4_y[i]):
		N2_y[i] = np.append(N2_y[i],np.zeros(np.abs(len(N2_y[i])-len(N4_y[i]))))
	
	if len(N2_y[i]) < len(N3_y[i]):
                N2_y[i] = np.append(N2_y[i],np.zeros(np.abs(len(N2_y[i])-len(N3_y[i]))))

	if len(N3_y[i]) < len(N4_y[i]):
                N3_y[i] = np.append(N3_y[i],np.zeros(np.abs(len(N3_y[i])-len(N4_y[i]))))	

	if len(N4_y[i]) < len(N3_y[i]):
                N4_y[i] = np.append(N4_y[i],np.zeros(np.abs(len(N3_y[i])-len(N4_y[i]))))
	diff_N2_N3.append(np.abs(np.subtract(N2_y[i],N3_y[i])))
	diff_N2_N4.append(np.abs(np.subtract(N2_y[i],N4_y[i])))
	diff_N3_N4.append(np.abs(np.subtract(N3_y[i],N4_y[i])))

## Plots all extrapolation orders on single graph given Lev
for i,val in enumerate(N2_y):
	lengths = []
	for j in np.arange(0,3):
		lengths.append(len(times[j][i]))
	large = np.argmax(lengths)
	plt.plot(times[large][i][:-1-np.abs(len(time_N2[i])-len(time_N3[i]))],N2_y[i][:-1-np.abs(len(time_N2[i])-len(time_N3[i]))],label='N2')
        plt.plot(times[large][i][:-1-np.abs(len(time_N3[i])-len(time_N4[i]))],N3_y[i][:-1-np.abs(len(time_N3[i])-len(time_N4[i]))],label='N3')
        plt.plot(times[large][i][:-1-np.abs(len(time_N4[i])-len(time_N2[i]))],N4_y[i][:-1-np.abs(len(time_N4[i])-len(time_N2[i]))],label='N4')	
	#plt.plot(times[large][i],N2_y[i],label='N2')
        #plt.plot(times[large][i],N3_y[i],label='N3')
        #plt.plot(times[large][i],N4_y[i],label='N4')
        plt.legend(loc='upper right')
        plt.title('Mimic Match Lev'+ str(2+i))
        plt.xlabel('Max time slice (seconds)')
        plt.ylabel('Match A(z(t))')
	#plt.show()
	plt.savefig('/home/eflynn/order_matches/BBH_SKS_d20_q5_sA_0_0_-0.900_sB_0_0_0/SEOBNRv1_order_Lev' + str(2+i) +'.png')
	plt.clf()

### Plot differences against each other 
for i,val in enumerate(N2_y):
        lengths = []
        for j in np.arange(0,3):
                lengths.append(len(times[j][i]))
        large = np.argmax(lengths)
	plt.plot(times[large][i][:-1-np.abs(len(time_N2[i])-len(time_N3[i]))],diff_N2_N3[i][:-1-np.abs(len(time_N2[i])-len(time_N3[i]))],label='N2-N3')
	plt.plot(times[large][i][:-1-np.abs(len(time_N3[i])-len(time_N4[i]))],diff_N3_N4[i][:-1-np.abs(len(time_N3[i])-len(time_N4[i]))],label='N3-N4')
	plt.plot(times[large][i][:-1-np.abs(len(time_N4[i])-len(time_N2[i]))],diff_N2_N4[i][:-1-np.abs(len(time_N4[i])-len(time_N2[i]))],label='N4-N2')
	#plt.plot(times[large][i],N2_y[i],label='N2')
        #plt.plot(times[large][i],N3_y[i],label='N3')
        #plt.plot(times[large][i],N4_y[i],label='N4')
	plt.legend(loc='upper left')
        plt.title('Mimic Lev'+ str(2+i) + 'Extrapolation Difference')
        plt.xlabel('Max time slice (seconds)')
        plt.ylabel('Difference in match')
	#plt.show()
	plt.savefig('/home/eflynn/order_matches/BBH_SKS_d20_q5_sA_0_0_-0.900_sB_0_0_0/SEOBNRv1_diff_order_Lev' + str(2+i) + '.png')
        plt.clf()

