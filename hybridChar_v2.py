import h5py
import itertools
import scipy as sci
import numpy as np
import os 
import scipy.signal as sig
import H5Graph
from scipy import interpolate as inter
from pycbc.waveform import td_approximants, get_td_waveform
from pycbc.types.timeseries import TimeSeries
import glob
import time
import multiprocessing as mp
from multiprocessing import Pool
from hybridmodules import *
from functools import partial
import signal
import matplotlib.pyplot as plt
solar_mass_mpc = 2.0896826e19
h_conversion = total_mass/solar_mass_mpc
t_conversion = total_mass*(4.92686088e-6)
q = 5.0
m_1 = 1.4
m_2 = 7.0
total_mass = m_1 + m_2
f_low = 70
distance = 1
sample_rate = 4096*10
delta_t = 1.0/sample_rate
sAx = 0.0
sAy = 0.0
sAz = 0.0
sBx = 0.0
sBy = 0.0
sBz = -0.9
inclination = 0.0
PN_names = ['SEOBNRv2','SEOBNRv3','SEOBNRv4']
PN_models = []
## getPN returns (shift_times,new_hp,new_hc) new meaning the 2-2 spherical harmonic is divided out 
for i,val in enumerate(PN_names): 
	if val == 'NR_hdf5_pycbc':
		pass
	elif val == 'NR_hdf5_pycbc_sxs':
		pass
	elif val == 'PhenSpinTaylor':
                pass
	elif val == 'PhenSpinTaylorRD':
                pass
	elif val == 'SpinDominatedWf':
                pass
	else: 
		try:		
			PN_models.append((i,val,getPN(val,m_1,m_2,f_low,distance,delta_t,sAx,sAy,sAz,sBx,sBy,sBy,inclination)))
		except Exception:
			print 'Outside Model Calculations  for: ' + val
			pass
## each PN_model entry is a tuple. within each tuple we have (index number,val (name of model),(time,PN_wave)) so PN_model is of the form PN_model[i][j]
PN_index = []
for i,val in enumerate(PN_models):
	PN_index.append(PN_models[i][0])
PN_all_waves = []
for i,val in enumerate(PN_models):
	PN_all_waves.append(PN_models[i][2])
PN_all_waves = np.array(PN_all_waves)
### each PN_all_waves entry contains an array containing arrays with (time,PN_wave) with the PN-wave in complex form.
# NOTE: Numerical waves have junk radiation in beginning. This causes problems in the frequency series. Solve this by reading in the numerical metadata
# for the relaxation time and cutting data according to that. The relaxation time is just the time it takes for the junk to go away.
num_levs  = glob.glob('Insert Simulation Directory Here /Lev*/rhOverM_Asymptotic_GeometricUnits.h5')
add_num_levs = [x + '/Extrapolated_N3.dir/Y_l2_m2.dat' for x in num_levs]
index_lev = np.arange(len(num_levs))
all_levs = []
### data has a different lev in its entries.each lev has extrapolatedN4.dir.extrapolatedN4.dir has each harmonic each lev has t,hp,hc.
### empty entry for now. we need to specify a lev
### iterate over number of entries in data.
def formatNum(i):
	#print add_num_levs[i]
	#data_ram = hdf5.File(num_levs[i])
	data_raw = H5Graph.importH5Data([add_num_levs[i]])
	data = H5Graph.formatData(data_raw)
	num_hp = data[0][0][:,1][550:]*h_conversion
	num_hc = data[0][0][:,2][550:]*h_conversion
	num_t = data[0][0][:,0][550:]*t_conversion
	interpo_hp = sci.interpolate.interp1d(num_t,num_hp, kind = 'linear')
	interpo_hc = sci.interpolate.interp1d(num_t,num_hc, kind = 'linear')
##### interpolate the numerical hp and hc with PN timeseries
	num_ts = np.arange(num_t[0],num_t[-1],delta_t)
	#num_t_zeros = np.concatenate((num_ts,np.zeros(np.absolute(len(PN_tc)-len(num_t)))),axis = 0)
	new_num_hp = interpo_hp(num_ts)
	new_num_hc = interpo_hc(num_ts)
#### Cast waves into complex form and take fft of num_wave
	num_wave = (new_num_hp - new_num_hc*1j)
	return('Lev'+ str(i),num_ts,num_wave)

for i in index_lev:
	all_levs.append(formatNum(i))

for i,val in enumerate(all_levs):
        num_ts = all_levs[i][1]
initial = 0
# N_lev_z will contain a tuple (num_z,freq) in each slot each corresponding to a lev given N  
#N_lev_z = []
# N_lev_freq_range will contain a tuple (best_z[0],freq_range) 
#N_lev_freq_range = []
# for the mimic wave use PN models: EOB,Phenom T4,T3,T2. multiple of 1.4 for mass
#for i,val in enumerate(all_levs):
#	num_ts = all_levs[i][1]
#### now we have num_ts = all_levs[i][1], complex numerical waveforms as num_wave = all_levs[i][2]
#### for PN we have PN_models[i][0] = name, PN_tc = PN_models[i][1][0], PN_wave = PN_models[i][1][1] = PN_all_waves
#start_time = time.time()
if __name__ == '__main__':
	p = mp.Pool(mp.cpu_count())	
	for i in itertools.islice(np.arange(0,len(PN_index)),0,len(PN_index)): 
		h1 = PN_all_waves[i][1]
		h1_ts = PN_all_waves[i][0]
		h2_fs = m_frequency_from_polarizations(np.real(h1),np.imag(h1),delta_t)
		for j in itertools.islice(np.arange(0,len(index_lev)),1,len(index_lev)): 
			h2 = all_levs[j][2]
                        h2_ts = all_levs[j][1]
			h2_fs = m_frequency_from_polarizations(np.real(h2),np.imag(h2),delta_t)	
			num_iteration = np.arange(1,len(h2),1)
			#PN_iteration = np.arange(len(h2),len(h1),1)
			func_num_match = partial(match_generator_num, h1, h2, initial)
			#func_PNmatch = partial(match_generator_PN,h1,h2, initial)
			num_z = p.map(func_num_match,num_iteration)
			#PN_z = p.map(func_PNmatch,PN_iteration)
			num_z = np.array(num_z)
			#### for best_z, give a minimum match threshold
			#best_z = np.where(num_z>=0.99)
			#array_z_index = np.argwhere(num_z>=0.99)
			#freq_range = []
			#for n in array_z_index:
        		#	freq_range.append(h2_fs[n][0])
			### native way to get rid of negative frequencies after collison argmax
			#N_lev_z.append((num_z[:np.argmax(h2_fs)],h2_fs[:np.argmax(h2_fs)]))
			#N_lev_freq_range.append((best_z[0],freq_range))
			#### 2-d with best match value and corresponding freq value	
			#best_z_freq = np.column_stack((best_z[0],freq_range))
			path_name_data = 'Data/Hybrids/BBH_SKS_d20_q5_sA_0_0_-0.900_sB_0_0_0/Data/' + PN_models[i][1]  
			if not os.path.exists(os.path.dirname(path_name_data)):
				try:
					os.makedirs(os.path.dirname(path_name_data))
				except OSError as exc:  
					if exc.errno != errno.EEXIST:
						raise
			#np.savetxt(path_name_data + '_N3_Lev'+str(j+1)+'num_wt_d1_flow70_freqrange.txt',np.transpose([best_z[0][:], freq_range[:]]), delimiter = ' ')
			#np.savetxt(path_name_data + '_N3_Lev'+str(j+1)+'_num_wt_d1_flow70.txt',np.transpose([num_z[:], small_h2_ts[:]]), delimiter = ' ')
			#np.savetxt(path_name_data + '_N3_Lev'+str(j+1)+'_num_wf_d1_flow70.txt',np.transpose([num_z[:], small_h2_fs[:]]), delimiter = ' ')
			np.savetxt(path_name_data + '_N3_Lev'+str(j+1)+'num_wt_d1_flow70Full.txt',np.transpose([num_z[:], h2_ts[1:]]), delimiter = ' ')
			#### Note: numerical phase is usually noisey in the beginning. This noise creates a garbage num_z for the first 500 data points or so
			np.savetxt(path_name_data + '_N3_Lev'+str(j+1)+'num_wf_d1_flow70Full.txt',np.transpose([num_z[:np.argmax(h2_fs)], h2_fs[:np.argmax(h2_fs)]]), delimiter = ' ')
			#### Below writes a hybrid wave using the parameters for each ij combination
			'''
			path_name_hybrid = 'Data/Hybrids/BBH_SKS_d20_q5_sA_0_0_-0.900_sB_0_0_0/Hybrids2/'+ PN_models[i][1]  
                        if not os.path.exists(os.path.dirname(path_name_hybrid)):
                                try:
                                        os.makedirs(os.path.dirname(path_name_hybrid))
                                except OSError as exc:  
                                       if exc.errno != errno.EEXIST:
                                                raise
			for k in num_iteration:
				hybrid = hybridize(h1,h2,h1_ts,h2_ts,0,k,M=300)	
				#hybrid_timeseries = (TimeSeries(hybrid[0],delta_t),TimeSeries(hybrid[1],delta_t,),TimeSeries(hybrid[2],delta_t,))
				np.savetxt(path_name_hybrid + '_hybrid_N3_lev'+str(j+1)+'_d1_flow70_i'+str(int(k))+'.txt',np.transpose([hybrid[0],hybrid[1],hybrid[2]]), delimiter = ' ')
			'''
#end_time = (time.time() - start_time
#print 'end_time', end_time
