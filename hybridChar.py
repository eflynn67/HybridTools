import h5py
import itertools
import numpy as np
import os 
import lal
import re
import scipy.signal as sig
from scipy import interpolate as inter
import scipy as sci
from pycbc.waveform import td_approximants, get_td_waveform
from pycbc.types.timeseries import TimeSeries
import glob
import multiprocessing as mp
from multiprocessing import Pool
import hybridmodules as hy 
from functools import partial
import matplotlib.pyplot as plt

####set numerical parameters, read, and convert data to SI units
solar_mass_mpc = lal.MRSUN_SI / (1e6*lal.PC_SI)
q = 1
m_1 = 1.4 ### mass 1
m_2 = q*m_1 ### mass 2 
total_mass = m_1 + m_2
sample_rate = 4096.0*10
delta_t = 1.0/sample_rate
PN_names = ['SEOBNRv2','SEOBNRv3','SEOBNRv4']
num_levs  = glob.glob('../BH_NS/Data/SimulationAnnex/q1SpinZero/BBH_SKS_d23.1_q1_sA_0_0_0_sB_0_0_0/Lev*/rhOverM_Asymptotic_GeometricUnits.h5')
print num_levs[0]
sim_name = re.search('/BBH(.+?)/L(.+?)/',num_levs[0]).group(0)
print sim_name
num_waves = []
for name in num_levs:
	sxs_wave = hy.getFormatSXSData(name,total_mass,delta_t=delta_t)
	num_waves.append(sxs_wave)
#### Now set PyCBC parameters
f_low = 70 ### lower frequency bound for PyCBC
s1x = 0.0 
s1y = 0.0
s1z = 0.0
s2x = 0.0
s2y = 0.0
s2z = 0.0
inclination = 0.0 
tidal_initial = 2700.0
tidal_final = 2700.0
delta_tidal = 1.0
tidal_range = np.arange(tidal_initial,tidal_final+1,delta_tidal) 
initial = 0.0
type = 'BBH'
sim_hybrid_name = 'BBH_q%d_sA%d_sB%d'%(s1z,s2z)
PN_models = []
shift_times = []
#### Hybrid Params
match_lower = 0.99

#### mostly concerned with ligo noise curve comparison. Keep threshold and write to standard h5 format with. h1-h2 L2 norm is good to do. Look at Lindbolm use and abuse of hybrid
#### look at Data formats for numerical relativity 2011 for hybrid metadata.
#### focus on format = 1. problem with ref_f in format=2.
#### 
if __name__ == '__main__':
    p = mp.Pool(mp.cpu_count())
    for approx in PN_names:	#### pick approx 	
        for tidal_1 in tidal_range:		#### pick the range of the tidal_1 parameter (this should work for an array with 1 element?)
            for tidal_2 in tidal_range: 		#### pick range for tidal_2
                PN_model = hy.getPN(approx,m1=m_1,m2=m_2,f_low=f_low,distance=distance,delta_t=delta_t,
                                sAx=s1x,sAy=s1y,sAz=s1z,sBx=s2x,sBy=s2y,sBz=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                for i in itertools.islice(np.arange(0,len(PN_names)),0,len(PN_names)): 
                    h1 = PN_model[1]
                    h1_ts = PN_model[0]
                    h1_fs = hy.m_frequency_from_polarizations(np.real(h1),np.imag(h1),delta_t)
                    for j in itertools.islice(np.arange(0,len(num_waves)),0,len(num_waves)):                         
                        sim_name = num_waves[j][0]
                        path_name_data = 'HybridAnnex/'+sim_name+approx+'/'
                        h2 = num_waves[j][2]
                        h2_ts = num_waves[j][1]
                        h2_fs = hy.m_frequency_from_polarizations(np.real(h2),np.imag(h2),delta_t)	
                        num_iteration = np.arange(1,len(h2),1)
                        PN_iteration = np.arange(len(h2),len(h1),1)
                        func_num_match = partial(hy.match_generator_num, h1, h2, initial)
                        func_PN_match = partial(hy.match_generator_num,h1,h2, initial)
                        match_fp2 = p.map(func_num_match,num_iteration)
                        match_fp1 = p.map(func_PN_match,PN_iteration)
                        num_z = np.array(num_z)
                        PN_z = np.array(PN_z)
                        #### for best_z, give a minimum match threshold. Uses h1 freq and time data.
                        best_numz = np.where(num_z>=match_lower)
                        best_numz_indices = np.argwhere(num_z>=match_lower)
                        best_PNz = np.where(PN_z>=match_lower)
                        best_PNz_indices = np.argwhere(PN_z>=match_lower)
                        freq_range_num = []
                        time_range_num = []
                        freq_range_PN = []
                        time_range_PN= []
                        for n in best_numz_indices:
                            time_range_num.append(h1_ts[n])
                            freq_range_num.append(h1_fs[n])
                        for n in best_PNz_indices:
                            time_range_PN.append(h1_ts[n])
                            freq_range_PN.append(h1_fs[n])
                        if not os.path.exists(os.path.dirname(path_name_data)):
                            try:
                                os.makedirs(os.path.dirname(path_name_data))
                            except OSError as exc:  
                                if exc.errno != errno.EEXISTi
                                raise 
                        with h5py.File(path_name_data+'HybridChars.h5','w') as fd:
                            fp.attrs.create('Read_Group', 'Flynn')
                            fp.attrs.create('type', 'Hybrid:%s'%type)
                            fp.attrs.create('approx', approx)
                            fp.attrs.create('sim_name', sim_name)
                            fp.create_dataset("match_PN_fp", data=PN_fp1)
                            fp.create_dataset("match_num_fp", data=num_fp2)
                            fp.create_dataset('time_slices',data=h2[0])
                            fp.create_dataset('freq_slices',data=h2_fs)
                            fp.create_dataset('tidal_rangeA',data=tidal_range)
                            fp.create_dataset('tidal_rangeB',data=tidal_range)
                            fp.create_dataset('bestFreqRange_sim',data=freq_range_num)
                            fp.create_dataset('bestTimeRange_sim',data=time_range_num)
                            fp.create_dataset('bestFreqRange_approx',data=freq_range_PN)
                            fp.create_dataset('bestTimeRange_approx',data=time_range_PN)
                            fp.close()
                        ##### Depending on what question is asked, a marginalized hybrid can be produced. In other words, We can currently only marginalize 
                        ##### question: when constructing the hybrid, do we need to marginalize  over all 4 window parameters? or can we marginalize over one and call it a 
                        ##### "good hybrid". This depends on the situation for example for an NS wave all 4 make a contribution where as a BBH only 1 or 2 contribute.
                        ##### choose to marginalize over numerical window f_p2
                        for l in best_numz_indices:    
                            hybridPN_Num = hy.hybridize(h1[1],h1[0],h2[1],h2[0],match_i=0,match_f=l,delta_t,M=300)
                            shift_times.append(hybridPN_Num[0])
                            h5file= path_name_data+name+'_'+sim_name+'.h5'
                            f_low_M = f_low * (lal.TWOPI * total_mass * lal.MTSUN_SI)
                            with h5py.File(path_name_data+sim_hybrid_name+approx,'w') as fd:
                                mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(mass1, mass2)
                                hashtag = hashlib.md5()
                                hashtag.update(fd.attrs['name'])
                                fd.attrs.create('hashtag', hashtag.digest())
                                fd.attrs.create('Read_Group', 'Flynn')
                                fd.attrs.create('type', 'Hybrid:%s'%type)
                                fd.attrs.create('approx', approx)
                                fd.attrs.create('sim_name', sim_name)
                                fd.attrs.create('f_lower_at_1MSUN', f_low_M)
                                fd.attrs.create('eta', eta)
                                fd.attrs.create('spin1x', s1x)
                                fd.attrs.create('spin1y', s1y)
                                fd.attrs.create('spin1z', s1z)
                                fd.attrs.create('spin2x', s2x)
                                fd.attrs.create('spin2y', s2y)
                                fd.attrs.create('spin2z', s2z)
                                fd.attrs.create('LNhatx', LNhatx)
                                fd.attrs.create('LNhaty', LNhaty)
                                fd.attrs.create('LNhatz', LNhatz)
                                fd.attrs.create('nhatx', n_hatx)
                                fd.attrs.create('nhaty', n_haty)
                                fd.attrs.create('nhatz', n_hatz)
                                fd.attrs.create('coa_phase', coa_phase)
                                fd.attrs.create('mass1', m_1)
                                fd.attrs.create('mass2', m_2)
                                fd.attrs.create('lambda1',tidal_1)
                                fd.attrs.create('lambda2',tidal_2)
                                fd.attrs.create('PN_f', len(h1))
                                fd.attrs.create('Num_begin_window_index',0 )
                                fd.attrs.create('check_match',hybridPN_Num[0])
                                gramp = fd.create_group('amp_l2_m2')
                                grphase = fd.create_group('phase_l2_m2')
                                times = hybridPN_Num[5][0]
                                hplus = hybridPN_Num[5][1]
                                hcross = hybridPN_Num[5][2]
                                massMpc = total_mass*solar_mass_mpc
                                hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
                                hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
                                times_M = times / (lal.MTSUN_SI * total_mass)
                                HlmAmp = wfutils.amplitude_from_polarizations(hplusMpc,
                                        hcrossMpc).data
                                HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data 
                                sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
                                sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
                                sAmph.write(gramp)
                                sPhaseh.write(grphase)
                                fd.close()
                        with h5py.File(path_name_data+'HybridChars.h5','w') as fd:
                            fd.create_dataset('shift_times',data=shift_times)
                            fd.close()             
