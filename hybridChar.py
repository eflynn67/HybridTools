import h5py
import itertools
import numpy as np
import os 
import lal
import re
import scipy.signal as sig
import hashlib
import romspline
from scipy import interpolate as inter
import scipy as sci
import pycbc
from pycbc.waveform import td_approximants, get_td_waveform
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import utils as wfutils
from pycbc import pnutils
import glob
import multiprocessing as mp
from multiprocessing import Pool
import hybridmodules as hy 
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

####set numerical parameters, read, and convert data to SI units
solar_mass_mpc = lal.MRSUN_SI / (1e6*lal.PC_SI)
q = 1
m_1 = 1.4 ### mass 1
m_2 = q*m_1 ### mass 2 
total_mass = m_1 + m_2
sample_rate = 4096.0*10
delta_t = 1.0/sample_rate
PN_names = ['TaylorT4']
num_levs  = glob.glob('../BH_NS/Data/SimulationAnnex/q1SpinZero/BBH_SKS_d23.1_q1_sA_0_0_0_sB_0_0_0/Lev*/rhOverM_Asymptotic_GeometricUnits.h5')
for i in num_levs:
    print i
sim_name = re.search('/BBH(.+?)/L(.+?)/',num_levs[0]).group(0)
print sim_name
num_waves = []
for name in num_levs:
	sxs_wave = hy.getFormatSXSData(name,total_mass,delta_t=delta_t)
	num_waves.append(sxs_wave)
#### numerical parameters for metadata (fake for now)
LNhatx = 0.0
LNhaty = 1.0
LNhatz = 0.0
n_hatx = 0.0
n_haty = 0.0
n_hatz = 1.0 
coa_phase = 2.0 
#### Now set PyCBC parameters
s1x = 0.0 
s1y = 0.0
s1z = 0.0
s2x = 0.0
s2y = 0.0
s2z = 0.0
inclination = 0.0 
tidal_initial = 0.0
tidal_final = 0.0
delta_tidal = 1.0
tidal_range = np.arange(tidal_initial,tidal_final+1,delta_tidal) 
ip1 = 0
ip2 = 0
type = 'BBH'
sim_hybrid_name = 'BBH_q%d_sA%d_sB%d'%(q,s1z,s2z)
PN_models = []
shift_times = []
hh_freqs = []
f_low = 70 ### lower frequency bound for PyCBC
distance = 1.0 ### in Mpcs
#### Hybrid Params
match_lower = 0.9999955
start_lev = 5
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
                    for j in itertools.islice(np.arange(0,len(num_waves)),start_lev,len(num_waves)):                         
                        best_numz_indices = []
                        best_PNz_indices = []
                        sim_name = num_waves[j][0]
                        path_name_data = 'HybridAnnex/'+type+sim_name+approx+'/'
                        h2 = num_waves[j][2]
                        h2_ts = num_waves[j][1]
                        h2_fs = hy.m_frequency_from_polarizations(np.real(h2),np.imag(h2),delta_t)	
                        num_iteration = np.arange(1,len(h2)+1,1,dtype=np.int64)
                        PN_iteration = np.arange(len(h1)-len(h2)+1,len(h1)+1,1,dtype=np.int64)
                        func_match_h2 = partial(hy.match_generator_num, h1, h2, ip2)
                        func_match_h1 = partial(hy.match_generator_PN, h1, h2, ip1)
                        #match_fp1 = np.zeros(len(PN_iteration)+1)
                        match_fp1 = p.map(func_match_h1,PN_iteration)
                        match_fp2 = p.map(func_match_h2,num_iteration)       
                        match_fp2 = np.array(match_fp2)
                        match_fp1 = np.array(match_fp1)
                        print 'finished match'
                        #### for best_z, give a minimum match threshold. Uses h1 freq and time data.
                        best_numz = np.where(match_fp2>=match_lower)
                        best_numz_indices_temp = np.argwhere(match_fp2>=match_lower)
                        for k in best_numz_indices_temp:
                            best_numz_indices.append(k[0])
                        best_PNz = np.where(match_fp1>=match_lower)
                        best_PNz_indices_temp = np.argwhere(match_fp1>=match_lower)
                        for k in best_PNz_indices_temp:
                            best_PNz_indices.append(k[0])
                        print best_numz_indices
                        freq_range_num = []
                        time_range_num = []
                        freq_range_PN = []
                        time_range_PN= []
                        for n in best_numz_indices:
                            time_range_num.append(h1_ts[n])
                            freq_range_num.append(h1_fs[n])
                        for n in best_PNz_indices:
                            time_range_PN.append(h1_ts[n])
                            freq_range_PN.append(h1_fs[n-1])
                        if not os.path.exists(os.path.dirname(path_name_data)):
                            try:
                                os.makedirs(os.path.dirname(path_name_data))
                            except OSError as exc:  
                                if exc.errno != errno.EEXIST:
                                    raise 
                        print 'writing char data to h5'
                        with h5py.File(path_name_data+'HybridChars.h5','w') as fd:
                            fd.attrs.create('Read_Group', 'Flynn')
                            fd.attrs.create('type', 'Hybrid:%s'%type)
                            fd.attrs.create('approx', approx)
                            fd.attrs.create('sim_name', sim_name)
                            fd.create_dataset("match_PN_fp", data=match_fp1)
                            fd.create_dataset("match_num_fp", data=match_fp2)
                            fd.create_dataset('time_slices',data=h2_ts)
                            fd.create_dataset('freq_slices',data=h2_fs)
                            fd.create_dataset('tidal_rangeA',data=tidal_range)
                            fd.create_dataset('tidal_rangeB',data=tidal_range)
                            fd.create_dataset('bestFreqRange_sim',data=freq_range_num)
                            fd.create_dataset('bestTimeRange_sim',data=time_range_num)
                            fd.create_dataset('bestFreqRange_approx',data=freq_range_PN)
                            fd.create_dataset('bestTimeRange_approx',data=time_range_PN)
                            fd.close()
                        print 'writing hybrids'
                        for l in best_numz_indices[2:]:
                            if len(best_numz_indices[2:])> 200:
                                break
                            else:
                                hybridPN_Num = hy.hybridize(h1,h1_ts,h2,h2_ts,match_i=0,match_f=l,delta_t=delta_t,M=300,info=1)
                                shift_times.append(hybridPN_Num[6])
                                hh_freqs.append(h1_fs[hybridPN_Num[1]])
                                f_low_M = f_low * (lal.TWOPI * total_mass * lal.MTSUN_SI)
                                with h5py.File(path_name_data+sim_hybrid_name+'_'+approx+'fp2_'+str(l)+'.h5','w') as fd:
                                    mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(m_1, m_2)
                                    hashtag = hashlib.md5()
                                    fd.attrs.create('type', 'Hybrid:%s'%type)
                                    hashtag.update(fd.attrs['type'])
                                    fd.attrs.create('hashtag', hashtag.digest())
                                    fd.attrs.create('Read_Group', 'Flynn')
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
                                    fd.attrs.create('PN_fp2', len(h1))
                                    fd.attrs.create('Num_begin_window_index',0 )
                                    #fd.attrs.create('check_match',hybridPN_Num[0])
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
                            with h5py.File(path_name_data+'HybridChars.h5','a') as fd:
                                fd.create_dataset('shift_times',data=shift_times)
                                fd.create_dataset('hybridize_freq',data=hh_freqs)
                                fd.close()
    '''                        
                            print 'saving plots'
                            with PdfPages(path_name_data+'match_v_maxSimTime.pdf') as pdf:
                                plt.plot(h2[0], match_fp2)
                                plt.legend(loc='best')
                                plt.title('Match vs. Max Simulation Time Window')
                                plt.xlabel('Time Slice (seconds)')
                                plt.ylabel('Match')
                                pdf.savefig()
                                plt.clf()
                            with PdfPages(path_name_data+'match_v_freq_range.pdf') as pdf:
                                plt.plot(h2_fs, match_fp2)
                                plt.legend(loc='best')
                                plt.title('Match vs. Mac Frequency Window')
                                plt.xlabel('Frequency slice (Hz)')
                                plt.ylabel('Match')
                                pdf.savefig()
                                plt.clf()
                            with PdfPages(path_name_data+'match_v_shift_time.pdf') as pdf
                                plt.plot(h2_fs, match_fp2)
                                plt.legend(loc='best')
                                plt.title('Match vs. Mac Frequency Window')
                                plt.xlabel('Frequency slice (Hz)')
                                plt.ylabel('Match')
                                pdf.savefig()
                                plt.clf()
    '''          
