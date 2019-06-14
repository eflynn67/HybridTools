import time
import cProfile
import os
import sys
import fnmatch
import glob
import hybridmodules as hy
from functools import partial
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy as sci
import lal
import lalsimulation as lalsim
from pycbc.filter import matchedfilter,sigmasq,make_frequency_series
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries,FrequencySeries
import multiprocessing as mp
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform
#######
### PLEASE NOTE I AM MULTIPLYING BY THE PYCBC SCALE FACTOR 2/2Y2_2 TO UNDO PYCBC'S SCALING FACTOR
#######

#Define CONSTANTS

Sph_Harm = 0.6307831305 # 2-2 spherical harmoniic at theta = 0 phi = 90 
pycbc_factor = 2.0/Sph_Harm
sample_rate = 4096.0*8.0
delta_t = 1.0/sample_rate

##################################
###### INPUT PARAMETERS
##################################
# All this program requires is the paths to the set of waveforms you want to compare.
# NOTE: EOS1 and EOS2 are just labels. These can have the same EOS's and different approximates, different EOS's and same approximates and so on.
hm_paths_EOS1 = glob.glob('HybridAnnex/BNS/CoRe_BAM_0004_R0*/ALF2_1.351_1.351_0.00_0.00_0.052_0.202_camr/TaylorT2/TaylorT2ip2_0_730.0804_730.0804/CoRe_BAM_0004_R0*_ALF2_TaylorT2fp2_*.h5')
#glob.glob('HybridAnnex/BNS/CoRe_BAM_0120_R*/SLy_1.375_1.375_0.00_0.00_0.0361_0.116/TEOBResum_ROM/TEOBResum_ROMip2_0_346.063_346.063/CoRe_BAM_0120_R*_SLy_TEOBResum_ROMfp2_*.h5')
hm_paths_EOS2 = glob.glob('HybridAnnex/BNS/CoRe_BAM_0004_R0*/ALF2_1.351_1.351_0.00_0.00_0.052_0.202_camr/TaylorT4/TaylorT4ip2_0_730.0804_730.0804/CoRe_BAM_0004_R0*_ALF2_TaylorT4fp2_*.h5')

#glob.glob('HybridAnnex/BNS/CoRe_BAM_0005_R*/ALF2_1.375_1.375_0.00_0.00_0.0360_0.167/TaylorT2/TaylorT2ip2_0_658.1463_658.1463/CoRe_BAM_0005_R*_ALF2_TaylorT2fp2_*.h5')
distance = 100.0 #in Mpc
inclination = 0.0
# set upper and lower frequency cutoffs for pycbc match filter 
f_low = 30 #### Must be an integer or else infinities pop up in the match and sigmasq functions
f_high = 10000 #### Must be an integer or else infinities pop up in the match and sigmasq functions
# set PSD type. For aLIGO design, set psd_type = 'aLIGOZeroDetHighPower'and set psd_path='None'. For other PSD's, set psd_type= 'other' and set psd_path = 'path_to_psd' 
# NOTE: I did not check to see if custom psds work with this. I pretty much just ran design sensitivity.
psd_type = 'aLIGOZeroDetHighPower' 
psd_path = 'None'


def _poolinit():
    global prof
    prof = cProfile.Profile()
    def finish():
        prof.dump_stats('./Profiles/profile-%s.out' % mp.current_process().pid)
    mp.util.Finalize(None,finish,exitpriority = 1)

def compare(psd,psd_path,hep,hec,distance,f_low,f_high,approx_e,EOS_e,hm_path): 
    if psd == 'aLIGOZeroDetHighPower':
        inclination = 0.0
        hhm = h5py.File(hm_path, 'r')
        EOS_m = hhm.attrs['EOS']
        grav_m1 = hhm.attrs['grav_mass1']
        grav_m2 = hhm.attrs['grav_mass2']
        s1x = hhm.attrs['spin1x']
        s1y = hhm.attrs['spin1y']
        s1z = hhm.attrs['spin1z']
        s2x = hhm.attrs['spin2x']
        s2y = hhm.attrs['spin2y']
        s2z = hhm.attrs['spin2z']
        f_lower_hhm = hhm.attrs['f_lower_at_1MSUN']#/(2.0 * lal.TWOPI * lal.MTSUN_SI*(m1+m2))
        hhm_freq = hhm.attrs['hybridize_freq']
        hhm_shift_time = hhm.attrs['shift_time']
        approx_m = hhm.attrs['approx'] 
        h_matchm = hhm.attrs['hybrid_match']
        sim_name = hhm.attrs['sim_name']
        hhm.close() 
        hmp,hmc = get_td_waveform(approximant='NR_hdf5',numrel_data=hm_path,mass1=grav_m1,
                                                                  mass2=grav_m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_hhm,
                                                                  inclination=inclination)
        ### multiply by stupid pycbc factor
        hmp = hmp*pycbc_factor
        hmc = hmc*pycbc_factor
        ### set a constant t_len. We do this because all the numerical waveforms will be padded to this length and a new fft plan will not be needed. Basically this makes everything
        ### MUCH faster.
        t_len = 300000 
        hep.resize(t_len)
        hmp.resize(t_len)
        #Generate the aLIGO ZDHP PSD 
        delta_f = 1.0 /hmp.duration
        f_len = t_len/2 + 1
        psd = aLIGOZeroDetHighPower(f_len, delta_f, f_low)
        #calculate L2 distance
        l2_distance = hy.delta_h(hep,hmp)
        #calculate snr and match values
        hep_hmp_match,index = matchedfilter.match(hep,hmp,psd=psd,low_frequency_cutoff=f_low,high_frequency_cutoff=f_high) 
        hmp_norm = matchedfilter.sigmasq(hmp,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high) 
        overlap = matchedfilter.overlap(hep,hmp,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
        mismatch = 1.0 - overlap
        data = (l2_distance,hep_hmp_match,overlap,mismatch,hmp_norm,h_matchm,hhm_shift_time,hhe_freq,hhm_freq,approx_e,approx_m,EOS_e,EOS_m,sim_name)
        return data
    elif psd != 'aLIGOZeroDetHighPower':
        # similar to previous section
        inclination = 0.0
        hhm = h5py.File(hm_path, 'r')
        EOS_m = hhm.attrs['EOS']
        grav_m1 = hhm.attrs['grav_mass1']
        grav_m2 = hhm.attrs['grav_mass2']
        s1x = hhm.attrs['spin1x']
        s1y = hhm.attrs['spin1y']
        s1z = hhm.attrs['spin1z']
        s2x = hhm.attrs['spin2x']
        s2y = hhm.attrs['spin2y']
        s2z = hhm.attrs['spin2z']
        f_lower_hhm = hhm.attrs['f_lower_at_1MSUN']#/(2.0 * lal.TWOPI * lal.MTSUN_SI*(m1+m2))
        hhm_freq = hhm.attrs['hybridize_freq']
        hhm_shift_time = hhm.attrs['shift_time']
        approx_m = hhm.attrs['approx'] 
        h_matchm = hhm.attrs['hybrid_match']
        sim_name = hhm.attrs['sim_name']
        hhm.close() 
        hmp,hmc = get_td_waveform(approximant='NR_hdf5',numrel_data=hm_path,mass1=grav_m1,
                                                                  mass2=grav_m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_hhm,
                                                                  inclination=inclination)
        ### multiply by stupid pycbc factor
        hmp = hmp*pycbc_factor
        hmc = hmc*pycbc_factor
        t_len = 300000 
        hep.resize(t_len)
        hmp.resize(t_len)
        #Generate the aLIGO ZDHP PSD 
        delta_f = 1.0 /hmp.duration
        f_len = t_len/2 + 1
        ### reads in a custom psd file
        psd = pycbc.psd.read.from_txt(psd_path, f_len, delta_f, f_cutoff, is_asd_file=True) 
        l2_distance = hy.delta_h(hep,hmp)
        hep_hmp_match,index = matchedfilter.match(hep,hmp,psd=psd,low_frequency_cutoff=f_low,high_frequency_cutoff=f_high) 
        hmp_norm = matchedfilter.sigmasq(hmp,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high) 
        overlap = matchedfilter.overlap(hep,hmp,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
        mismatch = 1.0 - overlap
        data = (l2_distance,hep_hmp_match,overlap,mismatch,hmp_norm,h_matchm,hhm_shift_time,hhe_freq,hhm_freq,approx_e,approx_m,EOS_e,EOS_m,sim_name)
        return data
    ##       0          1              2         3       4        5        6              7        8           9      10    11    12
d_EOS1 = {}
d_EOS2 = {}
# organizes the hybrids made by resolution. assumes that the pathes include a _R0* string the path.
for path in hm_paths_EOS1:
    index = path.find('_R')
    res = path[index+1:index+4]
    try:
        d_EO1[res].append(path)
    except:
        d_EOS1.setdefault(res,[])
        d_EOS1[res].append(path)
for path in hm_paths_EOS2:
    index = path.find('_R')
    res = path[index+1:index+4]
    try:
        d_EOS2[res].append(path)
    except:
        d_EOS2.setdefault(res,[])
        d_EOS2[res].append(path)
# to do comparisons, we choose the hybrid that was made with the highest L2 match as an "exact" waveform and see how others differ from it.
he_paths_EOS1 = {}
he_paths_EOS2 = {}
# this picks out the "exact" hybrid from the dictionaries defined above.
for key in d_EOS1:
    paths = d_EOS1[key]
    matches_res_EOS1 = []
    for path in paths:
        h = h5py.File(path, 'r')
        match = h.attrs['hybrid_match'] 
        matches_res_EOS1.append(match)
        h.close()
    he_match_index = np.argmax(np.array(matches_res_EOS1))
    he_paths_EOS1[key]=paths[he_match_index]

for key in d_EOS2:
    paths = d_EOS2[key]
    matches_res_EOS2 = []
    for path in paths:
        print path
        h = h5py.File(path, 'r')
        match = h.attrs['hybrid_match'] 
        matches_res_EOS2.append(match)
        h.close()
    he_match_index = np.argmax(np.array(matches_res_EOS2))
    he_paths_EOS2[key]=paths[he_match_index]

print 'Best Pathes: ', he_paths_EOS1
print 'Best Pathes: ', he_paths_EOS2
##### Now load in he_EOS1 and he_EOS2 and pass to the compare function in pmap
hhe_EOS1 = h5py.File(he_paths_EOS1['R01'], 'r')
EOS_1 = hhe_EOS1.attrs['EOS']
grav_m1_EOS1 = hhe_EOS1.attrs['grav_mass1']
grav_m2_EOS1 = hhe_EOS1.attrs['grav_mass2']
s1x_EOS1 = hhe_EOS1.attrs['spin1x']
s1y_EOS1 = hhe_EOS1.attrs['spin1y']
s1z_EOS1 = hhe_EOS1.attrs['spin1z']
s2x_EOS1 = hhe_EOS1.attrs['spin2x']
s2y_EOS1 = hhe_EOS1.attrs['spin2y']
s2z_EOS1 = hhe_EOS1.attrs['spin2z']
f_lower_hhe_EOS1= hhe_EOS1.attrs['f_lower_at_1MSUN']#/(2.0 * lal.TWOPI * lal.MTSUN_SI*(m1+m2)) ##convert back to Hz
hhe_freq = hhe_EOS1.attrs['hybridize_freq']
hhe_shift_time = hhe_EOS1.attrs['shift_time']
approx_e_EOS1 = hhe_EOS1.attrs['approx']
h_matche = hhe_EOS1.attrs['hybrid_match']
hhe_EOS1.close()

hhe_EOS2 = h5py.File(he_paths_EOS2['R01'], 'r')
EOS_2 = hhe_EOS2.attrs['EOS']
grav_m1_EOS2 = hhe_EOS2.attrs['grav_mass1']
grav_m2_EOS2 = hhe_EOS2.attrs['grav_mass2']
s1x_EOS2 = hhe_EOS2.attrs['spin1x']
s1y_EOS2 = hhe_EOS2.attrs['spin1y']
s1z_EOS2 = hhe_EOS2.attrs['spin1z']
s2x_EOS2 = hhe_EOS2.attrs['spin2x']
s2y_EOS2 = hhe_EOS2.attrs['spin2y']
s2z_EOS2 = hhe_EOS2.attrs['spin2z']
f_lower_hhe_EOS2= hhe_EOS2.attrs['f_lower_at_1MSUN']#/(2.0 * lal.TWOPI * lal.MTSUN_SI*(m1+m2)) ##convert back to Hz
hhe_freq = hhe_EOS2.attrs['hybridize_freq']
hhe_shift_time = hhe_EOS2.attrs['shift_time']
approx_e_EOS2 = hhe_EOS2.attrs['approx']
h_matche = hhe_EOS2.attrs['hybrid_match']
hhe_EOS1.close()
##### masses from simulation have junk after the 3rd decimal place. cut off everything after 3rd decimal
grav_m1_EOS1 = '%.3f' % grav_m1_EOS1
grav_m1_EOS2 = '%.3f' % grav_m1_EOS2
grav_m2_EOS1 = '%.3f' % grav_m2_EOS1
grav_m2_EOS2 = '%.3f' % grav_m2_EOS2
print grav_m1_EOS1,grav_m1_EOS2
print grav_m2_EOS1,grav_m2_EOS2

#### Now check to make sure paramaters from the two simulations agree with eachother
#### ADD CHECK TO MAKE SURE EACH SIMULATION HAS THE SAME NUMBER OF RESOLUTIONS

if grav_m1_EOS1 != grav_m1_EOS2:
    sys.exit('ERROR: grav_m1_EOS1 and grav_m1_EOS2 disagree.')
if grav_m2_EOS1 != grav_m2_EOS2:
    sys.exit('ERROR: grav_m2_EOS1 and grav_m2_EOS2 disagree.')
if (s1x_EOS1,s1y_EOS1,s1z_EOS1,s2x_EOS1,s2y_EOS1,s2z_EOS1) != (s1x_EOS2,s1y_EOS2,s1z_EOS2,s2x_EOS2,s2y_EOS2,s2z_EOS2):
    sys.exit('ERROR: spins from EOS1 and EOS2 disagree.')
if f_lower_hhe_EOS1 != f_lower_hhe_EOS2:
    sys.exit('ERROR: hybrids at different starting frequencies')
h5_dir = 'HybridAnnex/BNS/Comparisons/'+'sz_'+str(s1z_EOS1)+'m1_'+str(grav_m1_EOS1)+'m2_'+str(grav_m2_EOS1)+'/'+EOS_1+'-'+EOS_2+'_'+approx_e_EOS1+'-'+approx_e_EOS2+'_'+psd_type
##########################################
############ MAIN METHOD #################

### The idea for the main method is to compare a set of hybrids that have been made using hybridChan_numerical.py. Each hybrid made will be compared with eachother
### eachother using multi-processing.
### We outline what this main program is doing in a series of steps below
#### 1) Fix an approximate, and EOS. This is done by just reading in hybrids made with a particular approximate and EOS
#### 2) For the set of waveforms labeled EOS1, calculate mismatches between the "exact" EOS1 hybrid and every other hybrid in the EOS1 set made with different match 
####    window lengths. "Exact" hybrid in this case means hybrid made with the highest match
#### 3) For the set of waveforms labeled EOS1, calculate mismatches between the "exact" hybrid and hybrids with the best match from different resolutions. "Exact"
####    hybrids in this case refers to hybrids made with the highest resolution waveform and the highest L2 match
#### 4) For the set of waveforms labeled EOS2, calculate mismatches between the "exact" EOS2 hybrid and every other hybrid in the EOS2 set made with different match 
####    window lengths. "Exact" hybrid in this case means hybrid made with the highest match
#### 5) For the set of waveforms labeled EOS2, calculate mismatches between the "exact" hybrid and hybrids with the best match from different resolutions."Exact"
####    hybrids in this case refers to hybrids made with the highest resolution waveform and the highest L2 match 
#### 6) Now take the exact hybrids from each set labeled EOS1 and EOS2 and take their mismatch.
if __name__ == '__main__':
    p = mp.Pool(mp.cpu_count())
    ### Step 2
    for key in d_EOS1:
        print key
        l2_norm_EOS1 = []
        match_filter_EOS1 = []
        hhm_shift_time_EOS1 = []
        hhm_hybridize_freq_EOS1 = []
        hm_match_EOS1 = []
        overlap_EOS1 = []
        mismatch_EOS1 = []
        hm_norm = [] 
        hybridize_index = []
        hm_paths_EOS1 = d_EOS1[key]
        he_path_EOS1 = he_paths_EOS1[key]
        #### first compute sigmasqr for he outside of the compare function so we dont have to compute this over and over again
        he = h5py.File(he_path_EOS1, 'r')
        grav_m1 = he.attrs['grav_mass1']
        grav_m2 = he.attrs['grav_mass2']
        s1x = he.attrs['spin1x']
        s1y = he.attrs['spin1y']
        s1z = he.attrs['spin1z']
        s2x = he.attrs['spin2x']
        s2y = he.attrs['spin2y']
        s2z = he.attrs['spin2z']
        f_lower_he = he.attrs['f_lower_at_1MSUN']
        approx_he = he.attrs['approx'] 
        sim_name = he.attrs['sim_name']
        h_match_he = he.attrs['hybrid_match'] 
        he_hybridize_freq = he.attrs['hybridize_freq']
        he_shift_time = he.attrs['shift_time']
        # some hybrids have different names for the starting frequency
        try:
            num_start_freq = he.attrs['num_start_freq']
        except:
            num_start_freq = he.attrs['f_low_num']
 
        he.close()
        hep,hec = get_td_waveform(approximant='NR_hdf5',numrel_data=he_path_EOS1,mass1=grav_m1,
                                                                  mass2=grav_m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_he,
                                                                  inclination=inclination)
        ### multiply by stupid pycbc factor
        hep = hep*pycbc_factor
        hec = hec*pycbc_factor
        #Generate the aLIGO ZDHP PSD 
        hep.resize(300000)
        delta_f = 1.0 /hep.duration
        t_len = len(hep)
        f_len = t_len/2 + 1
        if psd_type == 'aLIGOZeroDetHighPower':
            psd = aLIGOZeroDetHighPower(f_len, delta_f, f_low)
        elif psd_type != 'aLIGOZeroDetHighPower':
            psd = pycbc.psd.read.from_txt(psd_path, f_len, delta_f, f_low, is_asd_file=True)
        else:
            sys.exit('ERROR: enter vaild psd type')
        sigma_sqr_he = matchedfilter.sigmasq(hep,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
        print sigma_sqr_he
        he_l2_norm = np.linalg.norm(np.array(hep))
        func_compare_EOS1 = partial(compare,psd,psd_path,hep,hec,distance,f_low,f_high,approx_he,approx_e_EOS1)
        print 'starting comparisons'
        comparisons_res_EOS1 = p.map(func_compare_EOS1,hm_paths_EOS1)    
###return(l2_distance,hep_hmp_match,overlap,mismatch,hmp_norm,h_matchm,hhm_shift_time,hhe_freq,hhm_freq,approx_e,approx_m,EOS_e,EOS_m)
###          0           1            2         3       4       5         6              7        8          9    10       11    12
        print 'done with comparisons'
        print len(comparisons_res_EOS1)
        # the map function returns and array of arrays so we need to organize them into separate arrays.
        for tuple_data in comparisons_res_EOS1:         
            l2_norm_EOS1.append(tuple_data[0]) # L2 match between hybrids made with different match window lengths
            match_filter_EOS1.append(tuple_data[1]) # noise weighted match between hybrids made with different match window lengths
            overlap_EOS1.append(tuple_data[2]) # normalized noise weighted match between hybrids made with different match window lengths
            mismatch_EOS1.append(tuple_data[3]) # 1 - overlap
            hm_norm.append(tuple_data[4])   # snr of hybrids made with different window lengths
            hm_match_EOS1.append(tuple_data[5]) # L2 hybrid match for each hybrid
            hhm_shift_time_EOS1.append(tuple_data[6][0]) # time shifts associated with each hybrid
            hhm_hybridize_freq_EOS1.append(tuple_data[8][0]) # hybridization frequency associated with each hybrid
            hybridize_index.append(tuple_data[6][1]) # index of the hybridization
        data = np.transpose([l2_norm_EOS1,match_filter_EOS1,overlap_EOS1,mismatch_EOS1,hm_norm,hm_match_EOS1,hhm_shift_time_EOS1,hhm_hybridize_freq_EOS1,hybridize_index]) 
        he_data = np.transpose([sigma_sqr_he,he_l2_norm,h_match_he,he_shift_time[0],he_hybridize_freq[0]])
        if not os.path.exists(os.path.dirname(h5_dir)):
            try:
                os.makedirs(os.path.dirname(h5_dir))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        # Now write all the single resolution comparisons (i.e write all the comparison data for step 2)
        with h5py.File(h5_dir+'.h5','a') as fd:
            fd.attrs.create('Group', 'CSUF_GWPAC_Flynn')
            fd.attrs.create('approx_EOS1', approx_e_EOS1)
            fd.attrs.create('EOS1',EOS_1)
            fd.attrs.create('EOS2',EOS_2)
            fd.attrs.create('sim_name_EOS1',sim_name)
            fd.attrs.create('psd',psd_type)
            fd.attrs.create('f_low',f_low)
            fd.attrs.create('f_high',f_high)
            fd.attrs.create('num_start_freq_EOS1',num_start_freq)
            fd.create_dataset('EOS1_window_compare_'+key,data=data)
            fd.create_dataset('EOS1_window_compare_he_data_'+key,data=he_data)
            fd.close()   
        print 'done with'+key
############## 
#### Step 3
#### Now do he comparisons between different resolutions for EOS1. we only do comparisons between highest resolution to other lower resolutions 
#############   
    he_path_array_EOS1 = [he_paths_EOS1['R01']]
    for r01_path in he_path_array_EOS1:
        hm_paths_EOS1 = []
        l2_norm_EOS1 = []
        match_filter_EOS1 = []
        overlap_EOS1 = [] 
        mismatch_EOS1 = []
        hm_norm = []   
        hm_match_EOS1 = []
        hhm_shift_time_EOS1 = []
        hhm_hybridize_freq_EOS1 = []
        sim_names = []
        he = h5py.File(r01_path, 'r')
        grav_m1 = he.attrs['grav_mass1']
        grav_m2 = he.attrs['grav_mass2']
        s1x = he.attrs['spin1x']
        s1y = he.attrs['spin1y']
        s1z = he.attrs['spin1z']
        s2x = he.attrs['spin2x']
        s2y = he.attrs['spin2y']
        s2z = he.attrs['spin2z']
        f_lower_he = he.attrs['f_lower_at_1MSUN']
        approx_he = he.attrs['approx'] 
        h_match_he = he.attrs['hybrid_match'] 
        he_hybridize_freq = he.attrs['hybridize_freq']
        he_shift_time = he.attrs['shift_time']
        he.close()
        hep,hec = get_td_waveform(approximant='NR_hdf5',numrel_data=r01_path,mass1=grav_m1,
                                                                  mass2=grav_m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_he,
                                                                  inclination=inclination)
        ### multiply by stupid pycbc factor
        hep = hep*pycbc_factor
        hec = hec*pycbc_factor
        #Generate the aLIGO ZDHP PSD 
        hep.resize(300000)
        delta_f = 1.0 /hep.duration
        t_len = len(hep)
        f_len = t_len/2 + 1
        if psd_type == 'aLIGOZeroDetHighPower':
            psd = aLIGOZeroDetHighPower(f_len, delta_f, f_low)
        elif psd_type != 'aLIGOZeroDetHighPower':
            psd = pycbc.psd.read.from_txt(psd_path, f_len, delta_f, f_low, is_asd_file=True)
        else:
            sys.exit('ERROR: enter vaild psd type')
        sigma_sqr_he = matchedfilter.sigmasq(hep,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
        he_l2_norm = np.linalg.norm(np.array(hep))
        func_compare_EOS1 = partial(compare,hep,hec,distance,f_low,f_high,approx_he,approx_e_EOS1) 
        func_compare_EOS1 = partial(compare,hep,hec,distance,f_low,f_high,approx_he,approx_e_EOS1)
        for key in dict(d_EOS1.items()[1:]):
            hm_paths_EOS1.append(he_paths_EOS1[key])    
        print 'hm_paths_eos1', hm_paths_EOS1
        print 'starting comparisons'
        # map the comparisons between resolutions of EOS1 now
        comparisons_res_res_EOS1 = p.map(func_compare_EOS1,hm_paths_EOS1)    
        for tuple_data in comparisons_res_res_EOS1:         
            l2_norm_EOS1.append(tuple_data[0])
            match_filter_EOS1.append(tuple_data[1])
            overlap_EOS1.append(tuple_data[2])
            mismatch_EOS1.append(tuple_data[3])
            hm_norm.append(tuple_data[4])   
            hm_match_EOS1.append(tuple_data[5])
            hhm_shift_time_EOS1.append(tuple_data[6][0])
            hhm_hybridize_freq_EOS1.append(tuple_data[8][0])
            sim_names.append(tuple_data[13])    
        res_data = np.transpose([l2_norm_EOS1,match_filter_EOS1,overlap_EOS1,mismatch_EOS1,hm_norm,hm_match_EOS1,hhm_shift_time_EOS1,hhm_hybridize_freq_EOS1,sim_names])
        he_data = np.transpose([sigma_sqr_he,he_l2_norm,h_match_he,he_shift_time[0],he_hybridize_freq[0]])
        with h5py.File(h5_dir+'.h5','a') as fd:
            fd.create_dataset('EOS1_res_compare',data=res_data)
            fd.create_dataset('EOS1_res_compare_he_data',data=he_data)

#####################################
#### Step 4
#### Now compute the same for EOS2
#####################################        
    for key in d_EOS2:
        print key
        l2_norm_EOS2 = []
        match_filter_EOS2 = []
        hhm_shift_time_EOS2 = []
        hhm_hybridize_freq_EOS2 = []
        hm_match_EOS2 = []
        overlap_EOS2 = []
        mismatch_EOS2 = []
        hm_norm = [] 
        hm_paths_EOS2 = d_EOS2[key]
        he_path_EOS2 = he_paths_EOS2[key]
        hybridize_index = []
        #### first compute sigmasqr for he outside of the compare function so we dont have to compute this over and over again
        he = h5py.File(he_path_EOS2, 'r')
        grav_m1 = he.attrs['grav_mass1']
        grav_m2 = he.attrs['grav_mass2']
        s1x = he.attrs['spin1x']
        s1y = he.attrs['spin1y']
        s1z = he.attrs['spin1z']
        s2x = he.attrs['spin2x']
        s2y = he.attrs['spin2y']
        s2z = he.attrs['spin2z']
        f_lower_he = he.attrs['f_lower_at_1MSUN']
        approx_he = he.attrs['approx'] 
        h_match_he = he.attrs['hybrid_match'] 
        he_hybridize_freq = he.attrs['hybridize_freq']
        sim_name = he.attrs['sim_name']
        try:
            num_start_freq = he.attrs['num_start_freq']
        except:
            num_start_freq = he.attrs['f_low_num']
        he_shift_time = he.attrs['shift_time']
        he.close()
        hep,hec = get_td_waveform(approximant='NR_hdf5',numrel_data=he_path_EOS2,mass1=grav_m1,
                                                                  mass2=grav_m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_he,
                                                                  inclination=inclination)
        ### multiply by stupid pycbc factor
        hep = hep*pycbc_factor
        hec = hec*pycbc_factor
        #Generate the aLIGO ZDHP PSD 
        hep.resize(300000)
        delta_f = 1.0 /hep.duration
        t_len = len(hep)
        f_len = t_len/2 + 1
        if psd_type == 'aLIGOZeroDetHighPower':
            psd = aLIGOZeroDetHighPower(f_len, delta_f, f_low)
        elif psd_type == 'CE':
            psd_path = '/home/jsread/O2PSDS/lalinferencemcmc-1-V1H1L1-1187008882.45-20.hdf5L1-PSD.txt'
            psd = pycbc.psd.read.from_txt(psd_path, f_len, delta_f, f_low, is_asd_file=True)
        else:
            sys.exit('ERROR: enter vaild psd type') 
        sigma_sqr_he = matchedfilter.sigmasq(hep,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
        he_l2_norm = np.linalg.norm(np.array(hep))
        func_compare_EOS2 = partial(compare,hep,hec,distance,f_low,f_high,approx_he,approx_e_EOS2)
        print 'starting comparisons'
        comparisons_res_EOS2 = p.map(func_compare_EOS2,hm_paths_EOS2)    
###return(l2_distance,hep_hmp_match,overlap,mismatch,hmp_norm,h_matchm,hhm_shift_time,hhe_freq,hhm_freq,approx_e,approx_m,EOS_e,EOS_m)
###          0           1            2         3       4       5         6              7        8          9    10       11    12
        for tuple_data in comparisons_res_EOS2:         
            l2_norm_EOS2.append(tuple_data[0])
            match_filter_EOS2.append(tuple_data[1])
            overlap_EOS2.append(tuple_data[2])
            mismatch_EOS2.append(tuple_data[3])
            hm_norm.append(tuple_data[4])   
            hm_match_EOS2.append(tuple_data[5])
            hhm_shift_time_EOS2.append(tuple_data[6][0])
            hhm_hybridize_freq_EOS2.append(tuple_data[8][0])
            hybridize_index.append(tuple_data[6][1])
        data = np.transpose([l2_norm_EOS2,match_filter_EOS2,overlap_EOS2,mismatch_EOS2,hm_norm,hm_match_EOS2,hhm_shift_time_EOS2,hhm_hybridize_freq_EOS2,hybridize_index])
        he_data = np.transpose([sigma_sqr_he,he_l2_norm,h_match_he,he_shift_time[0],he_hybridize_freq[0]])
        if not os.path.exists(os.path.dirname(h5_dir)):
            try:
                os.makedirs(os.path.dirname(h5_dir))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        with h5py.File(h5_dir+'.h5','a') as fd:
            fd.attrs.create('Group', 'CSUF_GWPAC_Flynn')
            fd.attrs.create('approx_EOS2', approx_e_EOS2)
            fd.attrs.create('sim_name_EOS2', sim_name)
            fd.attrs.create('num_start_freq_EOS2',num_start_freq)
            fd.create_dataset('EOS2_window_compare_'+key,data=data)
            fd.create_dataset('EOS2_window_compare_he_data_'+key,data=he_data)
            fd.close()   
            fd.close()
##############
#### Step 5
#### Now do he comparisons between different resolutions for EOS2. we only do comparisons between highest resolution to other lower resolutions 
#############   
    he_path_array_EOS2 = [he_paths_EOS2['R01']]
    for r01_path in he_path_array_EOS2:
        print 'starting EOS2 res comparisons'
        hm_paths_EOS2 = []
        l2_norm_EOS2 = []
        match_filter_EOS2 = []
        overlap_EOS2 = [] 
        mismatch_EOS2 = []
        hm_norm = []   
        hm_match_EOS2 = []
        hhm_shift_time_EOS2 = []
        hhm_hybridize_freq_EOS2 = []
        sim_names = []
        he = h5py.File(r01_path, 'r')
        grav_m1 = he.attrs['grav_mass1']
        grav_m2 = he.attrs['grav_mass2']
        s1x = he.attrs['spin1x']
        s1y = he.attrs['spin1y']
        s1z = he.attrs['spin1z']
        s2x = he.attrs['spin2x']
        s2y = he.attrs['spin2y']
        s2z = he.attrs['spin2z']
        f_lower_he = he.attrs['f_lower_at_1MSUN']
        approx_he = he.attrs['approx'] 
        h_match_he = he.attrs['hybrid_match'] 
        he_hybridize_freq = he.attrs['hybridize_freq']
        he_shift_time = he.attrs['shift_time']
        he.close()
        hep,hec = get_td_waveform(approximant='NR_hdf5',numrel_data=r01_path,mass1=grav_m1,
                                                                  mass2=grav_m2,
                                                                  spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                  spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                  delta_t=delta_t,distance=distance,
                                                                  f_lower=f_lower_he,
                                                                  inclination=inclination)
        ### multiply by stupid pycbc factor
        hep = hep*pycbc_factor
        hec = hec*pycbc_factor
        #Generate the aLIGO ZDHP PSD
        hep.resize(300000) 
        delta_f = 1.0 /hep.duration
        t_len = len(hep)
        f_len = t_len/2 + 1
        if psd_type == 'aLIGOZeroDetHighPower':
            psd = aLIGOZeroDetHighPower(f_len, delta_f, f_low)
        elif psd_type == 'CE':
            psd_path = '/home/jsread/O2PSDS/lalinferencemcmc-1-V1H1L1-1187008882.45-20.hdf5L1-PSD.txt'
            psd = pycbc.psd.read.from_txt(psd_path, f_len, delta_f, f_low, is_asd_file=True)
        else:
            sys.exit('ERROR: enter vaild psd type')
        sigma_sqr_he = matchedfilter.sigmasq(hep,psd=psd,low_frequency_cutoff=f_low, high_frequency_cutoff=f_high)
        he_l2_norm = np.linalg.norm(np.array(hep))
        func_compare_EOS2 = partial(compare,hep,hec,distance,f_low,f_high,approx_he,approx_e_EOS2)
        for key in dict(d_EOS2.items()[1:]):
            hm_paths_EOS2.append(he_paths_EOS2[key])    
        print 'starting comparisons'
        comparisons_res_res_EOS2 = p.map(func_compare_EOS2,hm_paths_EOS2)    
        for tuple_data in comparisons_res_res_EOS2:         
            l2_norm_EOS2.append(tuple_data[0])
            match_filter_EOS2.append(tuple_data[1])
            overlap_EOS2.append(tuple_data[2])
            mismatch_EOS2.append(tuple_data[3])
            hm_norm.append(tuple_data[4])   
            hm_match_EOS2.append(tuple_data[5])
            hhm_shift_time_EOS2.append(tuple_data[6][0])
            hhm_hybridize_freq_EOS2.append(tuple_data[8][0])
            sim_names.append(tuple_data[13])    
        res_data = np.transpose([l2_norm_EOS2,match_filter_EOS2,overlap_EOS2,mismatch_EOS2,hm_norm,hm_match_EOS2,hhm_shift_time_EOS2,hhm_hybridize_freq_EOS2,sim_names])
        he_data = np.transpose([sigma_sqr_he,he_l2_norm,h_match_he,he_shift_time[0],he_hybridize_freq[0]])
        with h5py.File(h5_dir+'.h5','a') as fd:
            fd.create_dataset('EOS2_res_compare',data=res_data)
            fd.create_dataset('EOS2_res_compare_he_data',data=he_data)
#############
#### Step 6
#### Now calcualte match between highest resolution waveform to each eos 
#############
    for he_EOS1_path in he_path_array_EOS1:
        for he_EOS2_path in he_path_array_EOS2:
            l2_norm_EOS2 = []
            match_filter_EOS2 = []
            overlap_EOS2 = [] 
            mismatch_EOS2 = []
            h_norm_EOS2 = []   
            h_match_EOS2 = []
            h_shift_time_EOS2 = []
            h_hybridize_freq_EOS2 = []
            sim_names = []
            he = h5py.File(he_EOS1_path, 'r')
            grav_m1 = he.attrs['grav_mass1']
            grav_m2 = he.attrs['grav_mass2']
            s1x = he.attrs['spin1x']
            s1y = he.attrs['spin1y']
            s1z = he.attrs['spin1z']
            s2x = he.attrs['spin2x']
            s2y = he.attrs['spin2y']
            s2z = he.attrs['spin2z']
            f_lower_he = he.attrs['f_lower_at_1MSUN']
            approx_he = he.attrs['approx'] 
            h_match_he = he.attrs['hybrid_match'] 
            he_hybridize_freq = he.attrs['hybridize_freq']
            he_shift_time = he.attrs['shift_time']
            he.close()
            hep1,hec1 = get_td_waveform(approximant='NR_hdf5',numrel_data=he_EOS1_path,mass1=grav_m1,
                                                                      mass2=grav_m2,
                                                                      spin1z=s1z,spin1x=s1x,spin1y=s1y,
                                                                      spin2z=s2z,spin2x=s2x,spin2y=s2y,
                                                                      delta_t=delta_t,distance=distance,
                                                                      f_lower=f_lower_he,
                                                                      inclination=inclination)

            
            ### multiply by stupid pycbc factor
            hep1 = hep1*pycbc_factor
            hec1 = hec1*pycbc_factor
            EOS1_EOS2_compare = compare(hep1,hec1,distance,f_low,f_high,approx_he,approx_e_EOS1,he_EOS2_path) 
            l2_norm_EOS2.append(EOS1_EOS2_compare[0])
            match_filter_EOS2.append(EOS1_EOS2_compare[1])
            overlap_EOS2.append(EOS1_EOS2_compare[2])
            mismatch_EOS2.append(EOS1_EOS2_compare[3])
            h_norm_EOS2.append(EOS1_EOS2_compare[4])   
            h_match_EOS2.append(EOS1_EOS2_compare[5])
            h_shift_time_EOS2.append(EOS1_EOS2_compare[6][0])
            h_hybridize_freq_EOS2.append(EOS1_EOS2_compare[8][0])
            sim_names.append(EOS1_EOS2_compare[13])    
            res_data = np.transpose([l2_norm_EOS2,match_filter_EOS2,overlap_EOS2,mismatch_EOS2,h_norm_EOS2,h_match_EOS2,h_shift_time_EOS2,h_hybridize_freq_EOS2,sim_names])
            with h5py.File(h5_dir+'.h5','a') as fd:
                fd.create_dataset('model1_model2_compare',data=res_data)  
