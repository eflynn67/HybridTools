import sys
import time
import h5py
import numpy as np
import os 
import re
import scipy.signal as sig
import hashlib
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
import random
#import matplotlib
#import matplotlib.pyplot as plt
import romspline
#################################
#################################
## Define constants 
TWOPI = 2.0*np.pi
MRSUN_SI = 1476.6250614
PC_SI = 3.08567758149e+16
MTSUN_SI = 4.92549102554e-06
solar_mass_mpc = MRSUN_SI / (1e6*PC_SI)
Sph_Harm = 0.6307831305 ## 2-2 spherical harmoniic at theta = 0 phi = 90 
#################################
# Define Sample rate
sample_rate = 4096.0*8.0
delta_t = 1.0/sample_rate
#################################

##################################
###### INPUT PARAMETERS
##################################
### First, input the names of the pycbc approximates you want to hybridize. Must be in the form [string1, string2,....]

PN_names = ['SEOBNRv4T'] ## TaylorT1,TaylorT2,TaylorT3,TaylorT4,SEOBNRv4T,TEOBResum_ROM,IMRPhenomD_NRTidal,IMRPhenomPv2_NRTidal
### names of the simulations you want to use. glob is useful since you can grab all resolutions of a single simulation using a wildcard (*). 
### It is assumed that all numerical simulations are in the pycbc numerical injection format.  
### CoRe_THC_0030_R0*_r00400.h5
### CoRe_BAM_0097_R0*_r00850.h5
### CoRe_BAM_0003_R0*_r00900.h5
sim_paths = glob.glob('/home/ericf2/Simulations/BNS/CoRe/unsorted/CoRe_BAM_0003_R0*_r00900.h5')
### grab physical parameters from simulation.
wave_read = h5py.File(sim_paths[0], 'r')
lambda_1 = wave_read.attrs['tidal-lambda1-l2'] 
lambda_2 = wave_read.attrs['tidal-lambda2-l2'] 
file_label = wave_read.attrs['alternative-names'] 
EOS = wave_read.attrs['EOS-name']
grav_m1 = wave_read.attrs['mass1-msol'] 
grav_m2 = wave_read.attrs['mass2-msol'] 
bary_m1 = 0.0#wave_read.attrs['baryon-mass1']
bary_m2 = 0.0#wave_read.attrs['baryon-mass2']
total_mass = grav_m1 + grav_m2 
m1 = wave_read.attrs['mass1']
m2 = wave_read.attrs['mass2']
### Some simulations have very small spins less than 10^-7. These values tend to cause problems when pycbc attempts to generate the waveforms. So we just floor spins below 10^-7
spins = [wave_read.attrs['spin1x'],wave_read.attrs['spin1y'],wave_read.attrs['spin1z'],wave_read.attrs['spin2x'],wave_read.attrs['spin2y'],wave_read.attrs['spin2z']]
for i,spin in enumerate(spins):
    if abs(spin) < 10**-7:
        spins[i] = np.floor(abs(spin))
s1x = spins[0]
s1y = spins[1]
s1z = spins[2]
s2x = spins[3]
s2y = spins[4]
s2z = spins[5]
### orbital angular momentum vectors (LNhat) and separation vectors (nhat) are simulation metadata. We import it in here so we can write it in the hybrid metadata
LNhatx = 0.0#wave_read.attrs['LNhatx']
LNhaty = 0.0#wave_read.attrs['LNhaty']
LNhatz = 1.0#wave_read.attrs['LNhatz']
n_hatx = 1.0#wave_read.attrs['nhatx']
n_haty = 0.0#wave_read.attrs['nhaty']
n_hatz = 0.0#wave_read.attrs['nhatx']
#delta_tidal = 1.0
### we can create a set of hybrids with different tidal parameter values. If you want a single lambda, just input a single number into the array.
tidal_range1 = [lambda_1] #### code is set to iterate over lamba values so these need to be in arrays or lists
tidal_range2 = [lambda_2]

ip1 = 0 ### initial window point on approximate
#final_start_index = 100
#delta_start_index = 50
starting_index = [0] 
### we can create hybrids with different numerical simulation starting points. Since we are treating the junk radiation using treat_junk() from hybrid modules, we tend not iterate 
### over this parameter by just setting starting index to [0] (i.e no changes in starting point).
match_i = 0 ### dummy parameter for the beginning window index for the numerical. Used for python multi-processing.
type_sim = 'BNS' # type of binary system we are hybridizing
f_low = 60 # lower frequency bound for short hybrids. Short hybrids are used for distinguishability and characteristics checks 
f_low_long = 60 # lower frequency for best hybrid. if you do not want to make a long hybrid, set this parameter to string 'None'
distance = 1.0 ### in Mpcs.
inclination = 0
coa_phase = 0
#### Hybrid Params
match_lower = 0.98 # lower match threshold for creating hybrids. procedure randomly samples hybrids made with matches above this threshold.  
num_hybrids = 100 # number of short hybrids you want to make and write to disk.  
#### treat numerical data for junk radiation
upper_f_bound = 1 ### upper bound on starting frequency difference. See treat_junk() and find_junk() in hybridmodules.py for more details
wave_read.close()


### Below is a function used to write hybrid waveforms in numerical format. This function can be used to write multiple hybrids at one time using python multiprocessing
#### We use format = 1. Problem with ref_f in format=2.
def writeHybrid_h5(path_name_part,metadata,approx,sim_name,h1,h1_ts,h2,h2_ts,delta_t,ip2,fp2):
    solar_mass_mpc = MRSUN_SI /((1* 10**6)*PC_SI)
    grav_m1 = metadata[0][0]
    grav_m2 = metadata[0][1]
    baryon_m1 = metadata[1][0]
    baryon_m2 = metadata[1][1]
    m1 = metadata[2][0]
    m2 = metadata[2][1]
    s1x = metadata[3][0]
    s1y = metadata[3][1]
    s1z = metadata[3][2]
    s2x = metadata[3][3]
    s2y = metadata[3][4]
    s2z = metadata[3][5]
    LNhatx = metadata[4][0]
    LNhaty = metadata[4][1]
    LNhatz = metadata[4][2]
    n_hatx = 1.0#metadata[5][0]
    n_haty = 0.0#metadata[5][1]
    n_hatz = 0.0#metadata[5][2]
    tidal_1 = metadata[6][0]
    tidal_2 = metadata[6][1]
    type_sim = metadata[7]
    f_low = metadata[8]
    EOS = metadata[9]
    f_low_num = metadata[10]
    M = 300 ## length of the windowing function 
    total_mass = grav_m1 + grav_m2 ### used masses are the masses used in the characterization procedure
    ### hybridize numerical (h2) and analytical waveform (h1) at a particular match window length set by fp2
    hybridPN_Num = hy.hybridize(h1,h1_ts,h2,h2_ts,match_i=0,match_f=fp2,delta_t=delta_t,M=M,info=1)
    shift_time = (hybridPN_Num[6],fp2)
    hh_freq = (hybridPN_Num[7],fp2)
    match = hybridPN_Num[0]
    ### path_name_part contains simulation details that must be passed into this function.
    path_name = path_name_part+'fp2_'+str(fp2)+'.h5'
    f_low_M = f_low * (TWOPI * total_mass * MTSUN_SI)
    with h5py.File(path_name,'w') as fd:
        mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(grav_m1, grav_m2)
        fd.attrs['type'] = 'Hybrid:%s'%type_sim
        fd.attrs['hgroup'] = 'Read_CSUF'
        fd.attrs['Format'] = 1
        fd.attrs['Lmax'] = 2
        fd.attrs['approx'] = approx
        fd.attrs['sim_name'] = sim_name
        fd.attrs['f_lower_at_1MSUN'] = f_low
        fd.attrs['eta'] = eta
        fd.attrs['spin1x'] = s1x
        fd.attrs['spin1y'] = s1y
        fd.attrs['spin1z'] = s1z
        fd.attrs['spin2x'] = s2x
        fd.attrs['spin2y'] = s2y
        fd.attrs['spin2z'] = s2z
        fd.attrs['LNhatx'] = LNhatx
        fd.attrs['LNhaty'] = LNhaty
        fd.attrs['LNhatz'] = LNhatz
        fd.attrs['nhatx'] = 1.0#n_hatx
        fd.attrs['nhaty'] = 0.0#n_haty
        fd.attrs['nhatz'] = 0.0#n_hatz
        fd.attrs['mass1'] = m1
        fd.attrs['mass2'] = m2
        fd.attrs['grav_mass1'] = grav_m1
        fd.attrs['grav_mass2'] = grav_m2
        fd.attrs['baryon_mass1'] = baryon_m1
        fd.attrs['baryon_mass2'] = baryon_m2
        fd.attrs['lambda1'] = tidal_1
        fd.attrs['lambda2'] = tidal_2
        fd.attrs['fp1'] =  len(h1)
        fd.attrs['ip2'] = ip2
        fd.attrs['hybrid_match'] = match
        fd.attrs['shift_time'] = shift_time
        fd.attrs['hybridize_freq'] = hh_freq
        fd.attrs['EOS'] = EOS
        fd.attrs['f_low_num'] = f_low_num
        gramp = fd.create_group('amp_l2_m2')
        grphase = fd.create_group('phase_l2_m2')
        times = hybridPN_Num[5][0]
        hplus = hybridPN_Num[5][1]
        hcross = hybridPN_Num[5][2]
        massMpc = total_mass*solar_mass_mpc
        hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
        hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
        times_M = times / (MTSUN_SI * total_mass)
        HlmAmp = wfutils.amplitude_from_polarizations(hplusMpc,hcrossMpc).data
        HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data
        sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
        sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
        sAmph.write(gramp)
        sPhaseh.write(grphase)
        fd.close()
    ## this function returns the hybridization frequency (hh_freq) and the time shift (shift_time) associated with each hybrid.
    return(shift_time,hh_freq)
### First remove as much junk radiation as we can using treat_junk() from hybridmodules.py
treated_num_data = hy.treat_junk(sim_paths,total_mass,inclination,distance,coa_phase,upper_f_bound,f_shift=4,cycle_number=8.0,delta_t=delta_t)
wavedata = treated_num_data[0]
num_metadata = treated_num_data[1]

##########################################
############ MAIN METHOD #################

### The idea for the main method is to iterate over the window length, resolutions, tidal parameters (if you want), and approximate choice.
### We outline what this main program is doing in a series of steps below
#### 1) Fix a simulation resolution (coming from the glob function sim_names. Numerical data is treated using treat_junk().) 
#### 2) Fix an approximate model (coming from the array PN_names under the input parameters) 
#### 3) Fix tidal parameters (coming from tidal_range1 and tidal_range2)
#### 4) Fix a numerical starting window point (coming from starting_index)
#### 5) calculate L2 matches for window lengths ranging from [1,len(h2)] on h2 (the numerical). This is done by mapping match_generator_num() from hybridmodules.pyto multiple 
####    processors
#### 6) calculate L2 matches for window lengths ranging from [len(h1)-len(h2)+1,len(h1)+1]. This is done by mapping match_generator_PN() from hybridmodules.py to multiple processors
#### 7) using the match values calculated from step 5), randomly sample from h2 window lengths that produce matches above the match threshold (variable match_lower) set by user
#### 8) Write sample hybrids to disk in standard numerical H5 format. Write HybridChars.h5 file that contains information about the construction parameters (details are given below)
#### 9) If a f_low_long was given, write a long hybrid. this hybrid has the highest match value for paritcular ip2 (beginning index of numerical) and fp2 (ending index of numerical)
#### 10) Loops begins again.
  
if __name__ == '__main__':
    ### setup multiprocessing pool. mp.Pool(mp.cpu_count()) just counts how many processors are available and prepares them. you can specify a particular number less than what
    ### mp.cpu_count() returns if you want. 
    p = mp.Pool(mp.cpu_count())
    for i,num_res in enumerate(wavedata):
        for approx  in PN_names:
            for tidal_1 in tidal_range1:
                for tidal_2 in tidal_range2:                        
                    match_funcs = [] # empty array to attach arrays of match_values that have first two_cycles worth of numerical data from match_generator_num() removed. 
                    matches_area = [] # empty array to attach the integrals of the match vs time_slice curve above match threshold.
                    # first generate approximate 
                    PN_model = hy.getApprox('pycbc',approx,m1=grav_m1,m2=grav_m2,f_ref=0.0,f_low=f_low,distance=distance,delta_t=delta_t,
                                s1x=s1x,s1y=s1y,s1z=s1z,s2x=s2x,s2y=s2y,s2z=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                    h1 = PN_model[1]
                    h1_ts = PN_model[0]
                    h1_fs = hy.m_frequency_from_polarizations(np.real(h1),np.imag(h1),delta_t)
                    h2_full = num_res[1] 
                    h2_ts_full = num_res[0]      
                    full_freq = num_res[2]
                    starting_freq = num_metadata[i][0] # starting frequency of the treated numerical waveform in Hz
                    sim_name = num_metadata[i][3] 
                    sim_name = sim_name.replace(':','_') # sometimes the simulations have : in their names. this fucks with the file paths
                    # metadata to pass to the  writeHybrid_h5() function defined above
                    metadata = [(grav_m1,grav_m2),(bary_m1,bary_m2),(m1,m2),(s1x,s1y,s1z,s2x,s2y,s2z),(LNhatx,LNhaty,LNhatz),(n_hatx,n_haty,n_hatz),(tidal_1,tidal_2),type_sim,f_low,EOS, starting_freq]
                    for ip2 in starting_index:
                        #if multiple ip2 values are given in starting_index array, this section finds the ip2 that produces the highest maximum match for a particular fp2
                        best_numz_indices = [] # empty array to place the indices corresponding t the time slice that produces the high match
                        hh_freqs = []
                        window_index = []
                        match_fp2 = []    
                        best_PNz_indices = []
                        path_name_data = 'HybridAnnex/'+type_sim+'/'+sim_name+'/'+file_label+'/'+approx+'/'+approx+'ip2_'+str(ip2)+'_'+str(tidal_1)+'_'+str(tidal_2)+'/'
			h2 = h2_full[ip2:]
                        h2_ts = h2_ts_full[ip2:]
                        h2_fs = full_freq[ip2:]
                        starting_freq = full_freq[ip2]
                        # ignore matches produced by only allowing less than two oscillations into the correlation integral from match_generator_num().
                        time_two_osc = 2.0/starting_freq 
                        time_shift = h2_ts - h2_ts[0]
                        two_osc_index = hy.find_nearest(time_shift,time_two_osc)
                        # define iterable array so that we can map different numerical window match calculations to different processors
                        num_iteration = np.arange(1,len(h2)+1,1,dtype=np.int64)
                        # define iterable array so that we can map different match approximate window match calculations to different processors 
                        PN_iteration = np.arange(len(h1)-len(h2)+1,len(h1)+1,1,dtype=np.int64)
                        # partial keeps all the other parameters required by match_generator_num and match_generator_PN constant while keeping fp2 and fp1 variable
                        func_match_h2 = partial(hy.match_generator_num, h1, h2,delta_t,match_i)
                        func_match_h1 = partial(hy.match_generator_PN, h1, h2, match_i)
                        # calculate matches for each fp1 (ending of approximate window). p.map returns a multi-dimensional array (matches, hybridizaton freqs, window indicies)
                        match_fp1 = p.map(func_match_h1,PN_iteration)
                        # calculate matches for each fp2 (ending of numerical window)
                        char_data = p.map(func_match_h2,num_iteration)
                        #pull out data from the map function. 
                        for data in char_data:
                            match_fp2.append(data[0])
                            hh_freqs.append(data[1])
                            window_index.append(data[2])
                        match_fp2 = np.array(match_fp2)
                        match_fp1 = np.array(match_fp1)
                        match_area_fp2 = sci.integrate.trapz(match_fp2,h2_ts) 
                        matches_area.append(match_area_fp2)
                        
                        match_funcs.append(match_fp2[two_osc_index[1]:])
                        # find the region of matches that are higher than match_lower 
                        best_numz = np.where(match_fp2[two_osc_index[1]:]>=match_lower)
                        # find the indices corresponding to time slices that produce matches greater than match_lower
                        best_numz_indices_temp = np.argwhere(match_fp2[two_osc_index[1]:]>=match_lower)
                        for k in best_numz_indices_temp:
                            best_numz_indices.append(k[0])
                        best_PNz = np.where(match_fp1>=match_lower)
                        best_PNz_indices_temp = np.argwhere(match_fp1>=match_lower)
                        for k in best_PNz_indices_temp:
                            best_PNz_indices.append(k[0])
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
                        with h5py.File(path_name_data+'HybridChars.h5','w') as fd:
                            fd.attrs.create('group', 'GWPAC')
                            fd.attrs.create('type', 'Hybrid:%s'%type_sim)
                            fd.attrs.create('approx', approx)
                            fd.attrs.create('sim_name', sim_name)
                            fd.attrs.create('ip2',ip2)
                            fd.attrs.create('num_start_freq',starting_freq)
                            fd.create_dataset('match_PN_fp', data=match_fp1)
                            fd.create_dataset('match_num_fp', data=match_fp2)
                            fd.create_dataset('hybridize_freq',data=hh_freqs)
                            fd.create_dataset('window_index',data=window_index)
                            fd.create_dataset('time_slices',data=h2_ts)
                            fd.create_dataset('freq_slices',data=h2_fs)
                            fd.create_dataset('tidal_rangeA',data=tidal_range1)
                            fd.create_dataset('tidal_rangeB',data=tidal_range2)
                            fd.create_dataset('bestFreqRange_sim',data=freq_range_num)
                            fd.create_dataset('bestTimeRange_sim',data=time_range_num)
                            fd.create_dataset('bestFreqRange_approx',data=freq_range_PN)
                            fd.create_dataset('bestTimeRange_approx',data=time_range_PN)
                            fd.close()
                        if len(best_numz_indices[two_osc_index[1]:])>num_hybrids:
                            index_choices = random.sample(best_numz_indices[two_osc_index[1]:],num_hybrids)
                        else:
                            index_choices = best_numz_indices[two_osc_index[1]:]
                        path_name_part = path_name_data+sim_name+'_'+EOS+'_'+approx 
                        path_name_part = path_name_part.replace(':','_')
			func_writeHybrid_h5 = partial(writeHybrid_h5,path_name_part,metadata,approx,sim_name,h1,h1_ts,h2,h2_ts,delta_t,ip2)
                        print index_choices
                        hh_times_freqs=p.map(func_writeHybrid_h5,index_choices)
                    if f_low_long != 'None':
#### now make a really long hybrid with highest match
                        best_ip2 = starting_index[np.argmax(matches_area)] 
                        best_fp2 = np.argmax(np.array(match_funcs[np.argmax(matches_area)]))   
                        #print best_ip2,'ip2'
                        #print best_fp2,'fp2'
                        long_PN = hy.getApprox('pycbc',approx,m1=grav_m1,m2=grav_m2,f_ref=0.0,f_low=f_low_long,distance=distance,delta_t=delta_t,
                                                s1x=s1x,s1y=s1y,s1z=s1z,s2x=s2x,s2y=s2y,s2z=s2z,inclination=inclination,tidal1=tidal_1,tidal2=tidal_2)
                        best_hybrid = hy.hybridize(long_PN[1],long_PN[0],h2[best_ip2:],h2_ts[best_ip2:],match_i=0,match_f=best_fp2,delta_t=delta_t,M=300,info=1)
                        shift_time = best_hybrid[6]
                        hh_freq = best_hybrid[7]
                        long_hybrid_name = 'HybridAnnex/'+type_sim+'/'+sim_name+'/'+file_label+'/'+approx+'/'+sim_name+'_'+EOS+'_'+approx+'ip2_'+str(best_ip2)+'fp2_'+str(best_fp2)+'_flow'+str(f_low_long)+'.h5'
                        f_low_M = f_low_long * (TWOPI * total_mass * MTSUN_SI)
                        with h5py.File(long_hybrid_name,'w') as fd:
                            mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(grav_m1, grav_m2)
                            hashtag = hashlib.md5()
                            fd.attrs['type'] = 'Hybrid:%s'%type_sim
                            hashtag.update(fd.attrs['type'])
                            fd.attrs['hashtag']  = hashtag.digest()
                            fd.attrs['group'] = 'GWPAC'
                            fd.attrs['Format'] = 1
                            fd.attrs['Lmax'] = 2
                            fd.attrs['approx'] = approx
                            fd.attrs['sim_name'] =  sim_name
                            fd.attrs['f_lower_at_1MSUN'] = f_low_M
                            fd.attrs['f_low_num'] = starting_freq
                            fd.attrs['eta'] = eta
                            fd.attrs['spin1x'] = s1x
                            fd.attrs['spin1y'] = s1y
                            fd.attrs['spin1z'] = s1z
                            fd.attrs['spin2x'] = s2x
                            fd.attrs['spin2y'] = s2y
                            fd.attrs['spin2z'] = s2z
                            fd.attrs['LNhatx'] = LNhatx
                            fd.attrs['LNhaty'] = LNhaty
                            fd.attrs['LNhatz'] = LNhatz
                            fd.attrs['nhatx'] = n_hatx
                            fd.attrs['nhaty'] = n_haty
                            fd.attrs['nhatz'] = n_hatz
                            fd.attrs['mass1'] = m1
                            fd.attrs['mass2'] = m2
                            fd.attrs['grav_mass1'] = grav_m1
                            fd.attrs['grav_mass2'] = grav_m2
                            fd.attrs['baryon_mass1'] = bary_m1
                            fd.attrs['baryon_mass2'] = bary_m2
                            fd.attrs['lambda1'] = tidal_1
                            fd.attrs['lambda2'] = tidal_2
                            fd.attrs['PN_fp1'] =  len(h1)
                            fd.attrs['num_ip2'] = best_ip2
                            fd.attrs['num_start_freq'] = full_freq[best_ip2]
                            fd.attrs['hybrid_match'] = best_hybrid[0]
                            fd.attrs['shift_time'] = shift_time
                            fd.attrs['hybridize_freq'] = hh_freq
                            fd.attrs['EOS'] = EOS
                            gramp = fd.create_group('amp_l2_m2')
                            grphase = fd.create_group('phase_l2_m2')
                            times = best_hybrid[5][0]
                            hplus = best_hybrid[5][1]
                            hcross = best_hybrid[5][2]
                            massMpc = total_mass*solar_mass_mpc
                            hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
                            hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
                            times_M = times / (MTSUN_SI * total_mass)
                            HlmAmp = wfutils.amplitude_from_polarizations(hplusMpc,hcrossMpc).data
                            HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data
                            sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
                            sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
                            sAmph.write(gramp)
                            sPhaseh.write(grphase)
                            fd.close()
                    else:
                        break        
'''
#end_time = (time.time() - start_time)/60.0
#print 'end_time (mins)', end_time
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
