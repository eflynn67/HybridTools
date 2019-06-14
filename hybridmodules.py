from functools import partial
import h5py
import cmath
import time
import scipy as sci
import numpy as np
import scipy.signal as sig
import H5Graph
import lal
import lalsimulation as lalsim
import re
from scipy import interpolate as inter
import multiprocessing as mp
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries
#import matplotlib.pyplot as plt
## def _poolinit() is just a profiler. The profiler outputs .out file which shows the time it takes to run functions in the code. Useful 
## for profiling the hybrid characterization script. 
def _poolinit():
	global prof
	prof = cProfile.Profile()
	def finish():
		prof.dump_stats('./Profiles/profile-%s.out' % mp.current_process().pid)
	mp.util.Finalize(None,finish,exitpriority = 1)

### find_nearest(): finds value in an array closest to user specified value. Note: this does not consider repeated values. It just gives the first closest value.
### returns the closest value and the index it was found at.
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx],idx

### takes a 2-d array and returns two 1-d arrays
def twodimarray_to_2array(array):
    first = []
    second = []
    for i in np.arange(0,len(array),1):
        first.append(array[i][0])
        second.append(array[i][1])
    return (first,second)
### similar to 2-d array function except its for 4-d arrays
def fourdimarray_to_4array(array):
    first = []
    second = []
    third = []
    fourth = []
    for i in np.arange(0,len(array),1):
        first.append(array[i][0])
        second.append(array[i][1])
        third.append(array[i][2])
        forth.append(array[i][3])
    return (first,second)

#### Converts polarizations in geometrical units(rh/M) to SI units (unitless). Takes in array with strain (h) data and the total mass of the binary system in Msol and outputs an array with scaled strain (h) data. Typically this is the format SXS waveforms are written in 

def rhOverM_to_SI(polarization,total_mass):
        solar_mass_mpc = 1.0/(1.0e6 * lal.PC_SI/lal.MRSUN_SI) 
        h_conversion = total_mass*solar_mass_mpc
        return polarization*h_conversion
### Converts time series in geometric units (t/M) to SI units (seconds). Takes in array with time (t/M) and the total mass of binary in Msol and outputs an array with in seconds)
def tOverM_to_SI(times,total_mass):
        t_conversion = total_mass*lal.MTSUN_SI
        return times*t_conversion
### Converts h (unitless SI) to rh/M geometric units. Takes in array with h and total mass of the binary in Msol. Outputs scaled array with h in rh/M units
def SI_to_rhOverM(polarization,total_mass):
	solar_mass_mpc = lal.MRSUN_SI/(1e6*lal.PC_SI)
        h_conversion = solar_mass_mpc/total_mass
        return polarization*h_conversion
### Converts t (seconds) to t/M geometric units. Takes in array with t (seconds) and total mass of binary in Msol. Outputs scaled array with t in t/M
def SI_to_rOverM(times,total_mass):
        t_conversion = total_mass*lal.MTSUN_SI
        return times/t_conversion
### find_junk(): function that attempts to isolate a region in the beginning of a numerical waveform that has junk radiation. It does this by creating a window in the beginning of 
### waveform with a length set by the number of cycles (the cycle_number variable). Note: This method does not always work as will be shown below
def find_junk(filepath,total_mass,inclination,distance,coa_phase,upper_f_bound,delta_t,f_shift=0,cycle_number=4.0,):
    f = h5py.File(filepath, 'r') 
    database_key = f.attrs['name']
    num_start_freq = f.attrs['f_lower_at_1MSUN'] + 1.0 ### Given numerical starting frequency. The +1.0 is to avoid input error
    ### import waveform
    hp, hc = get_td_waveform(approximant='NR_hdf5',
                                 numrel_data=filepath,
                                 mass1 = f.attrs['mass1']*total_mass,
                                 mass2 = f.attrs['mass2']*total_mass,
                                 spin1x = f.attrs['spin1x'],
                                 spin1y = f.attrs['spin1y'],
                                 spin1z = f.attrs['spin1z'],
                                 spin2x = f.attrs['spin2x'],
                                 spin2y = f.attrs['spin2y'],
                                 spin2z = f.attrs['spin2z'],
                                 delta_t=delta_t,
                                 f_lower = num_start_freq,
                                 inclination = inclination,
                                 distance = distance,
                                 coa_phase = coa_phase)
    f.close()
    ### cast h as a complex number 
    h = np.array(hp) + np.array(hc)*1j
    ### get sample times in a numpy array
    ht = np.array(hp.sample_times)
    ### get instantaneous frequency series d \phi/ dt
    f_series = m_frequency_from_polarizations(np.real(h),np.imag(h),delta_t)
    ### shift the time series so that it starts at t=0 instead of at a negative number 
    h_tc = ht - ht[0]
    ### selected_starting_frequency is the variable that is going to be itertated to find a starting frequency with reduced junk in the beginning  
    selected_starting_frequency = num_start_freq + f_shift
    t_junk_window = cycle_number/selected_starting_frequency ### approximate time length that includes junk radiation 
    ### get time from the numerical time series that is closest to t_junk_window at which
    time_nearest = find_nearest(h_tc,t_junk_window) 
    ### get frequency at time_nearest
    f_junk_series = f_series[:time_nearest[1]] 
    ### now reverse the window we have chosen to include junk radiation. The reason we do this is so that the largest oscillations due to junk radiation are now on the very end of
    ### the window. 
    rev_f_junk_series = f_junk_series[::-1]
    ### Now here is where the method can break down. We look for the largest junk radiation oscillation and define an "initial region" on the now reveresed frequnecy series. So at 
    ### this point, we have smaller region that only goes up to the index of the largest oscillation. We are now going to increment the selected_starting frequency so that the 
    ### distance between selected_starting frequency and the value in the initial region is minimized.
    max_index = np.argmax(rev_f_junk_series)
    initial_region = rev_f_junk_series[:max_index]
    first_f_start = find_nearest(initial_region,selected_starting_frequency)
    dist_f = abs(first_f_start[0] - selected_starting_frequency)
    ### we say the starting frequency in the good region cannot be farther than an upper_f_bound in Hz set by the user away from the real starting freq. So what this while is doing
    ### is iterating the selected starting frequnecy until it is within some distance away from the theory starting frequency.
    while dist_f > upper_f_bound: 
        f_shift+=1 ## shift it up by one then recalculate dist_f
        selected_starting_frequency = num_start_freq + f_shift
        t_junk_window = cycle_number/selected_starting_frequency
        time_nearest = find_nearest(h_tc,t_junk_window)
        f_junk_series = f_series[:time_nearest[1]]

        rev_f_junk_series = f_junk_series[::-1]
        max_index = np.argmax(rev_f_junk_series)
        ### now recut initial region so that the window doesn't include junk raditation.
        initial_region = rev_f_junk_series[:max_index]
        first_f_start = find_nearest(initial_region,selected_starting_frequency)
        dist_f = abs(first_f_start[0] - selected_starting_frequency)
    ### once the distance is minimized, we now cut off the beginning amount of the simulation set by max_index.
    shift = len(f_junk_series) - first_f_start[1]
    new_time_shift = h_tc[shift:] - h_tc[shift:][0]
    new_h_shift = h[shift:]
    new_f_series = f_series[shift:]
    ### returns the new cut waveform along with the new starting frequency, the time and frequency shifts required to remove junk
    return ((new_time_shift,new_h_shift,new_f_series),(first_f_start[0],f_shift,shift,database_key))

### treat_junk(): uses find_junk() function to cut sets of numerical waveforms so that they all start at the same frequency. Returns the waveforms and metadata in a tuple.
def treat_junk(filepaths,total_mass,inclination,distance,coa_phase,upper_f_bound,f_shift,delta_t=1.0/4096.0,cycle_number=4.0,):
    check = 0
    wavedata = []
    metadata = []
    f_shifts = []
    while check == 0:
        for name in filepaths:
            junk_removed = find_junk(name,total_mass,inclination,distance,coa_phase,upper_f_bound,delta_t,f_shift,cycle_number=4.0)
            wavedata.append(junk_removed[0])
            metadata.append(junk_removed[1])
        for i,val in enumerate(metadata):
            f_shifts.append(metadata[i][1])
        max_f = np.amax(f_shifts)
    
        for i,val in enumerate(f_shifts):
            if val != max_f:
                junk_removed = find_junk(filepaths[i],total_mass,inclination,distance,coa_phase,upper_f_bound,delta_t,f_shift=max_f,cycle_number=4.0)
                wavedata[i] = junk_removed[0]
                metadata[i] = junk_removed[1]
        check += 1
    return wavedata,metadata

### getApprox(): This is just a short hand version of get_td_waveform from pycbc. This function also shifts the time series up so that the waveform starts at t=0.
###              Also has the option to generate waveforms directly from lalsim. To switch between these two methods either pass type="pycbc" or type="laltd" 
def getApprox(type,name,m1=10.0,m2=10.0,f_ref=0.0,f_low=50.0,distance=1,delta_t=1.0/4096.0,s1x=0,s1y=0,s1z=0,s2x=0,s2y=0,s2z=0,inclination=0,tidal1=0,tidal2=0):
    if type == 'pycbc':
        hp, hc = get_td_waveform(approximant = name, mass1=m1,
                                                mass2=m2,
                                                f_lower=f_low,
                                                f_ref = f_ref,
                                                distance= distance,
                                                delta_t= delta_t,
                                                spin1x = s1x,
                                                spin1y = s1y,
                                                spin1z = s1z,
                                                spin2x = s2x,
                                                spin2y = s2y,
                                                spin2z = s2z,
                                                lambda1 = tidal1,
                                                lambda2 = tidal2,
                                                inclination= inclination)

        new_hp = np.array(hp)
        new_hc = np.array(hc)
        h = new_hp + new_hc*1j
        times = np.array(hp.sample_times)
        shift_times = times - times[0]
        return (shift_times,h)
    elif type == 'laltd':
        mpc = 1.0e6*lal.PC_SI
        m1 = m1*lal.MSUN_SI
        m2 = m2*lal.MSUN_SI
        distance = distance*mpc
        tidal_params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(tidal_params,tidal1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(tidal_params,tidal2)
        hp, hc = lalsim.SimInspiralTD(m1, m2,s1x,s1y,s1z,s2x,s2y,s2z,
                                                  distance, inclination, phiRef, 0.,0.,0.,
                                                  delta_t, f_low, f_ref,tidal_params,lalsim.SimInspiralGetApproximantFromString(name))
        times = np.arange(len(hp.data.data))*delta_t + hp.epoch
        shift_times = times - times[0]
        h = np.array(hp.data.data + 1j*hc.data.data)
        return(shift_times,h)
    else:
        print 'Specify pycbc or laltd for approx type'
#### m_phase_from_polarizations() function that unwraps phase from the polarizations
def m_phase_from_polarizations(hp,hc,remove_start_phase=True):
	p_wrapped = np.arctan2(hp,hc)
	p = np.unwrap(p_wrapped)
	if remove_start_phase:
		p += -p[0]	
	return np.abs(p)
#### m_frequency_from_polarizations(): calculates instantaneous frequency d \phi/dt 
def m_frequency_from_polarizations(hp,hc,delta_t):
	phase = m_phase_from_polarizations(hp,hc)
	freq = np.diff(phase)/(2.0*np.pi*delta_t)
	return freq
#### simple match between two waveforms using the convolution theorem.	
def matchfft(h1,h2):
	z_fft = sci.signal.fftconvolve(np.conj(h1),h2[::-1])
	abs_z_fft = np.abs(z_fft)
	w = np.argmax(abs_z_fft) - len(h2) + 1
	delta_w =  w + len(h2)
	h2_norm = np.linalg.norm(h2)
	h1_norm = np.linalg.norm(h1[w:delta_w])
	norm_z = abs_z_fft/(h1_norm*h2_norm)
	return np.amax(norm_z)
#### match_generator_num():  calculates matches between two waveforms with the ability to change the array lengths of h2. h2 is assumed to be the numerical waveform since it is 
#### shorter. This is the main match method to be used with python multiprocessing. 
def match_generator_num(h1,h2,delta_t,ip2,fp2):
	try:	
        	h2_seg = h2[ip2:fp2]
		z_fft = sci.signal.fftconvolve(h1,np.conj(h2_seg[::-1]))
        	abs_z_fft = np.abs(z_fft)
        	w = np.argmax(abs_z_fft) - len(h2_seg) + 1
        	delta_w =  w + len(h2_seg)
                freq = m_frequency_from_polarizations(np.real(h1),np.imag(h1),delta_t=delta_t)
                if  w >= len(freq):
                    h2_norm = np.linalg.norm(h2_seg)
                    h1_norm = np.linalg.norm(h1[w:delta_w])
                    norm_z = abs_z_fft/(h1_norm*h2_norm)
                    return (np.amax(norm_z),0,fp2)
                else:
                    hh_freq = freq[w]
                    h2_norm = np.linalg.norm(h2_seg)
       		    h1_norm = np.linalg.norm(h1[w:delta_w])
		    norm_z = abs_z_fft/(h1_norm*h2_norm)
        	    return (np.amax(norm_z),hh_freq,fp2)
	except RuntimeWarning:
		raise
#### match_generator_PN(): similar to match_generator_num() but allows the user to change array lengths in h1 which is assumed to be an analytic model. This needs to be debugged; 
#### it tends to crash.   
def match_generator_PN(h1,h2,match_i,fp1):
        if len(h1) > len(h2):
		match_f = fp1
                h1_seg = h1[match_i:match_f]
                z_fft = sci.signal.fftconvolve(h1_seg,np.conj(h2[::-1]))
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h1_seg) + 1
                delta_w =  w + len(h1_seg)
                h1_norm = np.linalg.norm(h1_seg)
                h2_norm = np.linalg.norm(h2[w:delta_w])
		norm_z = abs_z_fft/(h1_norm*h2_norm)
                return np.amax(norm_z)
        elif  len(h1) <= len(h2):
                match_f = fp1
                h2_seg = h2[match_i:match_f]
                z_fft = sci.signal.fftconvolve(h2_seg,np.conj(h1[::-1]))
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h1) + 1
                delta_w =  w + len(h1)
                h2_norm = np.linalg.norm(h2_seg)
                h1_norm = np.linalg.norm(h1[w:delta_w])
                norm_z = abs_z_fft/(h1_norm*h2_norm)
                return np.amax(norm_z)
        else:
                return 'error'

def corrintegral(h1,h2,initial,f):
	match_i = initial
        match_f = f
        z = sci.correlate(h1,h2[match_i:match_f],mode='full')
        abs_z = np.abs(z)
        w = np.argmax(abs_z) - len(h2[match_i:match_f]) + 1
        delta_w = w + len(h2[match_i:match_f])
        h2p_norm = np.linalg.norm(h2[match_i:match_f])
	h1p_norm = np.linalg.norm(h1[w:delta_w])
        norm_z = abs_z/(h1p_norm*h2p_norm)
        return  np.amax(norm_z),w
### match function takes in two waveforms with corresponding parameters and can either perform a simple match using function build = 0 (with output 
### (w,delta_w,np.amax(norm_z),phi,h2_phase_shift) where w is the match index, delta_w, the match number, the phase angle, and the corresponding phase shift for h2 respectively) or 
### using build = 1 constructs a full hybrid with windowing length M (an integer). Windowing function used: hann function
### assumes h1 is an approximate and h2 is a numerical simulation. 
def hybridize(h1,h1_ts,h2,h2_ts,match_i,match_f,delta_t=1/4096.0,M=200,info=0):	
    h2_seg = np.array(h2[match_i:match_f]) 
    #### first calculate match between h1 and h2
    z = sci.signal.fftconvolve(h1,np.conj(h2_seg[::-1]))
    abs_z = np.abs(z)
    ### now find index of maximum match. since fftconvolve has the length of len(h1) + len(h2) + 1, we need to subtract off len(h2) since anything above len(h1) correspends to no 
    ### overlap between the two waveforms (i.e h2 has finsihed sliding over h1)
    w = np.argmax(abs_z) - len(h2_seg) + 1
    delta_w = w + len(h2_seg)
    ### now find L2 norms for the match window 
    h2_norm = np.linalg.norm(h2_seg)
    h1_norm = np.linalg.norm(h1[w:delta_w])
    ### calculate normalized match
    norm_z = abs_z/(h1_norm*h2_norm)
    ### calculate the phase that gives the best overlap
    phi = np.angle(z[np.argmax(abs_z)])
    ### now shift h2 so that it is phase aligned with h1
    h2_phase_shift = np.exp(1j*phi)*h2
    ### shift up the waveform in time 
    shift_time  = (w - match_i)*delta_t
    h2_tc = np.array(h2_ts - h2_ts[0] + shift_time)
    freq = m_frequency_from_polarizations(np.real(h1),np.imag(h1),delta_t=delta_t)
    ### hh_frequency is the frequency at the point of hybridization
    hh_freq = freq[w]
    ### define window function lengths
    off_len = (M-1)/2 + 1
    on_len = (M+1)/2
    window = sig.hann(M)
    ##Initialize off and on arrays
    off_hp = np.zeros(off_len)
    on_hp = np.zeros(on_len)
    off_hc = np.zeros(off_len)
    on_hc = np.zeros(on_len)
    ##Bounds for windowing functions
    lb= w
    mid=off_len + w
    ub = M-1 + w
    ##multiply each off and on section by appropriate window section
    for i in range(on_len):
        on_hp[i] = np.real(h2_phase_shift[match_i+i])*window[i]
    for i in range(off_len):
        off_hp[i] = np.real(h1[w+i])*window[i+off_len-1]
    for i in range(on_len):
        on_hc[i] = np.imag(h2_phase_shift[match_i+i])*window[i]
    for i in range(off_len):
        off_hc[i] = np.imag(h1[w+i]*window[i+off_len-1])
    ##Next add the on and off sections together
    mix_hp = on_hp + off_hp
    mix_hc = on_hc + off_hc
    h1_hp_split = np.real(h1[:w])
    h1_hc_split = np.imag(h1[:w])	
    h1_ts_split = h1_ts[:w]
    ### last part concatenates the mixed hp and hc sections with the rest of h1 and h2
    hybrid_t = np.concatenate((np.real(h1_ts_split),np.real(h2_tc[match_i:])), axis =0)
    hybrid_hp = np.concatenate((h1_hp_split,mix_hp,np.real(h2_phase_shift[match_i+off_len:])),axis = 0)
    hybrid_hc = np.concatenate((h1_hc_split,mix_hc,np.imag(h2_phase_shift[match_i+off_len:])),axis =0)
    hybrid = (hybrid_t, hybrid_hp, hybrid_hc)
    ### option to return just a hybrid or the hybrid and the characteristic information 
    if info == 0:
        return hybrid
    if info == 1:
        return(np.max(norm_z),w,phi,h2_phase_shift,h2_tc,hybrid,h2_tc[0],hh_freq)
    else: 
        return 'use info = 0 or 1'




### Old function to reformat SXS waveforms
def getFormatSXSData(filepath,total_m,delta_t=1.0/4096.0,l=2,m=2,N=4):
        num = h5py.File(filepath, 'r')
### We only care about 2-2 mode for now.
        harmonics = 'Y_l%d_m%d.dat' %(l,m)
        order = 'Extrapolated_N%d.dir'%N
        ht = num[order][harmonics][:,0]
        hre = num[order][harmonics][:,1]
        him = num[order][harmonics][:,2]
        ht_SI = tOverM_to_SI(ht,total_m)
        hre_SI = rhOverM_to_SI(hre,total_m)
        him_SI = rhOverM_to_SI(him,total_m)
        sim_name = re.search('/BBH(.+?)/L(.+?)/',filepath).group(0) 
        interpo_hre = sci.interpolate.interp1d(ht_SI,hre_SI, kind = 'linear')
        interpo_him = sci.interpolate.interp1d(ht_SI,him_SI, kind = 'linear')
##### interpolate the numerical hp and hc with PN timeseries
        hts = np.arange(ht_SI[0],ht_SI[-1],delta_t)
        #num_t_zeros = np.concatenate((num_ts,np.zeros(np.absolute(len(PN_tc)-len(num_t)))),axis = 0)
        new_hre = interpo_hre(hts)
        new_him = interpo_him(hts)
        num_wave = new_hre - 1j*new_him
#### Cast waves into complex form and take fft of num_wave
        num.close()
        return (sim_name,hts,num_wave)
def writeHybrid_h5(path_name_part,metadata,approx,sim_name,h1,h1_ts,h2,h2_ts,delta_t,ip2,fp2):
    solar_mass_mpc = lal.MRSUN_SI / (1e6*lal.PC_SI)
    grav_m1 = metadata[0][0]
    grav_m2 = metadata[0][1]
    ADM_m1 = metadata[1][0]
    ADM_m2 = metadata[1][1]
    baryon_m1 = metadata[2][0]
    baryon_m2 = metadata[2][1]
    used_m1 = metadata[3][0]
    used_m2 = metadata[3][1]
    s1x = metadata[4][0]
    s1y = metadata[4][1]
    s1z = metadata[4][2]
    s2x = metadata[4][3]
    s2y = metadata[4][4]
    s2z = metadata[4][5]
    LNhatx = metadata[5][0]
    LNhaty = metadata[5][1]
    LNhatz = metadata[5][2]
    n_hatx = metadata[6][0]
    n_haty = metadata[6][1]
    n_hatz = metadata[6][2]
    tidal_1 = metadata[7][0]
    tidal_2 = metadata[7][1]
    type = metadata[8]
    f_low = metadata[9]
    M = 300
    ### used masses are the masses used in the characterization procedure
    total_mass = used_m1 + used_m2
    hybridPN_Num = hy.hybridize(h1,h1_ts,h2,h2_ts,match_i=0,match_f=fp2,delta_t=delta_t,M=M,info=1)
    shift_time = (hybridPN_Num[6],fp2)
    hh_freq = (hybridPN_Num[7],fp2)
    match = hybridPN_Num[0]
    path_name = path_name_part+'fp2_'+str(fp2)+'.h5'
    f_low_M = f_low * (lal.TWOPI * total_mass * lal.MTSUN_SI)
    with h5py.File(path_name,'w') as fd:
        mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(m1, m2)
        hashtag = hashlib.md5() 
        fd.attrs.create('type', 'Hybrid:%s'%type)
        hashtag.update(fd.attrs['type'])
        fd.attrs.create('hashtag', hashtag.digest())
        fd.attrs.create('hgroup', 'Read_CSUF')
        fd.attrs.create('Format',1)
        fd.attrs.create('Lmax',2)
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
        fd.attrs.create('grav_mass1', grav_m1)
        fd.attrs.create('grav_mass2', grav_m2)
        fd.attrs.create('ADM_mass1', ADM_m1)
        fd.attrs.create('ADM_mass2', ADM_m2)
        fd.attrs.create('baryon_mass1',bary_m1)
        fd.attrs.create('baryon_mass2',bary_m2)
        fd.attrs.create('lambda1',tidal_1)
        fd.attrs.create('lambda2',tidal_2)
        fd.attrs.create('fp1', len(h1))
        fd.attrs.create('ip2',ip2)
        fd.attrs,create('hybrid_match',match)
        fd.attrs.create('shift_time',shift_time)
        fd.attrs.create('hybridize_freq',hh_freq)
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
    return(shift_time,hh_freq)

'''
def iter_hybridize_write(h1,h1_ts,h2,h2_ts,match_i,delta_t,M):
    hybridPN_Num = hybridize(h1,h1_ts,h2,h2_ts,match_i=match_i,match_f=l,delta_t=delta_t,M=300,info=1)
    shift_times.append(hybridPN_Num[6])
    hh_freqs.append(h1_fs[hybridPN_Num[1]])
    h5file= path_name_data+name+'_'+sim_name+'.h5'
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
        HlmAmp = wfutils.amplitude_from_polarizations(hplusMpc,hcrossMpc).data
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

#### writes a given hybrid to a format 1 h5 file and writes it to disk 
def writeHybridtoSplineH5():
    with h5py.File(path_name_data+name+'_'+sim_name,'w') as fd:
                                mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(mass1, mass2)
                                hashtag = hashlib.md5()
                                hashtag.update(fd.attrs['name'])
                                fd.attrs.create('hashtag', hashtag.digest())
                                fd.attrs.create('Read_Group', 'Flynn')
                                fd.attrs.create('name', 'Hybrid:B0:%s'%simname)
                                fd.attrs.create('f_lower_at_1MSUN', f_low_M)
                                fd.attrs.create('eta', eta)
                                fd.attrs.create('Name of Simulation', num_waves[0])
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
                                gramp = fd.create_group('amp_l2_m2')
                                grphase = fd.create_group('phase_l2_m2')
                                times = hybridPN_Num[0]
                                native_delta_t = delta_t
                                hplus = hybridPN_Num[1]
                                hcross = hybridPN_Num[2]
                                massMpc = total_mass*solar_mass_mpc
                                hplusMpc  = pycbc.types.TimeSeries(hplus/massMpc, delta_t=delta_t)
                                hcrossMpc = pycbc.types.TimeSeries(hcross/massMpc, delta_t=delta_t)
                                times_M = times / (lal.MTSUN_SI * total_mass)
                                HlmAmp   = wfutils.amplitude_from_polarizations(hplusMpc,
                                        hcrossMpc).data
                                HlmPhase = wfutils.phase_from_polarizations(hplusMpc, hcrossMpc).data
                        #       if l!=2 or abs(m)!=2:
                        #           HlmAmp = np.zeros(len(HlmAmp))
                        #           HlmPhase = np.zeros(len(HlmPhase))

                                sAmph = romspline.ReducedOrderSpline(times_M, HlmAmp,rel=True ,verbose=False)
                                sPhaseh = romspline.ReducedOrderSpline(times_M, HlmPhase, rel=True,verbose=False)
                                sAmph.write(gramp)
                                sPhaseh.write(grphase)

'''
### finds the L2 distance between 2 waveforms. computes |h1-h2| using the L2 metric
def delta_h(h1,h2):
        h1 = np.array(h1)
        h2 = np.array(h2)
        ### pad either h1 or h2 with zeros so that we can use the subtraction operation
        if len(h1) > len(h2):
                h2 = np.append(h2,np.zeros(np.abs(len(h2)-len(h1))))
        if len(h1) < len(h2):
                h1 = np.append(h1,np.zeros(np.abs(len(h2)-len(h1))))
	norm_h1 = np.linalg.norm(h1)
        norm_h2 = np.linalg.norm(h2)
        h1 = h1/norm_h1
        h2 = h2/norm_h2
        norm_diff = np.linalg.norm(np.subtract(h1,h2)) 
	return norm_diff 		

########################################################################################
## Bandpass filter will only vary the high and low cutoffs at the same time. The filter cutoff
#  currently corresponds to where the middle of the filter is.          

def match_bpfil(h1,h2,order,sample_rate,cutoff,center=250):
        h1p = np.real(h1)
        h2p = np.real(h2)
        nyq = 0.5*sample_rate
        high = (center + cutoff)/nyq
        low = (center - cutoff)/nyq
        b, a = sig.butter(order,[low,high],btype='bandpass', analog=False)
        #w,h = sig.freqz(b,a) 
        h2p_filter = sig.lfilter(b, a, h2p)
        z_fft = sci.signal.fftconvolve(np.conj(h1p),h2p_filter[::-1])
        abs_z_fft = np.abs(z_fft)
        w = np.argmax(abs_z_fft) - len(h2p) + 1
        delta_w =  w + len(h2p)
        h2p_norm = np.linalg.norm(h2p)
        h1p_norm = np.linalg.norm(h1p[w:delta_w])
        norm_z = abs_z_fft/(h1p_norm*h2p_norm)
        return np.amax(norm_z)

def match_lpfil(h1,h2,order,sample_rate,cutoff):
        h1p = np.real(h1)
        h2p = np.real(h2)
        nyq = 0.5*sample_rate
        normal_cutoff = cutoff/nyq
        b,a = sig.butter(order,normal_cutoff,btype='lowpass', analog=False)
        #w,h = sig.freqz(b,a)
        h1p_fil = sig.lfilter(b, a, h1p)
        z_fft = sci.signal.fftconvolve(np.conj(h1p_fil),h2p[::-1])
        abs_z_fft = np.abs(z_fft)
        w = np.argmax(abs_z_fft) - len(h2p) + 1
        delta_w =  w + len(h2p)
        h2p_norm = np.linalg.norm(h2p)
        h1p_norm = np.linalg.norm(h1p[w:delta_w])
        norm_z = abs_z_fft/(h1p_norm*h2p_norm)
        return np.amax(norm_z)
## cond wave argumenet is the wave to be filtered
def match_hpfil(h1,h2,order,sample_rate,cutoff):
        h1p = np.real(h1)
        h2p = np.real(h2)
        nyq = 0.5*sample_rate
        normal_cutoff = cutoff/nyq
        b, a = sig.butter(order,normal_cutoff,btype='highpass', analog=False)
        #w,h = sig.freqz(b,a) 
        if len(h1) >= len(h2):
                h2p_filter = sig.lfilter(b, a, h2p)
                z_fft = sci.signal.fftconvolve(np.conj(h1p),h2p_filter[::-1])
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h2p) + 1
                delta_w =  w + len(h2p)
                h2p_norm = np.linalg.norm(h2p)
                h1p_norm = np.linalg.norm(h1p[w:delta_w])
                norm_z = abs_z_fft/(h1p_norm*h2p_norm)
                return np.amax(norm_z)
        elif len(h1) < len(h2):
                h2p_filter = sig.lfilter(b, a, h2p)
                z_fft = sci.signal.fftconvolve(np.conj(h2p_filter),h1p[::-1])
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h2p) + 1
                delta_w =  w + len(h2p)
                h2p_norm = np.linalg.norm(h2p)
                h1p_norm = np.linalg.norm(h1p[w:delta_w])
                norm_z = abs_z_fft/(h1p_norm*h2p_norm)
                return np.amax(norm_z)
