from functools import partial
import h5py
import cmath
import time
import scipy as sci
import numpy as np
import scipy.signal as sig
import H5Graph
from scipy import interpolate as inter
import multiprocessing as mp
from multiprocessing import Pool
from pycbc.waveform import get_td_waveform
from pycbc.types.timeseries import TimeSeries

sample_rate = 4096*10
delta_t = 1.0/sample_rate

## def _poolinit() is just a profiler. The profiler outputs .out file which shows the time it takes to run functions in the code. Useful 
## for profiling the hybrid characterization script. 
def _poolinit():
	global prof
	prof = cProfile.Profile()
	def finish():
		prof.dump_stats('./Profiles/profile-%s.out' % mp.current_process().pid)
	mp.util.Finalize(None,finish,exitpriority = 1)

def rhOverM_to_SI(polarization,total_mass):
        solar_mass_mpc = 2.0896826e19
        h_conversion = total_mass/solar_mass_mpc
        return polarization*h_conversion

def tOverM_to_SI(times,total_mass):
        t_conversion = total_mass*(4.92686088e-6)
        return times*t_conversion

def getPN(name,m1,m2,f_low,distance,delta_t,sAx,sAy,sAz,sBx,sBy,sBz,inclination):
        Sph_Harm = 0.6307831305
        hp, hc = get_td_waveform(approximant = name, mass1=m1,
                                                mass2=m2,
                                                f_lower=f_low,
                                                distance= distance,
                                                delta_t= delta_t,
                                                spin1x = sAx,
                                                spin1y = sAy,
                                                spin1z = sAz,
                                                spin2x = sBx,
                                                spin2y = sBy,
                                                spin2z = sBz,
                                                lambda1 = 0,
                                                lambda2 = 0,
                                                inclination= inclination)

        new_hp = hp/Sph_Harm
        new_hc = hc/Sph_Harm
        PN_wave = new_hp + new_hc*1j
        times = hp.sample_times
	shift_times = times - times[0]
	return (shift_times,PN_wave)

def m_phase_from_polarizations(hp,hc,remove_start_phase=True):
	p_wrapped = np.arctan2(hp,hc)
	p = np.unwrap(p_wrapped)
	if remove_start_phase:
		p += -p[0]	
	return np.abs(p)

def m_frequency_from_polarizations(hp,hc,delta_t):
	phase = m_phase_from_polarizations(hp,hc)
	freq = np.diff(phase)/(2.0*np.pi*delta_t)
	return freq
	
def matchfft(h1,h2):
	h1p = np.real(h1)
	h2p = np.real(h2)
	#h1c = np.imag(h1)
	#h2c = np.imag(h2)
	z_fft = sci.signal.fftconvolve(np.conj(h1p),h2p[::-1])
	abs_z_fft = np.abs(z_fft)
	w = np.argmax(abs_z_fft) - len(h2p) + 1
	delta_w =  w + len(h2p)
	h2p_norm = np.linalg.norm(h2p)
	h1p_norm = np.linalg.norm(h1p[w:delta_w])
	norm_z = abs_z_fft/(h1p_norm*h2p_norm)
	return np.amax(norm_z)

def match_generator_num(h1,h2,initial,f):
	try:	
		match_i = initial
        	match_f = f
        	h2_seg = h2[match_i:match_f]
		z_fft = sci.signal.fftconvolve(h1,np.conj(h2_seg[::-1]))
        	abs_z_fft = np.abs(z_fft)
        	w = np.argmax(abs_z_fft) - len(h2_seg) + 1
        	delta_w =  w + len(h2_seg)
        	h2_norm = np.linalg.norm(h2_seg)
       		h1_norm = np.linalg.norm(h1[w:delta_w])
		norm_z = abs_z_fft/(h1_norm*h2_norm)
        	return np.amax(norm_z)
	except RuntimeWarning:
		raise
def match_generator_PN(h1,h2,f,delta_t):
        try:
		match_f = f
                h1_seg = h1[match_i:match_f]
                z_fft = sci.signal.fftconvolve(h1,np.conj(h2[::-1]))
                abs_z_fft = np.abs(z_fft)
                w = np.argmax(abs_z_fft) - len(h1_seg) + 1
                delta_w =  w + len(h1_seg)
                h1_norm = np.linalg.norm(h1_seg)
                h2_norm = np.linalg.norm(h2[w:delta_w])
		norm_z = abs_z_fft/(h1_norm*h2_norm)
                return np.amax(norm_z)
        except RuntimeWarning:
                raise
def corrintegral(h1,h2,initial,f):
	match_i = initial
        match_f = f
        z = sci.correlate(h1,h2[match_i:match_f],mode='full')
        abs_z = np.abs(z)
        w = np.argmax(abs_z) - len(h2[match_i:match_f]) + 1
        delta_w = w + len(h2[match_i:match_f])
        h2p_norm = np.linalg.norm(h2[match_i:match_f])
	h1p_norm = np.linalg.norm(h1[w:delta_w])
        norm_z = abs_z/(h1p_norm*h1p_norm)
        return  np.amax(norm_z),w
### match function takes in two waveforms with corresponding parameters and can either perform a simple match using function build = 0 (with output 
### (w,delta_w,np.amax(norm_z),phi,h2_phase_shift) where w is the match index, delta_w, the match number, the phase angle, and the corresponding phase shift for h2 respectively) or 
### using build = 1 constructs a full hybrid with windowing length M (an integer). Windowing function used: hann function
def hybridize(h1,h2,h1_ts,h2_ts,match_i,match_f,M=200):
		z = sci.signal.fftconvolve(h1,np.conj(h2[::-1]))
		abs_z = np.abs(z)
		w = np.argmax(abs_z) - len(h2[match_i:match_f])
		delta_w = w + len(h2[match_i:match_f])
		h2_norm = np.linalg.norm(h2[match_i:match_f])
		h1_norm = np.linalg.norm(h1[w:delta_w])
		norm_z = abs_z/(h1_norm*h2_norm)
		phi = np.angle(z[np.argmax(abs_z)])
		h2_phase_shift = np.exp(1j*phi)*h2
		h2_tc = h2_ts - h2_ts[0] + (w - match_i)*delta_t
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
		hybrid_t = np.concatenate((np.real(h1_ts_split),np.real(h2_tc[match_i:])), axis =0)
		hybrid_hp = np.concatenate((h1_hp_split,mix_hp,np.real(h2_phase_shift[match_i+off_len:])),axis = 0)
		hybrid_hc = np.concatenate((h1_hc_split,mix_hc,np.imag(h2_phase_shift[match_i+off_len:])),axis =0)
		hybrid = (hybrid_t, hybrid_hp, hybrid_hc)
		#return(np.max(norm_z),phi,h2_phase_shift,hybrid)
		return hybrid 

def SItoNinjaTime(x):
### time conversion (for now): a*M * G/c^3 where M is in solar masses where a is some integer
	t_conversion = total_mass*(4.92686088e-6)
	return x/t_conversion
def SItoNinjah(h):
### amp conversion: 1M = (1 megaparsec) * c^3/G in solar masses (r is in terms of solar masses)
###       : 1mpc = 2.086e19
### the h_conversion is of the form M_tot/1mpc
	solar_mass_mpc = 2.0896826e19
	h_conversion = total_mass/solar_mass_mpc
	return h/h_conversion	
##########################################################################################
### UNDER CONSTRUCTION: calculates distance between two arbitrary waveforms relative to h1 
def delta_h(h1,h2):
        if len(h1) > len(h2):
                h2 = np.append(h2,np.zeros(np.abs(len(h2)-len(h1))))
        if len(h1) < len(h2):
                h1 = np.append(h1,np.zeros(np.abs(len(h2)-len(h1))))
	norm_diff = np.divide(np.linalg.norm(np.subtract(h1,h2)),np.linalg.norm(h2))
	return norm_diff 
#def delta_phi():
##########################################################################################
## Bandpass filter will only vary high and low cutoffs at the same time. The filter cutoff
#  currently corresponds to where the middle of the filter is.          
def match_bpfil(h1,h2,order,sample_rate,cutoff):
        h1p = np.real(h1)
        h2p = np.real(h2)
        nyq = 0.5*sample_rate
        center = 250
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
