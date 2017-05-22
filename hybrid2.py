import scipy as sci
import numpy as np
import scipy.signal as sig
from  pylab import plot,show
import matplotlib.pyplot as plt
import H5Graph 
from scipy import interpolate as inter
from pycbc.waveform import get_td_waveform
from hybridmodules import *
## This script constructs a simple hybrid model using cross correlation.


####### Define Numerical Wave Specs
m_1 = 7
m_2 = 1.4
q = 5
#observer distance
distance = 1
total_mass = m_1 + m_2
### this is the 2-2 spin weighted spherical harmonic
Sph_Harm = 0.6307831305
### amp conversion: 1M = (1 megaparsec) * c^3/G in solar masses (r is in terms of solar masses)
### 		  : 1mpc = 2.086e19
#### the h_conversion is of the form M_tot/1mpc 
solar_mass_mpc = 2.0896826e19
h_conversion = total_mass/solar_mass_mpc
t_conversion = total_mass*(4.92686088e-6)
####### Define Constants
f_low = 50
sample_rate = 4096*10
delta_t = 1.0/sample_rate

############## Find a correlation between the two waveforms
### typically len(h2) < len(h1)
### This function does NOT interpolate anything. 
### h1 and h2 must be tuples in the form (time,hp+jhc)
### i and f must be integer
### build full hybrid? must be 0 or 1 (no or yes) (outputs a tuple of the form (hybrid(time,hp,hc),match number, phase shift angle) 
### M is an integer for the window function length 
def match(h1,h2,i,f,build,M):
	if build == 0:
		z = sig.correlate(h1[1],h2[1][i:f],mode='full')
		abs_z = np.abs(z)
		w = np.argmax(abs_z)  - len(h2[1][i:f]) + 1
		delta_w = w + len(h2[1][i:f])
		h2_norm = np.linalg.norm(h2[1][i:f])
		h1_norm = np.linalg.norm(h1[1][w:delta_w])
		norm_z = abs_z/(h1_norm*h2_norm)
		phi = np.angle(z[np.argmax(abs_z)]) 
		h2_phase_shift = np.exp(1j*phi)*h2[1]
		return(w,delta_w,np.amax(norm_z),phi,h2_phase_shift)
#### bulid function needs a time series input. none right now.
	if build == 1:
		z = sig.correlate(h1[1],h2[1][i:f],mode='full')
                abs_z = np.abs(z)
                w = np.argmax(abs_z) - len(h2[1][i:f]) + 1
                delta_w = w + len(h2[1][i:f])
                h2_norm = np.linalg.norm(h2[1][i:f])
                h1_norm = np.linalg.norm(h1[1][w:delta_w])
                norm_z = abs_z/(h1_norm*h2_norm)
                phi = np.angle(z[np.argmax(abs_z)])
                h2_phase_shift = np.exp(1j*phi)*h2[1]
		h2_tc = h2[0] - h2[0][0] + (w - match_i)*delta_t	
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
                	off_hp[i] = np.real(h1[1][w+i])*window[i+off_len-1]

        	for i in range(on_len):
                	on_hc[i] = np.imag(h2_phase_shift[match_i+i])*window[i]

        	for i in range(off_len):
                	off_hc[i] = np.imag(h1[1][w+i])*window[i+off_len-1]
        	##Next add the on and off sections together
        	mix_hp = on_hp + off_hp
        	mix_hc = on_hc + off_hc 
		h1_hp_split = np.real(h1[1][:w])
        	h1_hc_split = np.imag(h1[1][:w])
        	h1_tc_split = h1[0][:w]
        	hybrid_t = np.concatenate((h1_tc_split,h2_tc[match_i:]), axis =0)
        	hybrid_hp = np.concatenate((h1_hp_split,mix_hp,h2_phase_shift[match_i+off_len:]),axis = 0)
        	hybrid_hc = np.concatenate((h1_hc_split,mix_hc,h2_phase_shift[match_i+off_len:]),axis =0)
                hybrid = (hybrid_t, hybrid_hp, hybrid_hc)
		return(hybrid,np.max(norm_z),phi,h2_tc,h2_phase_shift)
	else:
		return 'Error' 
#### Below is a test application of the functions above

#### Get numerical waveforms using Evan's script h5graph
data_raw = H5Graph.importH5Data(['InputNumericalDataPathHere/rhOverM_Asymptotic_GeometricUnits.h5/Extrapolated_N2.dir/Y_l2_m2.dat'])
data = H5Graph.formatData(data_raw)
num_hp = data[0][0][:,1][100:]*h_conversion
num_hc = data[0][0][:,2][100:]*h_conversion
num_t = data[0][0][:,0][100:]*t_conversion
##### Define interpolation functions

interpo_hp = sci.interpolate.interp1d(num_t,num_hp, kind = 'linear')
interpo_hc = sci.interpolate.interp1d(num_t,num_hc, kind = 'linear')

##### interpolate the numerical hp and hc with PN timeseries

num_ts = np.arange(num_t[0],num_t[-1],delta_t)
new_num_hp = interpo_hp(num_ts)
new_num_hc = interpo_hc(num_ts)
num_complex = new_num_hp - new_num_hc*1j
#######Now put the waves into complex form
##### Note: Depending on the spherical harmonic, m=-2 and m=2 for example are complex conjugates.
##So when constructing the complex form the m=-2 mode requires a + and m=2 requires a -
'''
num_wave = (num_ts,num_complex)
PN_wave = getPN("TaylorT4",m_1,m_2,f_low,distance=1,delta_t=delta_t,sAx=0.0,sAy=0.0,sAz=0.0,sBx=0.0,sBy=0.0,sBz=0.0,inclination=0.0)
plt.plot(PN_wave[0],np.real(PN_wave[1]))
plt.plot(num_wave[0],np.real(num_wave[1]))
plt.show()

###Select match region
match_i = np.floor((num_ts[0])/delta_t)
match_f = np.floor((.3 - num_ts[0])/delta_t)

hybrid = match(PN_wave,num_wave,match_i,match_f,1,200)
print 'Phase: ', hybrid[2] 
print 'Max Normed Match ', hybrid[1]

#### Wave-plots
## hp
plt.plot(PN_wave[0],PN_wave[1],label='SEOBNRv2')
plt.plot(hybrid[3],np.real(hybrid[4]),label='Numerical Wave')
#plt.plot(PN_wave[0][match[0]:match[1]],PN_wave[1][match[0]:match[1]],label='Overlap Area')
plt.legend()
plt.xlabel('Time (Seconds)')
plt.ylabel('h (Strain)')
plt.show()

## hc 
pylab.plot(PN_wave[0],PN_wave[2])
pylab.plot(num_tc,np.imag(hybrid_1[3]))
pylab.xlabel('Time (Seconds)')
pylab.ylabel('h (Strain)')
pylab.show()

## hybrid plot
plt.plot(hybrid[0][0],hybrid[0][1])
plt.show()
'''
