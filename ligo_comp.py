    
tlen = max(len(hplus), len(hplus1))
hplus.resize(tlen)
hplus1.resize(tlen)
delta_f = 1.0 / hplus1.duration
flen = tlen/2 + 1

psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)

x,y,norm11 = matched_filter_core(hplus,hplus,psd,low_frequency_cutoff=f_low)

m11, i = match(hplus, hplus, psd=psd, low_frequency_cutoff=f_low)

sig11 = sigmasq(hplus,psd,low_frequency_cutoff=f_low)

x,y,norm22 = matched_filter_core(hplus1,hplus1,psd,low_frequency_cutoff=f_low)

m22, i = match(hplus1, hplus1, psd=psd, low_frequency_cutoff=f_low)

sig22 = sigmasq(hplus1,psd,low_frequency_cutoff=f_low)

x,y,norm12 = matched_filter_core(hplus,hplus1,psd,low_frequency_cutoff=f_low)

m12, i = match(hplus, hplus1, psd=psd, low_frequency_cutoff=f_low)

sig12 =sigmasq(hplus1,psd,low_frequency_cutoff=f_low)

disting = (m11*sig11**(.5)/norm11+m22*sig22**(.5)/norm22-2*m12*sig12**(.5)/norm12)**.5

print disting

