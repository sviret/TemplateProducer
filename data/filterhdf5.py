import numpy as np
import h5py
import sys

from array import array

args = sys.argv[1:] # Get input params

hdffile=args[0]
chip_cut=float(args[1])
q_cut=float(args[2])

f=h5py.File(hdffile,"r")
a_group_key = list(f.keys())[0]
ds_obj = f[a_group_key]
data_fields=list(ds_obj.dtype.fields.keys())


lumiD    = array('d', [0])
M1     = array('d', [0])
M2     = array('d', [0])
M1s    = array('d', [0])
M2s    = array('d', [0])
s1x    = array('d', [0])
s1y    = array('d', [0])
s1z    = array('d', [0])
s2x    = array('d', [0])
s2y    = array('d', [0])
s2z    = array('d', [0])
chieff = array('d', [0])
chip   = array('d', [0])
t_coal = array('d', [0])
t_coalH = array('d', [0])
t_coalL = array('d', [0])
SNR_H  = array('d', [0])
SNR_L  = array('d', [0])
coaphase  = array('d', [0])

f = open(f"filter_chip{chip_cut}_q{q_cut}.txt", "w")

compt=0
fieldo={}
for field in data_fields:
    fieldo.update({field: compt})
    compt+=1

compt=0
sel=0
f.write('#!id,m1,m2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,chieff,chip,tcoal,SNRH,SNRL\n')
for inj in ds_obj:

    if compt%10000==0:
        print(compt)

    t_coalH[0]=inj[fieldo['time_H']]
    t_coalL[0]=inj[fieldo['time_L']]
    t_coal[0]=inj[fieldo['time_geocenter']]
    M1[0]=inj[fieldo['mass1_detector']]
    M2[0]=inj[fieldo['mass2_detector']]
    M1s[0]=inj[fieldo['mass1_source']]
    M2s[0]=inj[fieldo['mass2_source']]
    SNR_L[0]=inj[fieldo['observed_snr_L']]
    SNR_H[0]=inj[fieldo['observed_snr_H']]
    s1x[0]=inj[fieldo['spin1x']]
    s1y[0]=inj[fieldo['spin1y']]
    s1z[0]=inj[fieldo['spin1z']]
    s2x[0]=inj[fieldo['spin2x']]
    s2y[0]=inj[fieldo['spin2y']]
    s2z[0]=inj[fieldo['spin2z']]
    chieff[0]=inj[fieldo['chi_eff']]
    chip[0]=inj[fieldo['chi_p']]
    lumiD[0]=inj[fieldo['luminosity_distance']]
    coaphase[0]=inj[fieldo['coa_phase']]
    if SNR_L[0]>10000000000:
        print(SNR_L[0],t_coal[0]) 

    
    if (chip[0]>chip_cut and M2[0]/M1[0]<q_cut):
        f.write(f'{sel},{M1s[0]},{M2s[0]},{s1x[0]},{s1y[0]},{s1z[0]},{s2x[0]},{s2y[0]},{s2z[0]},{chieff[0]},{chip[0]},{t_coal[0]},{SNR_L[0]},{SNR_L[0]}\n')
        sel+=1

    compt+=1
f.close()
