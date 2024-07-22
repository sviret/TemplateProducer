import numpy as np
import h5py
import ROOT as root
from array import array

f=h5py.File("samples-rpo4a_v1-1366933504-23846400.hdf","r")
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

f = open("selected.txt", "w")
#f.write("Woops! I have deleted the content!")

file  = root.TFile.Open('test.root', 'recreate')

tree3 = root.TTree("injections", "injections")
tree3.Branch("mass1",     M1,     'M1/D')
tree3.Branch("mass2",     M2,     'M2/D')
tree3.Branch("mass1_s",   M1s,    'M1s/D')
tree3.Branch("mass2_s",   M2s,    'M2s/D')
tree3.Branch("tcoalH",    t_coalH,'t_coalH/D')
tree3.Branch("tcoalL",    t_coalL,'t_coalL/D')
tree3.Branch("tcoal",     t_coal, 't_coal/D')
tree3.Branch("SNR_H",     SNR_H,  'SNR_H/D')
tree3.Branch("SNR_L",     SNR_L,  'SNR_L/D')
tree3.Branch("s1x",       s1x,    's1x/D')
tree3.Branch("s1y",       s1y,    's1y/D')
tree3.Branch("s1z",       s1z,    's1z/D')
tree3.Branch("s2x",       s2x,    's2x/D')
tree3.Branch("s2y",       s2y,    's2y/D')
tree3.Branch("s2z",       s2z,    's2z/D')
tree3.Branch("chi_eff",   chieff, 'chieff/D')
tree3.Branch("chi_p",     chip,   'chip/D')
tree3.Branch("Deff",      lumiD,  'lumiD/D')
tree3.Branch("Theta0",    coaphase,  'coaphase/D')

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
    #if compt==100000:
    #    break

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

    
    if (SNR_L[0]>6 and SNR_H[0]>6 and chip[0]>0.5 and M1[0]/M2[0]>3):
        f.write(f'{sel},{M1s[0]},{M2s[0]},{s1x[0]},{s1y[0]},{s1z[0]},{s2x[0]},{s2y[0]},{s2z[0]},{chieff[0]},{chip[0]},{t_coal[0]},{SNR_L[0]},{SNR_L[0]}\n')
        sel+=1

    tree3.Fill()
    compt+=1
f.close()
file.Write()
file.Close()