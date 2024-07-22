'''
Small macro to retrieve a template from a bank and compare it to another template

Usage:

python plot_temp.py A B

with:
A: rank of the template in the bank
B: name of the bank file (pickle file)
'''


from scipy.stats import pearsonr
from numpy import load
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
import sys
import ROOT as root
from time import time
import gen_template as gt
import numpy as np
from scipy.spatial import distance
from scipy.signal import correlate
from array import array
from scipy import ndimage

def readFilePrecess(fname):    # Parser of the file containing the parameters we want to test

    nbcoals=[]
    masses=[]
    spins=[]
    with open(fname,"r") as f:
        for line in f:
            if line.startswith('#'):
                continue
            value=[float(v) for v in line.split(',')]
            
            nbcoals.append(value[0])
            masses.append(value[1:3])
            spins.append(value[3:9])
    return nbcoals,masses,spins

# The main routine

def main():
    import gendata as gd
    args = sys.argv[1:] # Get input params
    
    # Retrieve the bank (do this once)
    precess_data=args[3]
    bankkname=args[2]
    selected_idx_i=int(args[0])
    selected_idx_e=int(args[1])
    suffix=args[4]

    cache_tmpl=[]
    cache_start=[]
    cache_p_tmpl=[]
    cache_p_start=[]
    
    start_bk=float((bankkname.split('from')[1]).split('-')[1])
    end_bk=float((bankkname.split('from')[1]).split('-')[3])

    #print(start_bk,end_bk)

    data=gd.GenDataSet.readGenerator(bankkname)


    Mc_b      = array('d', [0])
    Mc_p      = array('d', [0])
    q_b       = array('d', [0])
    q_p       = array('d', [0])
    chieff_b  = array('d', [0])
    chieff_p  = array('d', [0])
    chip_p    = array('d', [0])
    tani      = array('d', [0])
    mf        = array('d', [0])
    cross     = array('d', [0])
    rk_b      = array('d', [0])
    rk_p      = array('d', [0])

    file  = root.TFile.Open(f'result_prec_{suffix}_{selected_idx_i}_{selected_idx_e}_bk_{start_bk}_{end_bk}.root', 'recreate')

    tree3 = root.TTree("comp", "comp")
    tree3.Branch("mchirp_b",     Mc_b,     'Mc_b/D')
    tree3.Branch("mchirp_p",     Mc_p,     'Mc_p/D')
    tree3.Branch("q_b",           q_b,     'q_b/D')
    tree3.Branch("q_p",           q_p,     'q_p/D')
    tree3.Branch("chieff_b", chieff_b,     'chieff_b/D')
    tree3.Branch("chieff_p", chieff_p,     'chieff_p/D')
    tree3.Branch("chip_p",     chip_p,     'chip_p/D')
    tree3.Branch("tani", tani,     'tani/D')
    tree3.Branch("mf", mf,     'mf/D')
    tree3.Branch("cross", cross,     'cross/D')
    tree3.Branch("rk_b",           rk_b,     'rk_b/D')
    tree3.Branch("rk_p",           rk_p,     'rk_p/D')

    TaniSim=[]
    CosinSim=[]
    TanS_save=[]
    CosS_save=[]
    rank_save=0.
    norm_max=[]
    time_max=[]
    norm_test_max=[]
    Results=[]

    
    nbcoals, masses, spins = readFilePrecess(precess_data)
    
    NbPrecess=range(len(nbcoals))
    NbBank=range(data.Ntemplate())

    selected_idx_e=np.min([selected_idx_e,len(nbcoals)])
    #print(selected_idx_i,selected_idx_e)
    bkinfo=data.getParams(rank=0)
    fe_bands=bkinfo[0]
    t_bands=bkinfo[1]
    femax=np.max(fe_bands)

    _brutePSD=[]

    fmin=15
    fmax=1000

    psd=[]
    freq=[]

    with open("data/aligo_O4high.txt","r") as f:
        for line in f:
            value=[float(v) for v in line.split(' ')]
            if (value[0]<fmin or value[0]>fmax):
                continue
            psd.append(value[1]**2)
            freq.append(value[0])

    f.close()

    _brutePSD.append([psd,freq])  

    print('Into realistic loop')
    print("Retrieved",len(_brutePSD),"different PSDs")

    TGenerator=gt.GenTemplate(Tsample=bkinfo[1],fe=bkinfo[0],kindTemplate='IMRPhenomTPHM',whitening=1,PSD='realistic',customPSD=_brutePSD)

    for prec in range(selected_idx_i,selected_idx_e):

        if prec%10==0:
            print("Caching template ",prec)

        sel_mass=masses[prec]
        sel_spins=spins[prec]



        TGenerator.majParams(sel_mass[0],sel_mass[1],s1x=sel_spins[0],s1y=sel_spins[1],s1z=sel_spins[2],s2x=sel_spins[3],s2y=sel_spins[4],s2z=sel_spins[5],Phic=0)
        tmpl_test,freqs_test=TGenerator.getNewSample(kindPSD='analytic',Tsample=TGenerator.duration(),tc=TGenerator.duration(),norm=True)

        # Put the template with precession info in same vectors than the bank one 
        h_of_t_test=[]

        nbands=len(fe_bands)        
        starting_idx_test=0
        iband=nbands-1

        for fe in reversed(fe_bands):
            ratio=int(femax/fe)
            strain_t=tmpl_test[iband]
            lenchunk=int(femax*t_bands[iband])

            if not isinstance(strain_t, np.ndarray): # A zero supressed chunk
                starting_idx_test+=lenchunk
                iband-=1
                continue

            if len(strain_t)!=lenchunk: # A partially zero suppressed chunk, repopulate it
                #patch=np.zeros(lenchunk-len(strain_t))
                starting_idx_test+=lenchunk-len(strain_t)*ratio

            if ratio>1:
                strain_t = ndimage.zoom(strain_t,ratio,order=1)

            for dat in strain_t:
                h_of_t_test.append(dat)
            iband-=1

        cache_p_tmpl.append(h_of_t_test)
        cache_p_start.append(starting_idx_test)

    # Caching bank data
    # 
    total=data.Ntemplate()
    npc=500
    nchunck = int(total/npc)+1

    #total=100
    #nchunck=3



    for step in range(nchunck):
    
        start=npc*step
        stop=np.min([npc*(step+1),data.Ntemplate()])
        cache_tmpl=[]
        cache_start=[]

        print("Analyzing chunk",step,"/",nchunck)
        print("STEP 1: Put the template bank temporal data into cache...") 

        for NbB in range(start,stop):
    
            if NbB%100==0:
                print(NbB,"/",stop)
            tmpl,_,_=data.getTemplate(rank=NbB)

            h_of_t=[]  # Strain at the corresp. time

            nbands=len(fe_bands)
            iband=nbands-1
            starting_idx=0

            for fe in reversed(fe_bands): # Loop over freq bands

                ratio=int(femax/fe)
                strain=tmpl[iband]
                lenchunk=int(femax*t_bands[iband])

                if not isinstance(strain, np.ndarray): # A zero supressed chunk
                    #strain=np.zeros(lenchunk)
                    starting_idx+=lenchunk
                    iband-=1
                    continue
            
                if len(strain)!=lenchunk: # A partially zero suppressed chunk, repopulate it
                    #patch=np.zeros(lenchunk-len(strain))
                    #strain=np.concatenate((patch,strain),axis=0)
                    starting_idx+=lenchunk-len(strain)*ratio

                if ratio>1:
                    strain = ndimage.zoom(strain,ratio,order=1)

                for dat in strain:
                    h_of_t.append(dat)
                iband-=1

            cache_tmpl.append(h_of_t)
            cache_start.append(starting_idx)


        print("STEP 2: Compare to the precessed template temporal data into cache...") 
        # Create the template with precession (do it once)

        for prec in range(selected_idx_i,selected_idx_e):

            if prec%10==0:
                print("Testing template ",prec)

            sel_mass=masses[prec]
            sel_spins=spins[prec]

            max_tmp=-1
            max_mf=-1

            sel_chirp=np.power(sel_mass[0]*sel_mass[1],0.6)/np.power(sel_mass[0]+sel_mass[1],0.2)
            sel_ratio=sel_mass[1]/sel_mass[0]

            chi1=np.sqrt(sel_spins[0]**2+sel_spins[1]**2+sel_spins[2]**2)
            chi2=np.sqrt(sel_spins[3]**2+sel_spins[4]**2+sel_spins[5]**2) 
            stheta1=np.sqrt(sel_spins[0]**2+sel_spins[1]**2)/chi1
            stheta2=np.sqrt(sel_spins[3]**2+sel_spins[4]**2)/chi2    
            m1=sel_mass[0]
            m2=sel_mass[1]
            q=sel_ratio

            chi_eff=(m1*chi1+m2*chi2)/(m1+m2)
            chi_p=np.max([chi1*stheta1,(4*q+3)/(3*q+4)*q*chi2*stheta2])

            h_of_t_test=cache_p_tmpl[prec-selected_idx_i]
            starting_idx_test=cache_p_start[prec-selected_idx_i]

            # Retrieve the templates and make the comparisons
            for NbB in range(start,stop):
                #if NbB%100==0:
                #    print("Testing template ",prec,NbB)

                # Retrieve just the template parameter (no decompression at this stage)
                #print(NbB)
                bkinfo=data.getParams(rank=NbB)
                m1=bkinfo[2][0]
                m2=bkinfo[2][1]
                mchirp=np.power(m1*m2,0.6)/np.power(m1+m2,0.2)
                mratio=m2/m1

                chirp_r=mchirp/sel_chirp

                if (chirp_r>3 or chirp_r<0.5) and mratio>0.3:
                    continue

                if (chirp_r>2 or chirp_r<0.1) and mratio<0.3:
                    continue

                Mc_b[0]=mchirp 
                Mc_p[0]=sel_chirp
                q_b[0]=mratio   
                q_p[0]=sel_ratio   
                chieff_b[0]=(m1*bkinfo[3][0]+m2*bkinfo[3][1])/(m1+m2)
                chieff_p[0]=chi_eff
                chip_p[0]=chi_p

                rk_b[0] = NbB+start_bk
                rk_p[0] = prec

                h_of_t=cache_tmpl[NbB%npc]
                starting_idx=cache_start[NbB%npc]

                # Where to start the comparison

                startcomp=starting_idx-starting_idx_test

                if startcomp<0: # Prec frame shorter than bank one
                    norm=h_of_t[-startcomp:]
                    norm_test=h_of_t_test
                else:
                    norm=h_of_t
                    norm_test=h_of_t_test[startcomp:]    

                
                overlap=correlate(norm, norm_test, mode='full')
                
                crossc=overlap[np.argmax(overlap)]
                
                if crossc<0.15:
                    continue

                norm_fft=np.fft.fft(norm,norm='ortho')
                norm_test_fft=np.fft.fft(norm_test,norm='ortho')
                mfilter=np.sqrt(2*len(norm_fft))*np.abs(np.fft.ifft(norm_fft*np.conjugate(norm_test_fft),norm='ortho'))
                
                mfi=np.max(mfilter)
               
                mf[0]        =  mfi
                cross[0]     =  crossc
                tree3.Fill()

    file.Write()
    file.Close()
    #f.close()



if __name__ == "__main__":
    main()