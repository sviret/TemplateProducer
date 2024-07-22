'''
Small macro to retrieve a template from a bank and compare it to another template

Usage:

python plot_temp.py A B

with:
A: rank of the template in the bank
B: name of the bank file (pickle file)
'''


from numpy import load
import matplotlib.pyplot as plt
import pandas as pd 
import pickle
import sys
import gen_template as gt
import numpy as np
from scipy.spatial import distance
from scipy import ndimage

def readPrecess(fname,rk):    # Parser of the file containing the parameters we want to test

    masses=[]
    spins=[]
    with open(fname,"r") as f:
        for line in f:
            if line.startswith('#'):
                continue
            value=[float(v) for v in line.split(',')]
            if int(value[0])==rk:
                masses.append(value[1:3])
                spins.append(value[3:9])
                break
    return masses,spins

def readBank(fname,rk):    # Parser of the file containing the parameters we want to test

    masses=[]
    spins=[]
    with open(fname,"r") as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('!'):
                continue
            value=[float(v) for v in line.split(' ')]
            if int(value[0])==rk:
                masses.append(value[1:3])
                spins.append(value[3:5])
                break
    return masses,spins


# The main routine

def main():
    import gendata    as gd
    args = sys.argv[1:] # Get input params
    
    #fe_bands=[2048,1024,512,256,128]
    #t_bands=[1.,4.,25.,240.,420.]

    femax=4096.

    fe_bands=[2048,1024,512]
    t_bands=[1.,4.,25.]

    #fe_bands=[1024,512,512]
    #t_bands=[1.,4.,25.]

    fe_bands=[4096]
    t_bands=[30.]

    tlen=np.sum(t_bands)

    _brutePSD=[]

    fmin=15
    fmax=2000

    psd=[]
    freq=[]

    with open("data/aligo_O4high.txt","r") as f:
        for line in f:
            value=[float(v) for v in line.split(' ')]
            if (value[0]<0.5*fmin or value[0]>fmax):
                continue
            psd.append(value[1]**2)
            freq.append(value[0])

    f.close()

    _brutePSD.append([psd,freq])  

    #print('Into realistic loop')
    #print("Retrieved",len(_brutePSD),"different PSDs")

    masses,spins=readPrecess(args[1],int(args[0]))
    masses2,spins2=readBank(args[3],int(args[2]))

    mc1=np.power(masses[0][0]*masses[0][1],0.6)/np.power(masses[0][0]+masses[0][1],0.2)
    mc2=np.power(masses2[0][0]*masses2[0][1],0.6)/np.power(masses2[0][0]+masses2[0][1],0.2)
    q1=masses[0][1]/masses[0][0]
    q2=masses2[0][1]/masses2[0][0]

    chi1=np.sqrt(spins[0][0]**2+spins[0][1]**2+spins[0][2]**2)
    chi2=np.sqrt(spins[0][3]**2+spins[0][4]**2+spins[0][5]**2) 
    stheta1=np.sqrt(spins[0][0]**2+spins[0][1]**2)/chi1
    stheta2=np.sqrt(spins[0][3]**2+spins[0][4]**2)/chi2    


    chi_eff1=(masses[0][0]*chi1+masses[0][1]*chi2)/(masses[0][0]+masses[0][1])
    chi_p1=np.max([chi1*stheta1,(4*q2+3)/(3*q2+4)*q2*chi2*stheta2])
    chi_eff2=(masses2[0][0]*spins2[0][0]+masses2[0][1]*spins2[0][1])/(masses2[0][0]+masses2[0][1])





    TGenerator=gt.GenTemplate(Tsample=t_bands,fe=fe_bands,fDmin=fmin,fDmax=fmax,kindTemplate='IMRPhenomTPHM',whitening=1,customPSD=_brutePSD,PSD='realistic')
    TGenerator.majParams(masses[0][0],masses[0][1],s1x=spins[0][0],s2x=spins[0][3],s1y=spins[0][1],s2y=spins[0][4],s1z=spins[0][2],s2z=spins[0][5])
    tmpl_test,_=TGenerator.getNewSample(kindPSD='realistic',Tsample=TGenerator.duration(),tc=TGenerator.duration(),norm=True)
    TGenerator=gt.GenTemplate(Tsample=t_bands,fe=fe_bands,fDmin=fmin,fDmax=fmax,kindTemplate='EOB',whitening=1,customPSD=_brutePSD,PSD='realistic')
    TGenerator.majParams(masses2[0][0],masses2[0][1],s1z=spins2[0][0],s2z=spins2[0][1],s1x=0,s2x=0,s1y=0,s2y=0)
    tmpl,freq=TGenerator.getNewSample(kindPSD='realistic',Tsample=TGenerator.duration(),tc=TGenerator.duration(),norm=True)
    time=[]
    h_of_t=[]
    h_of_t_test=[]

    #print("Bank template properties:",masses2,spins2)
    #print("Precess template properties:",masses,spins)


    nbands=len(fe_bands)
    iband=nbands-1
    tinit=0.
    starting_idx=0
    starting_idx_test=0
    for fe in reversed(fe_bands):

        ratio=int(femax/fe)
        #ratio=1
        #femax=fe
        strain=tmpl[iband]
        lenchunk=int(femax*t_bands[iband])

        if not isinstance(strain, np.ndarray): # A zero supressed chunk
            strain=np.zeros(lenchunk)
            starting_idx+=lenchunk
        else:
            realstrain=strain
            if ratio>1:
                realstrain = ndimage.zoom(strain,ratio,order=1)

            #print(femax,fe,iband,len(strain),len(realstrain),lenchunk,ratio)

            if len(realstrain)!=lenchunk: # A partially zero suppressed chunk, repopulate it
                patch=np.zeros(lenchunk-len(realstrain))
                realstrain=np.concatenate((patch,realstrain),axis=0)
                starting_idx+=len(patch)

        for dat in realstrain:
            time.append(tinit)
            h_of_t.append(dat)
            tinit+=1./femax

        iband-=1

    iband=nbands-1
    tinit=0.
    for fe in reversed(fe_bands):
        strain_t=tmpl_test[iband]

        ratio=int(femax/fe)
        #ratio=1
        #femax=fe
        lenchunk=int(femax*t_bands[iband])

        if not isinstance(strain_t, np.ndarray): # A zero supressed chunk
            strain_t=np.zeros(lenchunk)
            starting_idx_test+=lenchunk
        else:
            realstrain_t=strain_t
            if ratio>1:
                realstrain_t = ndimage.zoom(strain_t,ratio,order=1)


            #print(femax,fe,iband,len(strain_t),len(realstrain_t),lenchunk,ratio)

            if len(realstrain_t)!=lenchunk: # A partially zero suppressed chunk, repopulate it
                patch=np.zeros(lenchunk-len(realstrain_t))
                realstrain_t=np.concatenate((patch,realstrain_t),axis=0)
                starting_idx_test+=len(patch)
            
        for dat in realstrain_t:
            h_of_t_test.append(dat)
        iband-=1

    print(len(h_of_t),len(h_of_t_test))

    startcomp=starting_idx
    if starting_idx_test>starting_idx:
        startcomp=starting_idx_test

    norm=h_of_t[startcomp:]
    norm_test=h_of_t_test[startcomp:]

    max_h=np.argmax(np.abs(h_of_t))
    max_htest=np.argmax(np.abs(h_of_t_test))

    mult=-1
    if (max_htest<max_h):
        mult=1

    deltat=time[max_htest]-time[max_h]
    print(time[max_htest],time[max_h],deltat)

    time=time[startcomp:]

    #print(starting_idx,starting_idx_test)

    init=(time[1]-time[0])/2
    dt=tlen-time[0]-init

    norm_fft=np.fft.fft(norm,norm='ortho')
    norm_test_fft=np.fft.fft(norm_test,norm='ortho')
    
    mfilter_norm=np.sqrt(len(norm_fft))*np.abs(np.fft.ifft(norm_fft*np.conjugate(norm_test_fft),norm='ortho').real)
    mfilter=np.sqrt(len(norm_fft))*(np.fft.ifft(norm_fft*np.conjugate(norm_test_fft),norm='ortho').real)

    maxn=np.argmax(mfilter_norm)
    max=np.argmax(mfilter)

    
    diffn=(time[maxn]-time[0]-init)


    diff=(time[max]-time[0]-init)

    print(dt/2,time[maxn],diff,diffn)
          
    
    if diffn>dt/2.:
        diffn=tlen-time[maxn]

    if diff>dt/2.:
        diff=tlen-time[max]
    
    #print(max,maxn)
    print('The max MF value is :',np.max([np.max(mfilter_norm),np.max(mfilter)]))

    matched=norm_test
    maxmatching=np.max(mfilter)
    
    if maxn!=max:
        print("Phi_0=Pi/2 case")
        matched=np.multiply(norm_test,-1)
        diff=diffn
        maxmatching=np.max(mfilter_norm)
        
    #print(mult*deltat,mult*diff,mult*diffn)

    tshift=[]
    for t in time:
        tshift.append(t+mult*diff)
        #tshift.append(t-mult*deltat)
    fig, axs = plt.subplots(3)
    #fig.uptitle('Best precession match')
    axs[0].plot(time, norm,'-',label=f'Bank template: $M_c={mc2:.2f}M_\odot$,$q={q2:.2f}$,'+r'$\chi_{eff}'+f'={chi_eff2:.2f}$')
    axs[0].legend(loc="upper left",fontsize='13')
    axs[1].plot(time, norm_test,'-',label=f'Injection: $M_c={mc1:.2f}M_\odot$,$q={q1:.2f}$,'+r'$\chi_{eff}'+f'={chi_eff1:.2f}$,$\chi_p={chi_p1:.2f}$')
    axs[1].legend(loc="upper left",fontsize='13')
    #axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    #axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])

    axs[2].plot(time, norm,'',label=f'Best match = {maxmatching:.3f}')
    axs[2].plot(tshift, matched,'-')
    axs[2].legend(loc="upper left",fontsize='13')
    plt.grid(True, which="both", ls="-")
    plt.show() 

if __name__ == "__main__":
    main()