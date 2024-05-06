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

# Fractal Tanimoto similarity with weight
def f_tanimoto_w(x,y,w,d=0):
    return np.dot(x,w*y)/((2**d)*(np.linalg.norm(np.sqrt(w)*x)**2+np.linalg.norm(np.sqrt(w)*y)**2)-(2**(d+1)-1)*np.dot(x,w*y))

# Without weight
def f_tanimoto(x,y,d=0):
    return np.dot(x,y)/((2**d)*(np.linalg.norm(x)**2+np.linalg.norm(y)**2)-(2**(d+1)-1)*np.dot(x,y))

# Average fractal Tanimoto similarity with/without weight
def ave_ft_w(x,y,w,d=0):
    if d==0:
        d=1
    result=0.
    for i in range(d):
        result+=f_tanimoto_w(x,y,w,i)
    return result/d

def ave_ft(x,y,d=0):
    if d==0:
        d=1
    result=0.
    for i in range(d):
        result+=f_tanimoto(x,y,i)
    return result/d


# The main routine

def main():
    import gendata    as gd
    args = sys.argv[1:] # Get input params
    
    # Retrieve the bank (do this once)
    data=gd.GenDataSet.readGenerator(args[1])
    
    # Retrieve the template
    # Here we get one but one could possibly get more
    tmpl,freqs,bkinfo=data.getTemplate(rank=int(args[0]))

    # We retrieves the parameters with which the bank was produced
    # need this to produce the template we want to compare with the right bands
    
    masses=bkinfo[2]
    spins=bkinfo[3]
    
    fe_bands=bkinfo[0]
    t_bands=bkinfo[1]

    TGenerator=gt.GenTemplate(Tsample=bkinfo[1],fe=bkinfo[0],kindTemplate='IMRPhenomTPHM',whitening=0)
    TGenerator.majParams(masses[0],masses[1],s1=spins[0],s2=spins[1])
    tmpl_test,freqs_test=TGenerator.getNewSample(kindPSD='analytic',Tsample=TGenerator.duration(),tc=TGenerator.duration(),norm=False)

    time=[]
    f_of_t=[]    
    h_of_t=[]

    f_of_t_test=[]    
    h_of_t_test=[]

    nbands=len(fe_bands)
    iband=nbands-1
    tinit=0.
    starting_idx=0
    starting_idx_test=0
    for fe in reversed(fe_bands):
        strain=tmpl[iband]
        freq=freqs[iband]

        lenchunk=int(fe*t_bands[iband])

        if not isinstance(strain, np.ndarray): # A zero supressed chunk
            strain=np.zeros(lenchunk)
            freq=np.zeros(lenchunk)
            starting_idx+=lenchunk

        if len(strain)!=lenchunk: # A partially zero suppressed chunk, repopulate it
            patch=np.zeros(lenchunk-len(strain))
            strain=np.concatenate((patch,strain),axis=0)
            freq=np.concatenate((patch,freq),axis=0)
            starting_idx+=len(patch)

        #print(fe,len(strain))

        for dat in strain:
            time.append(tinit)
            h_of_t.append(dat)
            tinit+=1./fe
        for f in freq:
            f_of_t.append(f)
        iband-=1

    iband=nbands-1
    tinit=0.
    for fe in reversed(fe_bands):
        strain_t=tmpl_test[iband]
        freq_t=freqs_test[iband]

        lenchunk=int(fe*t_bands[iband])

        if not isinstance(strain_t, np.ndarray): # A zero supressed chunk
            strain_t=np.zeros(lenchunk)
            freq_t=np.zeros(lenchunk)
            starting_idx_test+=lenchunk

        if len(strain_t)!=lenchunk: # A partially zero suppressed chunk, repopulate it
            patch=np.zeros(lenchunk-len(strain_t))
            strain_t=np.concatenate((patch,strain_t),axis=0)
            freq_t=np.concatenate((patch,freq_t),axis=0)
            starting_idx_test+=len(patch)

        for dat in strain_t:
            h_of_t_test.append(dat)
        for f in freq_t:
            f_of_t_test.append(f)
        iband-=1

    #print(starting_idx,starting_idx_test)

    startcomp=starting_idx
    if starting_idx_test>starting_idx:
        startcomp=starting_idx_test

    max=np.max(np.abs(h_of_t))
    max_test=np.max(np.abs(h_of_t_test))

    norm=h_of_t[startcomp:]/max
    norm_test=h_of_t_test[startcomp:]/max_test
    time=time[startcomp:]

    print("Tanimoto similarity:",ave_ft(norm,norm_test,d=0))
    print("Cosine similarity:",1-distance.cosine(norm,norm_test))

    plt.plot(time, norm,'-',label='template')
    plt.plot(time, norm_test,'-',label='test')
    #plt.plot(time, norm-norm_test,'-',label='test')
    #plt.title('Template fi dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
    plt.xlabel('t (s)')
    plt.ylabel('h(t) (No Unit)')
    plt.grid(True, which="both", ls="-")
    plt.show() 
    
    plt.plot(time, f_of_t[startcomp:],'-',label='template') 
    plt.plot(time, f_of_t_test[startcomp:],'-',label='template')
    #plt.title('Template fi dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
    plt.xlabel('t (s)')
    plt.ylabel('f (in Hz)')
    plt.grid(True, which="both", ls="-")
    plt.show() 
    
if __name__ == "__main__":
    main()