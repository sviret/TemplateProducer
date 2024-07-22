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
    fname=args[0]

    file = root.TFile.Open(fname,"READ")

    tree3 = file.Get("comp")

    Mc_b      = array('d', [0])
    Mc_p      = array('d', [0])
    q_b       = array('d', [0])
    q_p       = array('d', [0])
    chieff_b  = array('d', [0])
    chieff_p  = array('d', [0])
    chip_p    = array('d', [0])
    mf        = array('d', [0])
    rk_b      = array('d', [0])
    rk_p      = array('d', [0])

    file2  = root.TFile.Open('filter.root', 'recreate')

    tree2 = root.TTree("comp", "comp")
    tree2.Branch("mchirp_b",     Mc_b,     'Mc_b/D')
    tree2.Branch("mchirp_p",     Mc_p,     'Mc_p/D')
    tree2.Branch("q_b",           q_b,     'q_b/D')
    tree2.Branch("q_p",           q_p,     'q_p/D')
    tree2.Branch("chieff_b", chieff_b,     'chieff_b/D')
    tree2.Branch("chieff_p", chieff_p,     'chieff_p/D')
    tree2.Branch("chip_p",     chip_p,     'chip_p/D')
    tree2.Branch("mf", mf,     'mf/D')
    tree2.Branch("rk_b",           rk_b,     'rk_b/D')
    tree2.Branch("rk_p",           rk_p,     'rk_p/D')


    mvals=np.zeros(5000)
    for i in range(5000):
        mvals[i]=-1
    bkvals=np.zeros(5000)
    ent=0

    for entry in tree3:
        rk=int(tree3.rk_p)
        rb=tree3.rk_b
        mfr=tree3.mf
        if mfr>mvals[rk]:
            mvals[rk]=mfr
            bkvals[rk]=ent
        ent+=1
        

    for i in range(4500):
        if (mvals[i]==-1):
            continue

        idx=int(bkvals[i])
        tree3.GetEntry(idx)
        Mc_p[0]=tree3.Mc_p
        q_p[0]=tree3.q_p
        chieff_p[0]=tree3.chieff_p
        rk_p[0]=tree3.rk_p
        chip_p[0]=tree3.chip_p
        Mc_b[0]=tree3.Mc_b
        q_b[0]=tree3.q_b
        chieff_b[0]=tree3.chieff_b
        rk_b[0]=tree3.rk_b
        if (mvals[i]/np.sqrt(2)<0.5):
            print(int(tree3.rk_p),int(tree3.rk_b),mvals[i]/np.sqrt(2),tree3.Mc_p)
        mf[0]=mvals[i]
        tree2.Fill()

    file2.Write()
    file2.Close()





if __name__ == "__main__":
    main()