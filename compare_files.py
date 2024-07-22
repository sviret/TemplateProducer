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

# The main routine

def main():
    import gendata as gd
    args = sys.argv[1:] # Get input params
    
    # Retrieve the bank (do this once)
    fname1=args[0]
    fname2=args[1]

    file1 = root.TFile.Open(fname1,"READ")
    file2 = root.TFile.Open(fname2,"READ")

    tree1 = file1.Get("comp")
    tree2 = file2.Get("comp")

    file3  = root.TFile.Open('compa.root', 'recreate')

    tree3 = root.TTree("comp", "comp")

    Mc_p      = array('d', [0])
    mf        = array('d', [0])
    Mc_b      = array('d', [0])
    q_b       = array('d', [0])
    q_p       = array('d', [0])
    chieff_b  = array('d', [0])
    chieff_p  = array('d', [0])
    chip_p    = array('d', [0])
    rk_b      = array('d', [0])
    rk_p      = array('d', [0])
    type      = array('d', [0])

    tree3.Branch("mchirp_p",     Mc_p,     'Mc_p/D')
    tree3.Branch("mchirp_b",     Mc_b,     'Mc_b/D')
    tree3.Branch("q_b",           q_b,     'q_b/D')
    tree3.Branch("q_p",           q_p,     'q_p/D')
    tree3.Branch("chieff_b", chieff_b,     'chieff_b/D')
    tree3.Branch("chieff_p", chieff_p,     'chieff_p/D')
    tree3.Branch("chip_p",     chip_p,     'chip_p/D')
    tree3.Branch("mf", mf,     'mf/D')
    tree3.Branch("rk_b",           rk_b,     'rk_b/D')
    tree3.Branch("rk_p",           rk_p,     'rk_p/D')
    tree3.Branch("bank",     type,     'bank/D')


    mf1c=np.zeros(5000)
    mf2c=np.zeros(5000)
    v2=np.zeros(5000)
    v1=np.zeros(5000)

    rank=0
    for entry in tree1:
        rk=int(tree1.rk_p)
        mf1c[rk]=tree1.mf/np.sqrt(2)
        v1[rk]=rank
        rank+=1

    rank=0
    for entry in tree2:
        rk=int(tree2.rk_p)
        mf2c[rk]=tree2.mf/np.sqrt(2)
        v2[rk]=rank
        rank+=1

    for i in range(0,5000):

        if mf2c[i]==0 and mf1c[i]==0:
            continue

        print(i,mf1c[i],mf2c[i])

        if (mf1c[i]>mf2c[i]):
            idx=int(v1[i])
            tree1.GetEntry(idx)
            mf[0]=tree1.mf/np.sqrt(2)
            Mc_p[0]=tree1.Mc_p
            q_p[0]=tree1.q_p
            chieff_p[0]=tree1.chieff_p
            rk_p[0]=tree1.rk_p
            chip_p[0]=tree1.chip_p
            Mc_b[0]=tree1.Mc_b
            q_b[0]=tree1.q_b
            chieff_b[0]=tree1.chieff_b
            rk_b[0]=tree1.rk_b
            type[0]=1
        else:
            idx=int(v2[i])
            tree2.GetEntry(idx)
            mf[0]=tree2.mf/np.sqrt(2)
            Mc_p[0]=tree2.Mc_p
            q_p[0]=tree2.q_p
            chieff_p[0]=tree2.chieff_p
            rk_p[0]=tree2.rk_p
            chip_p[0]=tree2.chip_p
            Mc_b[0]=tree2.Mc_b
            q_b[0]=tree2.q_b
            chieff_b[0]=tree2.chieff_b
            rk_b[0]=tree2.rk_b
            type[0]=2
        tree3.Fill()

    file3.Write()
    file3.Close()

if __name__ == "__main__":
    main()