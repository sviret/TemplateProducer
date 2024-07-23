'''

'''

import sys
import ROOT as root
from time import time
import numpy as np
from array import array

# The main routine

def main():
    import gendata as gd
    args = sys.argv[1:] # Get input params
    
    # Retrieve the root file name
    fname=args[0]
    # How many injections 
    ninj=int(args[1])

    fparts=fname.split('/')
    basename=fparts[len(fparts)-1]


    f = open(f"filter_{basename.split('root')[0]}txt", "w")

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
    cross     = array('d', [0])
    rk_b      = array('d', [0])
    rk_p      = array('d', [0])

    file2  = root.TFile.Open(f"filter_{basename.split('root')[0]}root", 'recreate')

    tree2 = root.TTree("comp", "comp")
    tree2.Branch("mchirp_b",     Mc_b,     'Mc_b/D')
    tree2.Branch("mchirp_p",     Mc_p,     'Mc_p/D')
    tree2.Branch("q_b",           q_b,     'q_b/D')
    tree2.Branch("q_p",           q_p,     'q_p/D')
    tree2.Branch("chieff_b", chieff_b,     'chieff_b/D')
    tree2.Branch("chieff_p", chieff_p,     'chieff_p/D')
    tree2.Branch("chip_p",     chip_p,     'chip_p/D')
    tree2.Branch("mf",             mf,     'mf/D')
    tree2.Branch("cross",       cross,     'cross/D')
    tree2.Branch("rk_b",         rk_b,     'rk_b/D')
    tree2.Branch("rk_p",         rk_p,     'rk_p/D')

    mvals=np.full((ninj),-1.)
    bkvals=np.zeros(ninj)
    ent=0

    for entry in tree3:
        if ent%100000==0:
            print(ent)
        rk=int(tree3.rk_p)
        if rk>=ninj: 
            ent+=1
            continue
        mfr=tree3.mf
        if mfr>mvals[rk]:
            mvals[rk]=mfr
            bkvals[rk]=ent
        ent+=1
        

    for i in range(ninj):
        if (mvals[i]<=0.):
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
        f.write(f"{int(tree3.rk_p)},{int(tree3.rk_b)},{mvals[i]},{tree3.Mc_p}\n")
        mf[0]=mvals[i]
        cross[0]=tree3.cross
        tree2.Fill()

    file2.Write()
    file2.Close()
    f.close()




if __name__ == "__main__":
    main()