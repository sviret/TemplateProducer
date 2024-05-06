import numpy as npy
import scipy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import math
import time
import matplotlib.cm as cm
import gen_noise as gn
import gen_template as gt

#constantes physiques
G=6.674184e-11
Msol=1.988e30
c=299792458
MPC=3.086e22


######################################################################################################################################################################################
parameters = {'font.size': 15,'axes.labelsize': 15,'axes.titlesize': 15,'figure.titlesize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'legend.fontsize': 15,'legend.title_fontsize': 15,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [10,5]}
plt.rcParams.update(parameters)
  
#################################################################################################################################################################################################
'''
Class handling gravidational wave dataset production (either for training or testing)

Options:

--> mint     : Solar masses range (mmin,mmax) for the (m1,m2) mass grid to be produced
--> step     : Mass step between two samples
--> NbB      : Number of background realisations per template
--> tcint    : Range for the coalescence time within the last signal chunk (in sec, should be <Ttot)
               if the last chunk has the length T, the coalescence will be randomly placed within the range
               [tcint[0]*T,tcint[1]*T]
--> choice   : The template type (EOB or EM)
--> kindBank : The type of mass grid used (linear or optimal)
--> whitening: Frequency-domain or time-domain whitening (1 or 2) 
'''


class GenDataSet:

    """Classe générant des signaux des sets de training 50% dsignaux+bruits 50% de bruits"""

    def __init__(self,mint=(10,50),NbB=1,tcint=(0.75,0.95),kindPSD='flat',kindTemplate='EM',Ttot=1,fe=2048,kindBank='linear',paramFile=None,step=0.1,choice='train',length=100,whitening=1,ninj=0,customPSD=None):
    
                
        self.__custPSD=[]
        if customPSD!=None:
            self.__custPSD=customPSD

        self.__choice=choice
        self.__length=length
        self.__ninj=ninj
        if os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            raise FileNotFoundError("Le fichier de paramètres n'existe pas")


        start_time = time.time()
        print("Starting dataset generation")

        # Noise stream generator
        #self.__NGenerator=gn.GenNoise(Ttot=self.__listTtot,fe=self.__listfe,kindPSD=self.__kindPSD,whitening=self.__whiten,customPSD=self.__custPSD)
        # Template stream generator
        self.__TGenerator=gt.GenTemplate(Tsample=self.__listTtot,fe=self.__listfe,kindTemplate=self.__kindTemplate,whitening=self.__whiten,customPSD=self.__custPSD)
    
        self.__listDelta_t=[] # pour le plot des portions de SNR
        self.__listSNRevolTime=[]
        self.__listSNRevol=[]
        self.__listfnames=[]
        self.__listSNRchunksAuto=[[] for x in range(self.__nTtot)]
        self.__tmplist=[]

        print("1 After init --- %s seconds ---" % (time.time() - start_time))
            
        self._genGrille()   # Binary objects mass matrix
        
        print("2 After grille --- %s seconds ---" % (time.time() - start_time))
        
        self._genSigSet()   # The signals
        
        print("3 After signal --- %s seconds ---" % (time.time() - start_time))

    
            
    '''
    DATASET 1/
    
    Parser of the parameters file
    '''

    def _readParamFile(self,paramFile):

        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]
          
        if lignes[0][0]!='Ttot' or lignes[1][0]!='fe' or lignes[2][0]!='kindPSD' or lignes[3][0]!='mint' or lignes[4][0]!='tcint' or lignes[5][0]!='NbB' or lignes[6][0]!='kindTemplate' or lignes[7][0]!='kindBank' or lignes[8][0]!='step' or lignes[9][0]!='whitening'or len(lignes)!=10:
            raise Exception("Dataset param file error")
        if not len(lignes[0])==len(lignes[1]):
            raise Exception("Ttot and fe vectors don't have the same size")
        
        nValue=len(lignes[0]) # taille de la liste des valeurs de Ttot +1 (légende)
        self.__nTtot=nValue-1
        self.__listTtot=[]
        self.__listfe=[]
        if nValue > 2:  # Multi band
            for x in range(1,nValue):
                self.__listTtot.append(float(lignes[0][x]))
                self.__listfe.append(float(lignes[1][x]))
            self.__Ttot=sum(self.__listTtot)
            self.__fe=max(self.__listfe)
        else: # Single band
            self.__Ttot=float(lignes[0][1])
            self.__listTtot.append(self.__Ttot)
            self.__fe=float(lignes[1][1])
            self.__listfe.append(self.__fe)
            
        self.__kindPSD=lignes[2][1]
        self.__mmin=min(float(lignes[3][1]),float(lignes[3][2]))
        self.__mmax=max(float(lignes[3][1]),float(lignes[3][2]))
        self.__tcmin=self.__Ttot+self.__listTtot[0]*(max(min(float(lignes[4][1]),float(lignes[4][2])),0.5)-1.)
        self.__tcmax=self.__Ttot+self.__listTtot[0]*(min(max(float(lignes[4][1]),float(lignes[4][2])),1.)-1.)
        self.__NbB=int(lignes[5][1])
        self.__kindTemplate=lignes[6][1]
        self.__step=float(lignes[8][1])
        self.__whiten=int(lignes[9][1])
        kindBank=lignes[7][1]
        self.__kindBank=kindBank
                    
    '''
    DATASET 2/
    
    Produce a data grid with the mass coordinates
    '''

    def _genGrille(self):

        if self.__kindBank=='linear':
            n = self.__step
            N = len(npy.arange(self.__mmin, self.__mmax, n))
            self.__Ntemplate=int((N*(N+1))/2)
            self.__GrilleMasses=npy.ones((self.__Ntemplate,2))
            self.__GrilleSpins=npy.zeros((self.__Ntemplate,2))
            self.__GrilleMasses.T[0:2]=self.__mmin
            c=0
            
            # Fill the grid (half-grid in fact)
            # Each new line starts at the diagonal, then reach the end
            # mmin,mmax
            # mmin+step,mmax
            #...
            
            for i in range(N):
                for j in range(i,N):
                
                    self.__GrilleMasses[c][0]=self.__mmin+i*self.__step
                    self.__GrilleMasses[c][1]=self.__mmin+j*self.__step
                    c+=1
                    
        else: # The optimized bank
            Mtmp=[]
            Sztmp=[]
            start=self.__length
            ntmps=self.__ninj
            compt=0
            print("Templates are taken into file ",self.__kindBank)
            with open(os.path.dirname(__file__)+'/params/'+self.__kindBank) as mon_fichier:
                lines=mon_fichier.readlines()
                for line in lines:
                    if '#' in line:
                        compt=0
                        continue
                    if '!' in line:
                        compt=0
                        continue
                    if compt>=start+ntmps:
                        break
                    if compt<start:
                        compt+=1
                        continue

                    data=line.strip()
                    pars=data.split(' ')
                    # Cuts on total mass
                    #print(pars[1]),float(pars[2])
                    if (float(pars[1])+float(pars[2])<self.__mmin and float(pars[1])+float(pars[2])>self.__mmax):
                        self.__tmplist.append([float(pars[0]),float(pars[1]),float(pars[2]),0])
                        compt+=1
                        continue
                    compt+=1
                    Mtmp.append([float(pars[1]),float(pars[2])])
                    Sztmp.append([float(pars[3]),float(pars[4])])
                    self.__tmplist.append([float(pars[0]),float(pars[1]),float(pars[2]),1])
            M=npy.asarray(Mtmp)
            Spins=npy.asarray(Sztmp)
            self.__GrilleMasses=M
            self.__GrilleSpins=Spins
            self.__Ntemplate=len(self.__GrilleMasses)
        
    '''
    DATASET 3/
    
    Produce the templates
    '''

    def _genSigSet(self):

        self.__Sig=[]
        self.__Noise=[]
        c=0
        
        # First we produce the object with the correct size
        # The size is Ntemplate
        for j in range(self.__nTtot): # Loop over samples

            self.__Sig.append(npy.zeros(self.__Ntemplate,dtype=object))
            self.__Noise.append(npy.zeros(self.__Ntemplate,dtype=object))


        # Now fill the object
        for i in range(0,self.__Ntemplate):
            if c%100==0:
                print("Producing sample ",c,"over",self.__Ntemplate)
            self.__TGenerator.majParams(m1=self.__GrilleMasses[i][0],m2=self.__GrilleMasses[i][1],s1=self.__GrilleSpins[i][0],s2=self.__GrilleSpins[i][1])

            # Create the template            
            temp,freqs=self.__TGenerator.getNewSample(kindPSD=self.__kindPSD,
                                                             Tsample=self.__Ttot,
                                                             tc=self.__TGenerator.duration(),norm=False)

            # Fill the corresponding data
            for j in range(self.__nTtot):
                self.__Sig[j][c]=temp[j]
                self.__Noise[j][c]=freqs[j]
            
            c+=1

             
     
    '''
    DATASET 4/
    
    Setters/getters
    '''
    
    def saveGenerator(self,dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")

        # Save the sample in an efficient way
        for k in range(self.__nTtot):
            fname=dossier+self.__choice+'-'+self.__kindBank+'-templates_from-'+str(self.__length)+'-to-'+str(self.__length+self.__ninj)+'-'+str(k)+'of'+str(self.__nTtot)+'_samples'
            npy.savez_compressed(fname,self.__Noise[k],self.__Sig[k])
            self.__listfnames.append(fname)
        
        # Save the object without the samples
        self.__Sig=[]
        self.__Noise=[]
        self.__TGenerator=[]
        fichier=dossier+'summary-'+self.__choice+'-'+self.__kindBank+'-templates_from-'+str(self.__length)+'-to-'+str(self.__length+self.__ninj)+'-'+str(self.__nTtot)+'bands-'+str(self.__Ttot)+'s'+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()


    @classmethod
    def readGenerator(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
        
        print("We deal with a dataset containing",obj.__nTtot,"frequency bands")
        for i in range(obj.__nTtot):
            print("Band",i,"data is contained in file",obj.__listfnames[i]+'.npz')
            data=npy.load(str(obj.__listfnames[i])+'.npz', allow_pickle=True)
            obj.__Sig.append(data['arr_1'])
            obj.__Noise.append(data['arr_0'])
        data=[] # Release space
        f.close()
        return obj


    def getBkParams(self):
        return self.__tmplist
    
    def getTemplate(self,rank=0):
        nbands=self.__nTtot

        print("Getting a template based on",nbands,"frequency bands")
        dset=[]
        fset=[]
        #print("Masses:",self.__GrilleMasses[rank])
        #print("Spins:",self.__GrilleSpins[rank])

        for i in range(nbands):
            dset.append(self.__Sig[i][rank])
            fset.append(self.__Noise[i][rank])
            
        return dset,fset,(self.__listfe,self.__listTtot,self.__GrilleMasses[rank],self.__GrilleSpins[rank],self.__Ttot)

    @property
    def Ntemplate(self):
        return self.__Ntemplate
    @property
    def Nsample(self):
        return self.__Ntemplate*2*self.__NbB
    @property
    def Labels(self):
        return self.__Labels
    @property
    def mInt(self):
        return self.__mmin,self.__mmax
    @property
    def mStep(self):
        return self.__step
    @property
    def kindPSD(self):
        return self.__kindPSD
    @property
    def kindTemplate(self):
        return self.__kindTemplate
    @property
    def kindBank(self):
        return self.__kindBank
############################################################################################################################################################################################
def parse_cmd_line():
    import argparse
    """Parseur pour la commande gendata"""
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="commande à choisir",choices=['noise', 'template', 'set', 'chunck', 'all'])
    parser.add_argument("--kindPSD","-kp",help="Type de PSD à choisir",choices=['flat', 'analytic','realistic'],default='analytic')
    parser.add_argument("--kindTemplate","-kt",help="Type de PSD à choisir",choices=['EM', 'EOB'],default='EM')
    parser.add_argument("--time","-t",help="Durée du signal",type=float,nargs='+',default=[1., 9.])
    parser.add_argument("-fe",help="Fréquence du signal",type=float,nargs='+',default=[2048, 512])
    parser.add_argument("-fmin",help="Fréquence minimale visible par le détecteur",type=float,default=20)
    parser.add_argument("-fmax",help="Fréquence maximale visible par le détecteur",type=float,default=1000)
    parser.add_argument("-white",help="Whitening type: none (0), frequency domain (1), time domain (2)",type=int,default=1)
    parser.add_argument("-snr",help="SNR (pour creation de sequence)",type=float,default=7.5)
    parser.add_argument("-m1",help="Masse du premier objet",type=float,default=20)
    parser.add_argument("-m2",help="Masse du deuxième objet",type=float,default=20)
    parser.add_argument("-n",help="Number of injections for frame gen",type=float,default=10)
    parser.add_argument("-length",help="Length of data chunck (in s)",type=float,default=300)
    parser.add_argument("--set","-s",help="Choix du type de set par défaut à générer",default=None)
    parser.add_argument("-step",help="Pas considéré pour la génération des paires de masses",type=float,default=0.1)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres de set",default=None)
    parser.add_argument("--verbose","-v",help="verbose mode",type=bool,default=False)
    
    args = parser.parse_args()
    return args

'''
Main part of gendata.py
'''

def main():

    import gendata as gd
    import gen_template as gt

    args = parse_cmd_line()

    if args.cmd=='template': # Simple Template generation
        TGenerator=gt.GenTemplate(Tsample=args.time,fe=args.fe,kindTemplate=args.kindTemplate,fDmin=args.fmin,fDmax=args.fmax,whitening=0,verbose=args.verbose,customPSD=_brutePSD)
        TGenerator.majParams(args.m1,args.m2)
        TGenerator.getNewSample(kindPSD=args.kindPSD,Tsample=TGenerator.duration(),tc=TGenerator.duration(),norm=False)
    
    else: # Bank of template in the timing domain

        cheminout = './generators/'
        Generator=gd.GenDataSet(paramFile=args.paramfile,choice=args.set,length=args.length,ninj=args.n)
        Generator.saveGenerator(cheminout)
    



############################################################################################################################################################################################
if __name__ == "__main__":
    main()
