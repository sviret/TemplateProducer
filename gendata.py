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
from numcompress import compress, decompress
from gwpy.timeseries import TimeSeries

"""
Normallement gendata fonction exactement comme avant avec les meme commande.
Mais mtn quand on rentre la commande pour generer la Frame dans le terminal,
On peut ajouter un terme -tf chemin/du/fichier/texte
On prend des lignes aleatoire dans le fichier et genere les signaux a partir de ca.
"""
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

    def __init__(self,mint=(10,50),NbB=1,tcint=(0.75,0.95),kindPSD='flat',kindTemplate='EM',Ttot=1,fe=2048,kindBank='linear',paramFile=None,step=0.1,choice='train',length=100,whitening=1,ninj=0, txtfile=None):
    
                
        self.__choice=choice
        self.__length=length
        self.__ninj=ninj
        if os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            raise FileNotFoundError("Le fichier de paramètres n'existe pas")
        self.fromtxtfile=True if txtfile is not None else False

        _brutePSD=[]

        if self.__kindPSD=='realistic':

            psd=[]
            freq=[]

            with open(self.__PSDfile,"r") as f:
                for line in f:
                    value=[float(v) for v in line.split(' ')]
                    if (value[0]<0.2*self.__fmin or value[0]>self.__fmax):
                        continue
                    psd.append(value[1]**2)
                    freq.append(value[0])

            f.close()
            _brutePSD.append([psd,freq])  
            self.__custPSD=_brutePSD 

            print('Generate dataset with custom PSD:',self.__PSDfile)

        if self.__choice!='frame':
            start_time = time.time()
            print(f"Starting dataset generation of {int(ninj)} templates from file {self.__kindBank}")

            # Template (do it once)
            self.__TGenerator=gt.GenTemplate(Tsample=self.__listTtot,fDmin=self.__fmin,fDmax=self.__fmax,fe=self.__listfe,kindTemplate=self.__kindTemplate,whitening=self.__whiten,customPSD=self.__custPSD,PSD=self.__kindPSD,verbose=False)
    
            self.__listfnames=[]
            self.__tmplist=[]

            print("1 After init --- %s seconds ---" % (time.time() - start_time))
            
            self._genGrille()   # Binary objects mass matrix
        
            print("2 After grille --- %s seconds ---" % (time.time() - start_time))
        
            self._genSigSet()   # The signals
        
            print("3 After signal --- %s seconds ---" % (time.time() - start_time))
        else:
            start_time = time.time()
            print("Starting frame generation")
            self.__NGenerator=gn.GenNoise(Ttot=self.__listTtot,fe=self.__listfe,kindPSD=self.__kindPSD,fmin=self.__fmin,fmax=self.__fmax,whitening=self.__whiten,customPSD=self.__custPSD,verbose=False)
            self.__TGenerator=gt.GenTemplate(Tsample=self.__listTtot,fe=self.__listfe,kindTemplate=self.__kindTemplate,fDmin=self.__fmin,whitening=self.__whiten,customPSD=self.__custPSD,PSD=self.__kindPSD,verbose=False)#, fromtxtfile = self.fromtxtfile)
            self.__listfnames=[]

            self._genNoiseSequence() # Produce a noise sequence over the full length requested 
     
            # Then produce the injections and superimpose them
            ninj=int(self.__ninj)
            interval=self.__length/(self.__ninj+1)

            if self.fromtxtfile :
                if os.path.isfile(txtfile):
                    self.txt_params = self._readtxtFile(txtfile)
                    print("Retrieving templates parameters from txt file")
                else:
                    raise FileNotFoundError("Le fichier texte des templates n'existe pas")


            print(ninj,"signals will be injected in the data stream")
            self.__injections=[]
            
            for i in range(ninj):
                if self.fromtxtfile :
                    ligne_aleatoire = npy.random.choice(self.txt_params).strip().split(',')
                    # !id,m1,m2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,chieff,chip,tcoal,SNRH,SNRL
                    id = int(ligne_aleatoire[0])
                    m1 = float(ligne_aleatoire[1])
                    m2 = float(ligne_aleatoire[2])
                    spin1x = float(ligne_aleatoire[3])
                    spin1y = float(ligne_aleatoire[4])
                    spin1z = float(ligne_aleatoire[5])
                    spin2x = float(ligne_aleatoire[6])
                    spin2y = float(ligne_aleatoire[7])
                    spin2z = float(ligne_aleatoire[8])
                    chieff = float(ligne_aleatoire[9])
                    chip = float(ligne_aleatoire[10])
                    tcoal = float(ligne_aleatoire[11])
                    SNRH = float(ligne_aleatoire[12])
                    SNRL = float(ligne_aleatoire[13])
                    self.__TGenerator.majParams(m1,m2,s1x=spin1x,s2x=spin2x,s1y=spin1y,s2y=spin2y,s1z=spin1z,s2z=spin2z,D=None,Phic=None)

                else :
                    m1=npy.random.uniform(5,75)
                    m2=npy.random.uniform(5,75)
                    self.__TGenerator.majParams(m1,m2)

                SNR=npy.random.uniform(15,15)
                self.__TGenerator.getNewSample(kindPSD=self.__kindPSD,Tsample=self.__TGenerator.duration(),norm=True)
                data=self.__TGenerator.signal()
                data_r=self.__TGenerator.signal_raw()
                norm=self.__TGenerator.norma()
                #print(norm)


                #randt=npy.random.uniform(self.__TGenerator.duration(),self.__length)
                randt=npy.random.normal((i+1)*interval,interval/20.)
                inj=[m1,m2,SNR,randt]
                self.__injections.append(inj)
                if self.fromtxtfile :
                    print("Injection",i,"(id,m1,m2,SNR,tc)=(",f'{id}',f'{m1:.1f}',f'{m2:.1f}',f'{SNR:.1f}',f'{randt:.1f}',")")
                else :
                    print("Injection",i,"(m1,m2,SNR,tc)=(",f'{m1:.1f}',f'{m2:.1f}',f'{SNR:.1f}',f'{randt:.1f}',")")

                idxstart=int((randt-self.__TGenerator.duration())*self.__listfe[0])
    
                for i in range(len(data)):
                    self.__Signal[0][idxstart+i]+=SNR*data[i]
                    self.__Signal_raw[0][idxstart+i]+=SNR*data_r[i]/norm
    
                randt=npy.random.uniform(self.__TGenerator.duration(),self.__length)
                data=[]
                
            self.__Noise[0] += self.__Signal[0] # Hanford
            self.__Noise[1] += self.__Signal[0] # Livingston
            self.__Noise_raw[0] += self.__Signal_raw[0] # Hanford
            #self.__Noise_raw[1] += self.__Signal_raw[0] # Livingston
            
            npts=float(len(self.__Noise[0]))
            norm=self.__length/npts
            plt.figure(figsize=(10,5))
            plt.xlabel('t (s)')
            plt.ylabel('h(t)')
            plt.grid(True, which="both", ls="-")
            plt.plot(npy.arange(len(self.__Noise[0]))*norm, self.__Noise[0])
            plt.plot(npy.arange(len(self.__Noise[0]))*norm, self.__Signal[0])
            plt.show()

            plt.figure(figsize=(10,5))
            plt.xlabel('t (s)')
            plt.ylabel('h(t)')
            plt.grid(True, which="both", ls="-")
            plt.plot(npy.arange(len(self.__Noise[0]))*norm, self.__Noise_raw[0])
            plt.plot(npy.arange(len(self.__Noise[0]))*norm, self.__Signal_raw[0])
            plt.show()
            
    '''
    DATASET 1/
    
    Parser of the parameters file
    '''

    def _readParamFile(self,paramFile):

        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]
          
        if lignes[0][0]!='Ttot' or lignes[1][0]!='fe' or lignes[2][0]!='kindPSD' or lignes[3][0]!='mint' or lignes[4][0]!='tcint' or lignes[5][0]!='NbB' or lignes[6][0]!='kindTemplate' or lignes[7][0]!='kindBank' or lignes[8][0]!='step' or lignes[9][0]!='whitening' or len(lignes)!=11:
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
        if (self.__kindPSD=='realistic'):
            self.__PSDfile=lignes[2][2]
        self.__mmin=min(float(lignes[3][1]),float(lignes[3][2]))
        self.__mmax=max(float(lignes[3][1]),float(lignes[3][2]))
        self.__tcmin=self.__Ttot+self.__listTtot[0]*(max(min(float(lignes[4][1]),float(lignes[4][2])),0.5)-1.)
        self.__tcmax=self.__Ttot+self.__listTtot[0]*(min(max(float(lignes[4][1]),float(lignes[4][2])),1.)-1.)
        self.__NbB=int(lignes[5][1])
        self.__kindTemplate=lignes[6][1]
        self.__step=float(lignes[8][1])
        self.__whiten=int(lignes[9][1])
        self.__fmin=float(lignes[10][1])
        self.__fmax=float(lignes[10][2])        
        kindBank=lignes[7][1]
        self.__kindBank=kindBank


    def _readtxtFile(self,txtfile) :
        with open(txtfile, 'r') as f:
            lignes = f.readlines()

        lignes_parametres = [ligne for ligne in lignes if not ligne.startswith('#') and not ligne.startswith('!')]
        return lignes_parametres

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
            self.__TGenerator.majParams(m1=self.__GrilleMasses[i][0],m2=self.__GrilleMasses[i][1],s1z=self.__GrilleSpins[i][0],s2z=self.__GrilleSpins[i][1])
        
            # Create the template            
            temp,freqs=self.__TGenerator.getNewSample(kindPSD=self.__kindPSD,
                                                             Tsample=self.__Ttot,
                                                             tc=self.__TGenerator.duration(),norm=True)

            # Fill the corresponding data
            for j in range(self.__nTtot):

                if (not isinstance(freqs[j],int)):
                    fact=1.
                    if self.__whiten==0: # Special case, rescale if no whitening applied
                        fact=1e20

                    trunc=0
                    if (len(temp[j])>len(freqs[j])):
                        trunc=len(temp[j])-len(freqs[j])

                    # Compress the data
                    self.__Sig[j][c]=compress(list(fact*temp[j][trunc:]), precision=10)
                    self.__Noise[j][c]=compress(list(freqs[j]), precision=5)
                else:
                    self.__Sig[j][c]=0
                    self.__Noise[j][c]=0

            del temp,freqs            
            c+=1

    # Noise 
    def _genNoiseSequence(self):

        # Noise is produced from PSD unsing inverse FFTs, therefore we try to 
        # keep them reasonably long

        nsamples=int(self.__length/self.__NGenerator.Ttot)
        if self.__length/self.__NGenerator.Ttot-nsamples>0:
            nsamples+=1

        npts=int(self.__length*self.__fe)

        # Whitened info
        self.__Noise=[]
        self.__Signal=[]

        # Raw info
        self.__Noise_raw=[]
        self.__Signal_raw=[]

        chunck_V=[]
        chunck_H=[]
        strain_V=[]
        strain_H=[]
        for i in range(nsamples):
            self.__NGenerator.getNewSample()
            chunck_V.append(self.__NGenerator.getNoise())
            strain_V.append(self.__NGenerator.getNoise_unwhite())
            self.__NGenerator.getNewSample()
            chunck_H.append(self.__NGenerator.getNoise())
            strain_H.append(self.__NGenerator.getNoise_unwhite())

        self.__Noise.append(npy.ravel(npy.squeeze(chunck_V))[0:npts]) # Hanford
        self.__Noise.append(npy.ravel(npy.squeeze(chunck_H))[0:npts]) # Livingston
        self.__Noise_raw.append(npy.ravel(npy.squeeze(strain_V))[0:npts]) # Hanford
        self.__Noise_raw.append(npy.ravel(npy.squeeze(strain_H))[0:npts]) # Livingston
        self.__Signal.append(npy.zeros(len(self.__Noise[0])))
        self.__Signal.append(npy.zeros(len(self.__Noise[0])))
        self.__Signal_raw.append(npy.zeros(len(self.__Noise[0])))
        self.__Signal_raw.append(npy.zeros(len(self.__Noise[0])))
 
     
    '''
    DATASET 4/
    
    Setters/getters
    '''
    
    def saveGenerator(self,dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")

        # Save the sample in an efficient way
        for k in range(self.__nTtot):
            fname=dossier+self.__choice+'-'+self.__kindBank+'-templates_from-'+str(int(self.__length))+'-to-'+str(int(self.__length+self.__ninj))+'-'+str(k)+'of'+str(self.__nTtot)+'_samples'
            npy.savez_compressed(fname,self.__Noise[k],self.__Sig[k])
            self.__listfnames.append(fname)
        
        # Save the object without the samples
        self.__Sig=[]
        self.__Noise=[]
        self.__TGenerator=[]
        fichier=dossier+'summary-'+self.__choice+'-'+self.__kindBank+'-templates_from-'+str(int(self.__length))+'-to-'+str(int(self.__length+self.__ninj))+'-'+str(self.__nTtot)+'bands-'+str(self.__Ttot)+'s'+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()

    def saveFrame(self,dossier):
        if not(os.path.isdir(dossier)):
            raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")

        # Save the sample in an efficient way
        fname=dossier+self.__choice+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+str(self.__length)+'s'+'-data'
        npy.savez_compressed(fname,self.__Noise[0],self.__Noise[1],self.__Signal[0])
        self.__listfnames.append(fname)
        
        gpsinit=1400000000
      
        t = TimeSeries(self.__Noise_raw[0],channel="Noise_and_injections",sample_rate=self.__fe,unit="time",t0=self.__length+gpsinit)
        u = TimeSeries(self.__Noise_raw[1],channel="Noise_alone",sample_rate=self.__fe,unit="time",t0=gpsinit)
        v = TimeSeries(self.__Noise_raw[1],channel="Noise_alone",sample_rate=self.__fe,unit="time",t0=gpsinit+self.__length)
        t.write(f"{fname}_Strain.gwf")
        u.write(f"{fname}_Noise.gwf") # MBTA needs pure noise to compute the PSD
        v.write(f"{fname}_Noise2.gwf") 

        # Save the object without the samples (basically just the weights)
        self.__Signal=[]
        self.__Noise=[]
        fichier=dossier+self.__choice+'-'+str(self.__nTtot)+'chunk'+'-'+self.__kindPSD+'-'+self.__kindTemplate+'-'+str(self.__whiten)+'-'+str(self.__length)+'s'+'.p'
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


    def getBkParams(self,rank=0):
        return self.__tmplist[rank]
    
    def getTemplate(self,rank=0):
        nbands=self.__nTtot

        dset=[]
        fset=[]

        fact=1.
        if self.__whiten==0:
            fact=1e-20

        for i in range(nbands):

            if (not isinstance(self.__Sig[i][rank],int)):
                dset.append(fact*npy.asarray(decompress(self.__Sig[i][rank])))
                fset.append(npy.asarray(decompress(self.__Noise[i][rank])))
            else:
                dset.append(self.__Sig[i][rank])
                fset.append(self.__Noise[i][rank])
        return dset,fset,(self.__listfe,self.__listTtot,self.__GrilleMasses[rank],self.__GrilleSpins[rank],self.__Ttot)

    def getParams(self,rank=0):

        return (self.__listfe,self.__listTtot,self.__GrilleMasses[rank],self.__GrilleSpins[rank],self.__Ttot,self.__custPSD,self.__kindPSD,self.__fmin,self.__fmax)


    #@property
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
    parser.add_argument("cmd", help="commande à choisir",choices=['bank','frame'])
    parser.add_argument("-n",help="Number template to produce",type=float,default=10)
    parser.add_argument("-start",help="Where to start in the bank",type=float,default=0)
    parser.add_argument("--set","-s",help="Name of the bank",default='test')
    parser.add_argument("-length",help="Length of data chunck (in s)",type=float,default=300)
    parser.add_argument("--paramfile","-pf",help="Fichier csv des paramètres de set",default=None)
    parser.add_argument("--txtfile","-tf",help="Fichier txt des parametre des templates",default=None)

    
    args = parser.parse_args()
    return args

'''
Main part of gendata.py
'''

def main():

    import gendata as gd
    import gen_template as gt

    args = parse_cmd_line()
    cheminout = './generators/'
    
    if args.set=='frame':
        set='frame'
        Generator=gd.GenDataSet(paramFile=args.paramfile,choice=set,length=args.length,ninj=args.n, txtfile=args.txtfile)
        Generator.saveFrame(cheminout)
    else:
        Generator=gd.GenDataSet(paramFile=args.paramfile,choice=args.set,length=args.start,ninj=args.n)
        Generator.saveGenerator(cheminout)
    



############################################################################################################################################################################################
if __name__ == "__main__":
    main()
