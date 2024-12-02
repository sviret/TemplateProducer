import numpy as npy
import scipy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import math
import time
from pycbc.waveform import get_td_waveform
from pycbc.waveform import utils
from pycbc.pnutils import f_FRD
from scipy.stats import norm
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import gen_noise as gn

#constantes physiques
G=6.674184e-11
Msol=1.988e30
c=299792458
MPC=3.086e22


######################################################################################################################################################################################
parameters = {'font.size': 15,'axes.labelsize': 15,'axes.titlesize': 15,'figure.titlesize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'legend.fontsize': 15,'legend.title_fontsize': 15,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [10,5]}
plt.rcParams.update(parameters)



#########################################################################################################################################################################################
'''
Class handling signal generation (templates)

Option:

->Tsample      : sampling durations, in s (Default is 1)
->fe           : sampling frequencies, in Hz (Default is 2048)
->kindTemplate : generator type: 'EM' or 'EOB' (Default is EM)
->fDmin        : the minimal sensitivity of the detector (Default is 15Hz)
->fDmax        : the maximal sensitivity of the detector (Default is 1000Hz)
->whitening    : type of signal whitening:
                0: No whitening
                1: Frequency-domain whitening (Default standard procedure)
                2: Time-domain whithening (Zero latency, as described in https://dcc.ligo.org/public/0141/P1700094/005/main_v5.pdf)


'''

class GenTemplate:

    def __init__(self,Tsample=1,fe=2048,fDmin=15,fDmax=1000,kindTemplate='EM',whitening=1,verbose=False,customPSD=None,PSD='flat'):

        if not (isinstance(fe,int) or isinstance(fe,float) or isinstance(fe,list)) and not (isinstance(fDmin,int) or isinstance(fDmin,float)):
            raise TypeError("fe et fDmin doivent être des ints ou des floats (fe peut aussi être une list)")
    
        if not(isinstance(kindTemplate,str)):
            raise TypeError("kindTemplate doit être de type str")

        
        self.__type=1
        if kindTemplate!='EM':
            self.__type=0
        
        self.__custPSD=[]
        if customPSD!=None:
            self.__custPSD=customPSD
        
        # We play the same trick than for the noise here
        # We will generate one big sample at max frequency and resample only afterwards
        # As this is for ML we just have to do this once
        
        if isinstance(fe,list):
            if not isinstance(Tsample,list):
                raise TypeError("Tsample et fe doivent être du même type")
            elif not len(Tsample)==len(fe):
                raise ValueError("Les list Ttot et fe doivent faire la même taille")
            else:
                self.__listTsample=Tsample
                self.__listfe=fe
                self.__Tsample=sum(Tsample)
                self.__fe=max(fe)
                self.__nTsample=len(self.__listTsample)
        else:
            self.__Tsample=Tsample
            self.__fe=fe
            self.__nTsample=1
        
        #print(self.__fe)
        
        # Then we just instantiate some very basic params

        self.__whiten=whitening
        self.__fDmind=fDmin            # The minimum detectable frequency
        self.__fDmin=0.95*fDmin        # Minimum frequency with a margin (need it to smooth the template FFT)
        self.__fDmaxd=fDmax            # The maximum detectable frequency
        self.__delta_t=1/self.__fe     # Time step
        self.__Tdepass=0.1*self.__type # For the EM mode we add a small period after the chirp (to smooth FFT).
                                       # This is included in EOB by construction
        self.__m1=10*Msol              # m1,m2 in solar masses
        self.__m2=10*Msol              #
        self.__D=1*MPC                 # D in Mpc
        self.__Phic=0                  # Initial phase
        
        self.__kindTemplate=kindTemplate              # Template type

        # Parameters related to template SNR sharing
        self._tint=npy.zeros(self.__nTsample)
        self._fint=npy.zeros(self.__nTsample)
        self._currentSnr=npy.zeros(self.__nTsample)
        self._rawSnr=npy.zeros(self.__nTsample)
        self._evolSnrTime = []
        self._evolSnr = []
        self._evolSnrFreq = []
        self.__verb=verbose
        self.__Ttot=self.__Tsample
        self.__Noise=gn.GenNoise(Ttot=self.__Ttot,fe=self.__fe, kindPSD=PSD,fmin=fDmin,fmax=self.__fDmaxd,whitening=self.__whiten,customPSD=self.__custPSD,verbose=False)
        self.__Noise.getNewSample()
 
    '''
    Template 1/10
    
    EM approximation calculations
    
    Here the time is computed with the equation (1.68) of the following document:

    http://sviret.web.cern.ch/sviret/docs/Chirps.pdf
    
    f0 is the frequency at which the detector start to be sensitive to the signal, we define t0=0
    The output is tc-t0
    Value is computed with the newtonian approximation, maybe we should change for EOB
    '''
  
    def phi(self,t):
        return -npy.power(2.*(self.__tc-t)/(5.*self.__tSC),5./8.)
                        
    def h(self,t):
        A=npy.power(2./5.,-1./4.)*self.__rSC/(32*self.__D)
        return A*npy.power((self.__tc-t)/self.__tSC,-1./4.)*npy.cos(2*(self.phi(t)+self.__Phic))

    def hq(self,t):
        A=npy.power(2./5.,-1./4.)*self.__rSC/(32*self.__D)
        return A*npy.power((self.__tc-t)/self.__tSC,-1./4.)*npy.sin(2*(self.phi(t)+self.__Phic))

    def getTchirp(self,f0):
        return npy.power(125./128.,1./3.)*npy.power(self.__tSC,-5./3.)*npy.power(2*npy.pi*f0,-8./3.)/16.#pas terrible pour calculer tisco car w(t) etabli dans le cadre Newtonien
        
    def get_t(self,f0):#f frequence de l'onde grav omega/2pi --> si on souhaite frequence max dans le detecteur fmax ca correspond a fGW=fmax/2
        return self.__tc-self.getTchirp(f0)
        
    def get_f(self,delta_t):
        return npy.power(125./(128.*(16**3)),1./8.)*npy.power(self.__tSC,-5./8.)*npy.power(delta_t,-3./8.)/(2*npy.pi)
    
    
    '''
    Template 2/10
    
    Method updating the templates properties, should be called right after the initialization
    
    Info on template types:
    
    EOB uses the option SEOBNRv4, which includes the merger and ringdown stages
    EM is a simple EM equivalent model
    '''
    
    def majParams(self,m1,m2,s1x=0,s2x=0,s1y=0,s2y=0,s1z=0,s2z=0,D=None,Phic=None):
    
        # Start by updating the main params
        self.__s1x=s1x 
        self.__s2x=s2x
        self.__s1y=s1y 
        self.__s2y=s2y 
        self.__s1z=s1z 
        self.__s2z=s2z 
        self.__m1=self.__m1 if m1 is None else m1*Msol
        self.__m2=self.__m2 if m2 is None else m2*Msol
        self.__D=self.__D if D is None else D*MPC
        self.__Phic=self.__Phic if Phic is None else Phic
        self.__M=self.__m1+self.__m2                        # Total mass
        self.__eta=self.__m1*self.__m2/(self.__M**2)        # Reduced mass
        self.__MC=npy.power(self.__eta,3./5.)*self.__M      # Chirp mass
        self.__rSC=2.*G*self.__MC/(c**2)                    # Schwarzchild radius of chirp
        self.__tSC=self.__rSC/c                             # Schwarzchild time of chirp
        self.__fisco=c**3/(6*npy.pi*npy.sqrt(6)*G*self.__M) # Innermost closest approach freq
        
        
        # Below we compute some relevant values for the EM approach
        
        # Chirp duration between fDmin and coalescence
        # We increase the vector size if larger than Tsample
      
        self.__Tchirp=max(self.getTchirp(self.__fDmin/2),self.__Tsample)
        
        # Duration in the detector acceptance (<Tchirp)
        self.__Tchirpd=self.getTchirp(self.__fDmind/2)
    
        # The length difference is not in the det acceptance
        # and can therefore be used to compute the blackman window
            
        self.__Tblack=self.__Tchirp-self.__Tchirpd  # End of Blackman window
        self.__Tblack_start=0                       # Start of Blackman window
    
        # Values for EOB templates are different here
        # We use SEOBnr approximant via pyCBC
        # https://pycbc.org/pycbc/latest/html/pycbc.waveform.html#pycbc.waveform.waveform.get_td_waveform
        #
        
        if self.__type==0:
            
            # Here we have the correct values
            self.__Tchirp=self.__Tsample
            
            self.__Tchirpd=self.__Tchirp-0.05 # Apply blackman window to the last 50ms of the frame
    
            # Blackman window is defined differently here
            # Because there is some signal after the merger for those templates
            
            self.__Tblack=1. # First second by default
            self.__Tblack_start=0. # Will be update on demand


        # The total length of signal to produce (and total num of samples)
        self.__Ttot=self.__Tchirp
        N=int(self.__Ttot*self.__fe)
        self.__N=N+N%2

        self.__TFnorm=npy.sqrt(self.__N)
        self.__Ttot=float(self.__N)/float(self.__fe) # Total time produced
        self.__TchirpAndTdepass=self.__Ttot
        self.__delta_f=self.__fe/self.__N
        self.__tc=self.__Ttot         # Where we put the end of template generated
        
        # Vectors
        
        self.__T=npy.arange(self.__N)*self.__delta_t
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f
        self.__St=npy.zeros(self.__N)
        self.__Stquad=npy.zeros(self.__N)
        self.__Stinit=npy.zeros(self.__N)
        self.__Sf=npy.zeros(self.__N,dtype=complex)
        self.__Sfn=npy.zeros(self.__N,dtype=complex)
        self.__Filtf=npy.zeros(self.__N,dtype=complex)
        self.__Filt=npy.zeros(self.__N)
        self.__norm=1.
        self.__Stfreqs=npy.arange(len(self.__Sf))*self.__delta_f
        
        if self.__verb:
            print("____________________________")
            print("Template generation")
            print("____________________________")
            print("")
            print(f"We will produce the signal for a CBC with masses ({m1},{m2})")
            print(f"Template type is {self.__kindTemplate}")
            print(f"Total length of signal produced is {self.__Ttot:.1f}s")
            print(f"Chirp duration in the det. acceptance is {(self.__Tchirp-self.__Tblack):.2f}s")
            print(f"Coalescence is at t={(self.__Ttot-self.__Tdepass):.2f}s")
            print(f"So the signal will enter detector acceptance at t={self.__Tblack:.2f}s")
            
            
    '''
    Template 3/10
    
    Produce the full template signal in the time-domain
    '''

    def _genStFromParams(self):

        itmin=0
        itmax=int(self.__tc/self.__delta_t)

        if self.__kindTemplate=='EM':
        
            # Simple approach, generate signal between 0 and tc, and then add some zeroes
            self.__St[:]= npy.concatenate((self.h(self.__T[itmin:itmax]),npy.zeros(self.__N-itmax)))
            self.__Stquad[:]= npy.concatenate((self.hq(self.__T[itmin:itmax]),npy.zeros(self.__N-itmax)))
            self.__Stinit[:]= npy.concatenate((self.h(self.__T[itmin:itmax]),npy.zeros(self.__N-itmax)))
            
        elif self.__kindTemplate=='EOB':

            m1=self.__m1/Msol 
            m2=self.__m2/Msol

            # Trick to avoid template gen issue with SEOBNR
            # One need to compute the ringdown frequency first

            frd=2.*f_FRD(m1,m2)
            d=5
            while 2**d<frd:
                d+=1

            #print(m1,m2,frd)
            fs_tmp=2**d
            if fs_tmp/frd<1.5:
                fs_tmp=2**(d+1)
            nyq=self.__fe/2
            fetmp=self.__fe
            ratio=1.
            if fs_tmp>nyq/2:
                #print("Need to upsample because of ringdown freq")
                fetmp=2*fs_tmp
                ratio=fetmp/self.__fe

            #print(m1,m2,ratio,self.__fDmin)
            if self.__m1/Msol+self.__m2/Msol>=4.:
                hp,hq = get_td_waveform(coa_phase=self.__Phic,approximant='SEOBNRv4_opt', mass1=self.__m1/Msol,mass2=self.__m2/Msol,spin1z=self.__s1z,spin2z=self.__s2z,delta_t=1./fetmp,f_lower=self.__fDmin)

                # Resample on the fly
                if (ratio>1):
                    #print(f"Resample the signal by {ratio}")
                    hp=hp[::int(ratio)]
                    hq=hq[::int(ratio)]
            else:
                hp,hq = get_td_waveform(coa_phase=self.__Phic,approximant='SpinTaylorT4', mass1=self.__m1/Msol,mass2=self.__m2/Msol,spin1z=self.__s1z,spin2z=self.__s2z,delta_t=self.__delta_t,f_lower=self.__fDmin)
            
            
            
            f = utils.frequency_from_polarizations(hp, hq)

            limit=0.01*npy.max(npy.abs(npy.asarray(hp))) # Look for 99% amplitude drop after coalescence
            c1=0
            for c1 in range(len(hp)-1,-1,-1): # Don't consider 0 at the end
                if abs(hp.numpy()[c1])>limit:
                    break
            hp_tab=hp.numpy()[:c1]
            freqs=f.numpy()[:c1]
           
            #print(time.time()-start_time)
            #print('Frame lengths',len(self.__St),len(hp_tab))
            itmin=max(0,len(self.__St)-len(hp_tab))
            if itmin==0: # Template is not fully contained, truncate it
                self.__St[:]= hp_tab[len(hp_tab)-len(self.__St):]
                self.__Stfreqs[:]= freqs[len(hp_tab)-len(self.__St):]
            else:        # Template is fully contained, pad St
                self.__St[:]= npy.concatenate((npy.zeros(itmin),hp_tab))
                self.__Stfreqs[:]= npy.concatenate((npy.zeros(itmin),freqs))
                self.__Tblack_start=itmin*self.__delta_t
                self.__Tblack=(itmin+100)*self.__delta_t
            del hp,hq,hp_tab
        elif self.__kindTemplate=='IMRPhenomTPHM':
                
            # The signal starting at frequency fDmin
            hp,hq = get_td_waveform(coa_phase=self.__Phic,approximant='IMRPhenomTPHM', mass1=self.__m1/Msol,mass2=self.__m2/Msol,spin1z=self.__s1z,spin2z=self.__s2z,spin2x=self.__s2x,spin2y=self.__s2y,spin1x=self.__s1x,spin1y=self.__s1y,delta_t=self.__delta_t,f_lower=self.__fDmin)
            f = utils.frequency_from_polarizations(hp, hq)

            limit=0.01*npy.max(npy.abs(npy.asarray(hp))) # Look for 99% amplitude drop after coalescence

            for c1 in range(len(hp)-1,-1,-1): # Don't consider info after coalescence
                if abs(hp.numpy()[c1])>limit:
                    #print(c1,len(hp))
                    break
            hp_tab=hp.numpy()[:c1]
            freqs=f.numpy()[:c1]
           
            
            itmin=max(0,len(self.__St)-len(hp_tab))
            if itmin==0: # Template is not fully contained, truncate it
                self.__St[:]= hp_tab[len(hp_tab)-len(self.__St):]
                self.__Stfreqs[:]= freqs[len(hp_tab)-len(self.__St):]
            else:        # Template is fully contained, pad St
                self.__St[:]= npy.concatenate((npy.zeros(itmin),hp_tab))
                self.__Stfreqs[:]= npy.concatenate((npy.zeros(itmin),freqs))
                self.__Tblack_start=itmin*self.__delta_t
                self.__Tblack=(itmin+100)*self.__delta_t
            del hp,hq,hp_tab
        else:
            raise ValueError("Valeur pour kindTemplate non prise en charge")

        
    '''
    Template 4/10
    
    Express the signal in frequency-domain
    
    We add screening at the beginning and at the end (for EM only)
    in order to avoid discontinuity
    
    Screening is obtained with a Blackman window:
    https://numpy.org/doc/stable/reference/generated/numpy.blackman.html
    '''
    
    def _genSfFromSt(self):
        S=npy.zeros(self.__N)
        S[:]=self.__St[:]
 
        # Blackman window at low freq
        # Compute the bins where the signal will be screend to avoid discontinuity
        # Should be between fDmin and fDmind
        #
        #print(self.__Tblack,self.__Tblack_start)
        iwmax=int(self.__Tblack/self.__delta_t)
        iwmin=int(self.__Tblack_start/self.__delta_t)

        if self.__verb:
            print("Producing frequency spectrum from temporal signal")
            print(f"To avoid artifacts, blackman window will be applied to the signal start between times {(iwmin*self.__delta_t):.2f} and {(iwmax*self.__delta_t):.2f}")
 
        S[iwmin:iwmax]*=npy.blackman((iwmax-iwmin)*2)[:iwmax-iwmin]
         
        # Blackman window at low freq (EM only)
        #
        
        if self.__kindTemplate=='EM':
            weight=10
            iwmin=int(self.__tc/self.__delta_t)-weight
            iwmax=int(self.__tc/self.__delta_t)+weight
            S[iwmin:iwmax]*=npy.blackman((iwmax-iwmin)*2)[iwmin-iwmax:]
        
        # Finally compute the FFT of the screened signal
        # We use binning dependent params here, Sf value defined this way does
        # depend on frequency, as the norm factor is 1/sqrt(N) only
        #
        self.__Stblack=S

        w = npy.arange(len(self.__St))
        if self.__verb:
            plt.plot(w, self.__St)
            plt.plot(w, self.__Stblack)
            plt.grid()
            plt.show()
        self.__Sf[:]=npy.fft.fft(S,norm='ortho')
        
        # Normalized to be coherent w/PSD def (the option ortho is doing a special normalization which is
        # fine only if you do the invert tranform afterward)
        #self.__Sfn[:]=self.__Sf[:]/(npy.sqrt(len(S))*self.__delta_f)
        self.__Sfn[:]=npy.fft.fft(S,norm='forward')/self.__delta_f
        del S
    
    '''
    Template 5/10
    
    Compute rhoOpt, which is the output we would get when filtering the template with noise only
    In other words this corresponds to the filtered power we should get in absence of signal
    
    The frequency range here is in principle the detector one
    But one could also use full frequency range (actually this is a good question ;-) )
    
    Often called the optimal SNR or SNRmax, which is kind of misleading
    
    '''
      
    def rhoOpt(self,Noise):  
        #print("A")   
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmind/self.__delta_f)
        ifmin2=int(20/self.__delta_f)
        if self.__verb:
            print(f"Calculations will be done in frequency range ({self.__fDmind},{min(self.__fDmaxd,self.__fe/2)})")
        
        # Evolution of the time w.r.t of the chirp frequency,
        # between fmind and fmaxd, ie the detector acceptance
        # This is w.r.t to Tchirp=Ttot, so numbers can be negative at start
        # if Ttot is lower than time in the detector acceptance
            
        freqs=npy.arange(len(self.__Sf))*self.__delta_f
        #print(len(freqs),ifmin,ifmax)
        # <x**2(t)> calculation
        # Sfn and PSD have the same normalisation
        #
        # !!! PSD=Sn/2 !!!

        '''
        f = open("datasim.txt", "a")
        for i in range(ifmin,ifmax):
            #f.write(f"{i*self.__delta_f}/{npy.sqrt((self.__Sfn[i]*npy.conjugate(self.__Sfn[i])).real)}/{1/(Noise.PSD[i])}\n")
            f.write(f"{i*self.__delta_f} / {npy.sqrt((self.__Sfn[i]*npy.conjugate(self.__Sfn[i])).real)} / {npy.sqrt(2*self.__delta_f*npy.sum(self.__Sfn[ifmin:i]*npy.conjugate(self.__Sfn[ifmin:i])/(Noise.PSD[ifmin:i])).real):.2f} / {1/(2*Noise.PSD[i])}\n")
        f.close()
        
        f = open("datasim2.txt", "a")
        for i in range(ifmin2,ifmax):
            #f.write(f"{i*self.__delta_f}/{npy.sqrt((self.__Sfn[i]*npy.conjugate(self.__Sfn[i])).real)}/{1/(Noise.PSD[i])}\n")
            f.write(f"{i*self.__delta_f} / {npy.sqrt((self.__Sfn[i]*npy.conjugate(self.__Sfn[i])).real)} / {npy.sqrt(2*self.__delta_f*npy.sum(self.__Sfn[ifmin2:i]*npy.conjugate(self.__Sfn[ifmin2:i])/(Noise.PSDloc[ifmin2:i])).real):.2f} / {npy.sqrt(2*Noise.PSDloc[i])}\n")
        f.close()
        '''
        #print(Noise.PSDloc[590:605],Noise.PSD[590:605])
        #check=npy.average((Noise.PSDloc[ifmin:ifmax])/(Noise.PSD[ifmin:ifmax])).real


        #plt.plot(self.__F[ifmin:ifmax],(Noise.PSD[ifmin:ifmax]/Noise.PSDloc[ifmin:ifmax]).real,'-',label='Sn(f)')
        #plt.hist((Noise.PSDloc[ifmin:ifmax]/Noise.PSD[ifmin:ifmax]).real,bins=100,label='Sn(f)')
        #plt.xlabel('f (Hz)')
        #plt.show()


        #ropt=2*self.__delta_f*npy.sum((1/(1e42*Noise.PSD[ifmin:ifmax])-1/(1e42*Noise.PSDloc[ifmin:ifmax])).real)
        #noisy=(2*self.__delta_f*npy.sum(1/(Noise.PSD[ifmin:ifmax])).real)
        #signaly=self.__delta_f*npy.sum(self.__Sfn[ifmin:ifmax]*npy.conjugate(self.__Sfn[ifmin:ifmax])).real
        #noisy=(self.__delta_f*npy.sum(1/(2*Noise.PSDloc[ifmin:ifmax])).real)
        #signaly=self.__delta_f*npy.sum(self.__Sfn[ifmin:ifmax]*npy.conjugate(self.__Sfn[ifmin:ifmax])).real

        #print(tmpl,self.__delta_f*npy.sum(1/(2*Noise.PSD[ifmin:ifmax])).real,self.__delta_f*npy.sum(1/(2*Noise.PSDloc[ifmin:ifmax])).real)

        #print(ropt,ropt1)
        #ropt2=npy.sqrt(2*self.__delta_f*npy.sum((self.__Sfn**2).real/(Noise.PSDloc)).real)
        ropt=npy.sqrt(2*self.__delta_f*npy.sum(self.__Sfn[ifmin:ifmax]*npy.conjugate(self.__Sfn[ifmin:ifmax])/(Noise.PSDloc[ifmin:ifmax])).real)
        
        #print(noisy,signaly,ropt)
        #ropt1=npy.sqrt(2*self.__delta_f*npy.sum(self.__Sfn[ifmin:ifmax]*npy.conjugate(self.__Sfn[ifmin:ifmax])/(Noise.PSD[ifmin:ifmax])).real)
        #ropt1=npy.sqrt(2*npy.trapz(self.__Sfn[ifmin:ifmax]*npy.conjugate(self.__Sfn[ifmin:ifmax])/(Noise.PSD[ifmin:ifmax]),freqs[ifmin:ifmax]).real)
        #
        # Definition of the SNRmax used here is available in Eqn 26 of the foll. paper:
        #
        # https://arxiv.org/pdf/gr-qc/9402014.pdf
        #
        # !!! This value is not depending on Topt or fe !!!
        
        #print(check,ropt,ropt1,ropt2)
        if self.__verb:
            print(f'MF output value when template is filtered by noise (No angular or antenna effects, D=1Mpc) over the total period is equal to {ropt:.2f}',self.__delta_f)
        self.__norm=ropt
 
        return self.__norm

    '''
    Template 6/10
    
    Then we normalize the signal, first with PSD (like the noise)
    Then with rhoOpt (to get SNR=1)
    With this method to get a signal at a given SNR one just have to rescale it by a factor SNR
    
    It's important to have the same normalization everywhere otherwise it's a mess
    '''

    def _whitening(self,kindPSD,Tsample,norm):
        if kindPSD is None:
            print('No PSD given, one cannot normalize!!!')
            self.__St[:]=npy.fft.ifft(self.__Sf,norm='ortho').real
            return self.__St

        # Important point, noise and signal are produced over the same length, it prevent binning pbs
        # We just produce the PSD here, to do the weighting
        rho=1.


        self.__Noise.getNewSample() # To update the SNRmax value

        #print(self.__Noise.PSDloc[1000:1005],self.__Noise.PSD[1000:1005])
        

        if norm:
            rho=self.rhoOpt(Noise=self.__Noise)         # Get SNRopt
        
        Sf=npy.zeros(self.__N,dtype=complex)
        Sf=self.__Sf/npy.sqrt(self.__Noise.PSD*self.__N*self.__delta_f)     # FD Whitening of the signal
        # Need to take care of the fact that PSD has not the normalization of Sf.
        #self.__Sfn=Sf                        # Signal whitened
      
        # The withened and normalized signal in time domain
        #
        
        iwmax=int(self.__Tblack/self.__delta_t)

        self.__St[iwmax:]=npy.fft.ifft(Sf,norm='ortho').real[iwmax:]/rho
        
    '''
    TEMPLATE 7/10
    
    Signal resampling
    '''
    
    def _resample(self,signal):
    
        S=[]
        cutlimit=[]
        vmax=len(signal)

        for j in range(self.__nTsample):  # Loop over subsamples
            decimate=int(self.__fe/self.__listfe[j])
            N=int(self.__listTsample[j]*self.__fe)  # Length of the subsample
            T=npy.arange(N)*self.__delta_t
            cutlimit.append(vmax-len(T))
            if cutlimit[j] > len(signal):
                raise ValueError("La valeur de début de cut ne doit pas être en dehors du vecteur cut")
            chunk=signal[cutlimit[j]:vmax]
            # Apply some kind of zero-suppression here
            if (not chunk.any()):
                #print("Empty chunk")
                S.append(0)
            else:
                skimmedchunk=chunk[::int(decimate)]
                
                skimmedchunk=npy.trim_zeros(skimmedchunk, 'f')
                
                if (not skimmedchunk.any()): # Rare but could happen
                    #print("Empty chunk after resample")
                    S.append(0)
                    continue
                #'''
                if (skimmedchunk[0]<1e-10):
                    for c1 in range(len(skimmedchunk)):
                        if skimmedchunk[c1]>1e-10:
                            break
                    skimmedchunk=skimmedchunk[c1:]
                #'''
                S.append(skimmedchunk)
                
            vmax=cutlimit[j]
        return S
    
    '''
    TEMPLATE 8/10
    
    The main template generation macro
    '''
    
    def getNewSample(self,kindPSD='flat',Tsample=1,tc=None,norm=True):
        

        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
    
        if kindPSD!='flat' and kindPSD!='analytic' and kindPSD!='realistic' and kindPSD!=None:
            raise ValueError("Les seules valeurs autorisées sont None, 'flat', et 'analytic'")
        
        if not(isinstance(norm,bool)):
            raise TypeError("Un booléen est attendu")
        
        # The signal is produced along time, no normalization yet
        self._genStFromParams()

        # Go in the frequency domain
        if self.__whiten!=-10:
            self._genSfFromSt()

        # Whiten the signal and normalize it to SNR=1
        if self.__whiten!=-10:
            self._whitening(kindPSD,Tsample,norm)
        
        # Resample it if needed
        tc= self.__Tchirp if tc==None else min(tc,self.__Tchirp)

        # Here is the number of points of the required frame
        N=int(Tsample*self.__fe)
        #N=N+N%2
        S=npy.zeros(N)
        F=npy.zeros(N)


        itc=int(tc/self.__delta_t)
        #print(itc,len(S),len(self.__St),Tsample,tc,self.__Ttot)
        if tc<=self.__Ttot:
            S[:itc]=self.__St[-itc:] # There will be 0s at the start
            F[:itc]=self.__Stfreqs[-itc:] # There will be 0s at the start
        else:
            S[itc-self.__N:itc]=self.__St[:] # There will be 0s at the end
            F[itc-self.__N:itc]=self.__Stfreqs[:] # There will be 0s at the end

        if self.__nTsample > 1:
            return self._resample(S),self._resample(F)
        else :
            listS=[]
            listS.append(S)
            listF=[]
            listF.append(F)
            return listS,listF
          

    '''
    TEMPLATE 9/10
    
    Produce a new template with a different value of tc (useful for training sample prod)
    '''
    
    def getSameSample(self,Tsample=1,tc=None):
        
        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
    
        tc= self.__Ttot-self.__Tdepass if tc==None else min(tc,self.__Tchirp)
        N=int(Tsample*self.__fe)
        N=N+N%2
        S=npy.zeros(N)

        itc=int(tc/self.__delta_t)
        if tc<=self.__Ttot:
            if len(S)!=itc:
                itc = len(S)
            S[:itc]=self.__St[-itc:]
        else:
            S[itc-self.__N:itc]=self.__St[:]
        if self.__nTsample > 1:
            return self._resample(S)
        else :
            listS=[]
            listS.append(S)
            return listS
        
        
    '''
    TEMPLATE 10/10
    
    Plot macros
    '''
    
    # The template in time domain (with a different color for the samples
    
    def plot(self,Tsample=1,tc=0.95,SNR=1):
        
        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
        
        listT=[]
        if self.__nTsample > 1:
            maxT=self.__Tsample
            for j in range(self.__nTsample):
                delta_t=1/self.__listfe[j]
                N=int(self.__listTsample[j]*self.__listfe[j])
                listT.append(npy.arange((maxT-self.__listTsample[j])/delta_t,maxT/delta_t)*delta_t)
                maxT-=self.__listTsample[j]
        else:
            N=int(Tsample*self.__fe)
            listT.append(npy.arange(N)*self.__delta_t)
        
        for j in range(self.__nTsample):
            plt.plot(npy.resize(listT[j],len(self.getSameSample(Tsample=Tsample,tc=tc)[j])), self.getSameSample(Tsample=Tsample,tc=tc)[j]*SNR,'-',label=f"signal at {self.__listfe[j]} Hz")
            
        plt.title('Template dans le domaine temporel de masses ('+str(self.__m1/Msol)+', '+str(self.__m2/Msol)+') Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('h(t)')
        plt.grid(True, which="both", ls="-")
        plt.legend()

    # 1-D signal projection
    def plotSignal1D(self):
        _, bins, _ = plt.hist(self.__St,bins=100, density=1)
        #_, bins, _ = plt.hist(npy.abs(self.__Sfn),bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__St)
        print("Largeur de la distribution temporelle (normalisée):",sigma)
        print("Valeur moyenne:",mu)
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        #plt.yscale('log')
        plt.plot(bins, best_fit_line)
        plt.title('Bruit temporel normalisé')

    # The Filter
    def plotFilt(self,Tsample=1,SNR=1):
    
        if isinstance(Tsample,list):
            Tsample=sum(Tsample)
        
        N=int(Tsample*self.__fe)
        T=npy.arange(N)*self.__delta_t
        
        plt.plot(T, self.__Filt,'-',label='filt(t)')
        plt.title('Template filtré dans le domaine temporel de masses ('+str(self.__m1/Msol)+','+str(self.__m2/Msol)+')Msolaire')
        plt.xlabel('t (s)')
        plt.ylabel('rho(t) (No Unit)')
        plt.grid(True, which="both", ls="-")

    # Fourier transform of whitened signal and noise. Signal normalized to S/N=1
    def plotTF(self):
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmin/self.__delta_f)
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.Noise.PSD[ifmin:ifmax]/self.__delta_f),'-',label='Sn(f)')
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Sf[ifmin:ifmax])/self.__norm,'.')
        plt.title('Template dans le domaine frequentiel')
        plt.xlabel('f (Hz)')
        plt.ylabel('h(f) (No Unit)')
        plt.yscale('log')
          
    # Normalized FFT of signal and SNR proportion evolution
    def plotTFn(self):
                         
        ifmax=int(min(self.__fDmaxd,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fDmind/self.__delta_f)
        
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.__F[ifmin:ifmax],npy.abs(self.__Sfn[ifmin:ifmax]),'.')
        ax2.plot(self.__F[ifmin:ifmax],self._evolSnr,'.',color='red')
        ax1.set_yscale('log')
        plt.grid(True, which="both", ls="-")

    # Evolution of the SNR vs time, divided by samples
    def plotSNRevol(self):

        props=self._rawSnr/100.
        
        fig3, ax3 = plt.subplots()
        fig3.suptitle('SNR accumulation vs template time')
        ax3.plot(self._evolSnrTime,self._Snr_vs_freq)
        ax3.set_xlabel('t')
        ax3.set_ylabel('SNR proportion collected',rotation=270)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.yaxis.set_label_coords(1.1,0.5)
        ax3.set_ylim([0., 1.])
        ymin, ymax = ax3.get_ylim()
        xmin, xmax = ax3.get_xlim()
        totprop=1.
        
        # Select the color map named rainbow
        cmap = cm.get_cmap(name='gist_rainbow')
        tstart=self.__Ttot
            
        for j in range(self.__nTsample):
            propchunk=props[j]
            tstart-=self.__listTsample[j]
            ax3.axvspan(tstart, self.__Ttot, (totprop-propchunk)*ymax, totprop*ymax, facecolor=cmap(int(256/(2*self.__nTsample))*j),alpha=0.5)
            plt.axvline(x = tstart, color = 'b', linestyle='dashed')
            totprop-=propchunk
    
    @property
    def length(self):
        return self.__N
        
    def duration(self):
        return self.__Tchirp

    def signal(self):
        return self.__St

    def filtre(self):
        return self.__Filtf
    
    def norma(self):
        return self.__norm
    
    def signal_raw(self):
        return self.__Stblack
  
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
