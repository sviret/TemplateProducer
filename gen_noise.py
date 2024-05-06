import numpy as npy
import scipy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import math
import time
import random
from scipy.stats import norm
from scipy import signal
from scipy import ndimage
from scipy.interpolate import interp1d
import matplotlib.cm as cm


######################################################################################################################################################################################
parameters = {'font.size': 15,'axes.labelsize': 15,'axes.titlesize': 15,'figure.titlesize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'legend.fontsize': 15,'legend.title_fontsize': 15,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [10,5]}
plt.rcParams.update(parameters)


'''
Class handling noise generation

Option:

->Ttot     : noise samples duration, in seconds (Default is 1)
->fe       : sampling frequencies, in Hz (Default is 2048)
->kindPSD  : noise type: 'flat', 'analytic', or 'realistic' (Default is g. flat).
             Noise is stationary in both cases (see below)
->fmin     : minimal frequency for noise definition (Def = 20Hz)
->fmax     : maximal frequency for noise definition (Def = 1500Hz)
->nsamp    : for each frequency the PSD is considered as a gaussian centered on PSD
             of width PSD/sqrt(nsamp), number of samples used to estimate the PSD
->whitening: type of signal whitening:
                0: No whitening
                1: Frequency-domain whitening (Default standard procedure)
                2: Time-domain whithening (Zero latency, as described in https://dcc.ligo.org/public/0141/P1700094/005/main_v5.pdf)
            

flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution
            still it's not based on real data, ie not including glitches for example.
            So that's gaussian colored noise
            Reference used for analytic noise is cited below
realistic is based on real data, and is non-stationary

Noise is produced over a given frequency range.

Indeed there is no reason to produce noise well outside
detector acceptance

'''

class GenNoise:

    def __init__(self,Ttot=1,fe=2048,kindPSD='flat',nsamp=160,fmin=20,fmax=1500,whitening=1,customPSD=None,verbose=False):

        if not((isinstance(Ttot,int) or isinstance(Ttot,float) or isinstance(Ttot,list)) and (isinstance(fe,int) or isinstance(fe,float) or isinstance(fe,list))):
            raise TypeError("Ttot et fe doivent être des ints, des floats, ou des list")
        if not(isinstance(kindPSD,str)):
            raise TypeError("kindPSD doit être de type str")
        if kindPSD!='flat' and kindPSD!='analytic' and kindPSD!='realistic':
            raise ValueError("Les seules valeurs autorisées pour kindPSD sont 'flat','analytic', et 'realistic'")

        # Deal with the fact that we can sample the frame with different frequencies
        if isinstance(Ttot,list):
            if not isinstance(fe,list):
                raise TypeError("Ttot et fe doivent être du même type")
            elif not len(Ttot)==len(fe):
                raise ValueError("Les list Ttot et fe doivent faire la même taille")
            else:
                self.__listTtot=Ttot           # List of chunk lengths
                self.__listfe=fe               # List of corresponding sampling freqs
                self.__Ttot=sum(Ttot)          # Total sample length
                self.__fe=max(fe)              # Max sampling freq
                self.__nTtot=len(Ttot)         # Total number of subsamples
        else:
            self.__Ttot=Ttot                   # Total sample length
            self.__fe=fe                       # Sampling freq
            self.__nTtot=1
        
        # We will generate a sample with the total length and the max sampling freq, and resample
        # only at the end
        
        self.__whiten=whitening
        self.__verb=verbose
        self.__fmin=fmin
        self.__fmax=fmax
        self.__N=int(self.__Ttot*self.__fe)    # The total number of time steps produced
        self.__delta_t=1/self.__fe             # Time step
        self.__delta_f=self.__fe/self.__N      # Frequency step
        self.__kindPSD=kindPSD                 # PSD type
        self.__nsamp=nsamp                     # Noise width per freq step.

        # N being defined we can generate all the necessary vectors
        
        self.__norm=npy.sqrt(self.__N)

        self.__T=npy.arange(self.__N)*self.__delta_t  # Time values
        
        # Frequencies (FFT-friendly)
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f
        self.__Fnorm=self.__F/(self.__fe/2.)
        
        self.__PSD=npy.ones(self.__N, dtype=npy.float64) # Setting the PSD to one means infinite noise
        self.__invPSD=npy.ones(self.__N, dtype=npy.float64)

        # Then we produce the PSD which will be use to generate the noise in freq range.

        self.__Nf=npy.zeros(self.__N,dtype=complex)          # Noise FFT
        self.__Nf2=npy.zeros(self.__N,dtype=complex)         # Noise FFT (whitened if required)
        self.__Nfr=npy.zeros(self.__N, dtype=npy.float64)    # Noise FFT real part
        self.__Nfi=npy.zeros(self.__N, dtype=npy.float64)    # Noise FFT imaginary part
        self.__custom=[]

        if kindPSD=='realistic' and len(customPSD)>0:
            self.__custom=customPSD
            nPSDs=len(customPSD)
            rk=random.randint(0,nPSDs-1)
            #print(rk)
            self._brutePSD=customPSD[rk]
            factor=float((self.__N//2+1)/len(self._brutePSD))
            self.__realPSD = npy.abs(ndimage.zoom(self._brutePSD,factor))
    
        self._genPSD()
            
        if self.__verb:
            print("____________________________")
            print("Noise generation")
            print("____________________________")

        
    '''
    NOISE 1/8
    
    Noise generation for analytic option, account for shot, thermal, quantum and seismic noises
    This is the one sided PSD, as defined in part IV.A of:
    https://arxiv.org/pdf/gr-qc/9301003.pdf
    '''

    def Sh(self,f):

        ##Shot noise (Eq 4.1)
        hbar=1.05457182e-34 #m2kg/s
        lamda=5139e-10 #m
        etaI0=60 #W
        Asq=2e-5
        L=4e3 #m
        fc=100 #Hz
        
        Sshot=(hbar*lamda/etaI0)*(Asq/L)*fc*(1+(f/fc)**2)
        
        ##Thermal Noise (Eq 4.2 to 4.4)
        kb=1.380649e-23 #J/K
        T=300 #K
        f0=1 #Hz
        m=1000 #kg
        Q0=1e9
        Lsq=L**2
        fint=5e3 #Hz
        Qint=1e5
        Spend=kb*T*f0/(2*(npy.pi**3)*m*Q0*Lsq*((f**2-f0**2)**2+(f*f0/Q0)**2))
        Sint=2*kb*T*fint/((npy.pi**3)*m*Qint*Lsq*((f**2-fint**2)**2+(f*fint/Qint)**2))
    
        Sthermal=4*Spend+Sint
        
        #Seismic Noise (Eq 4.6)
        S0prime=1e-20 #Hz**23
        f0=1 #Hz
        with npy.errstate(divide='ignore'):
            Sseismic=npy.where((f!=f0) & (f!=0),S0prime*npy.power(f,-4)/(f**2-f0**2)**10,(1e-11)**2)
            
        #Quantum noise (Eq 4.8)
        with npy.errstate(divide='ignore'):
            Squant=npy.where((f!=0),8*hbar/(m*Lsq*(2*npy.pi*f)**2),(1e-11)**2)
        
        return (Squant+Sshot+Sthermal+Sseismic)

    '''
    NOISE 2/8
    
    Produce the PSD, so in the noise power in frequency domain
    We don't normalize it, so be carefuf if the signal is not whitened
    '''

    def _genPSD(self):

        # Frequency range for the PSD
        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)
        
        # Generate the function
        if self.__kindPSD=='flat':
            sigma=2e-23
            self.__PSD[ifmin:ifmax]=sigma**2
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2] # Double sided
        elif self.__kindPSD=='analytic':
            self.__PSD[ifmin:ifmax]=self.Sh(abs(self.__F[ifmin:ifmax]))
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2]
        elif self.__kindPSD=='realistic':
            self.__PSD[ifmin:ifmax]=self.__realPSD[ifmin:ifmax]
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2]


        # Prepare the time-domain whitening filters

        self.__invPSD=npy.sqrt(1./npy.abs(self.__PSD))
        #print(self.__invPSD)
        freqs=self.__Fnorm[0:self.__N//2]
        ampl=self.__invPSD[0:self.__N//2]
        for i in range(len(ampl)):
            if ampl[i]<=1.5:
                ampl[i]=0.
        freqs[self.__N//2-1]=1.
        ampl[self.__N//2-1]=0.
        self.__nf=ampl.max()
        normamp=ampl/ampl.max()
        self.__whitener = signal.firwin2(self.__N//2, freqs, normamp)
        self.__whitener_MP = signal.minimum_phase(signal.firwin2(self.__N//2, freqs, normamp**2), method='hilbert',n_fft=10*self.__N)


        if self.__verb:

            tmp = npy.zeros(len(self.__whitener))
            tmp[:len(self.__whitener_MP)]=self.__whitener_MP
            #print(len(self.__whitener_MP),len(self.__whitener))
        
            w, h = signal.freqz(self.__whitener)
            w2, h2 = signal.freqz(self.__whitener_MP)

            plt.title('Digital filter frequency response')
            plt.plot(w/npy.pi, npy.abs(h))
            plt.plot(freqs, normamp)
            plt.plot(w2/npy.pi, npy.abs(h2))
            plt.title('Digital filter frequency response')
            plt.ylabel('Amplitude Response')
            plt.xlabel('Frequency (rad/sample)')
            plt.grid()
            plt.show()
            plt.title('Digital filter frequency response')
            plt.plot(w, npy.unwrap(npy.angle(h,deg=True)))
            plt.plot(w2, npy.unwrap(npy.angle(h2,deg=True)))
            plt.title('Digital filter frequency response')
            plt.ylabel('Amplitude Response')
            plt.xlabel('Frequency (rad/sample)')
            plt.grid()
            plt.show()
        
        
    '''
    NOISE 3/8
    
    PSD type change
    '''

    def changePSD(self,kindPSD):

        del self.__PSD
        self.__PSD=npy.ones(self.__N, dtype=npy.float64)
        if kindPSD!='flat' and kindPSD!='analytic' and kindPSD!='realistic':
            raise ValueError("Les seules valeurs autorisées sont 'flat' et 'analytic'")
        self.__kindPSD=kindPSD
        
        if kindPSD=='realistic' and len(self.__custom)>0:
            nPSDs=len(self.__custom)
            rk=random.randint(0,nPSDs-1)
            #print(nPSDs,rk)
            self._brutePSD=self.__custom[rk]
            #print(self.__N//2+1,len(self._brutePSD))
            factor=float((self.__N//2+1)/len(self._brutePSD))
            self.__realPSD = npy.abs(ndimage.zoom(self._brutePSD,factor))
    
        self._genPSD()


    '''
    NOISE 4/8
    
    Create the noise in freq domain from the PSD

    We start from a PSD which provides us the power of the noise at a given frequency
    In order to create a noise ralisation, we need first to generate a random noise realisation of the noise
    in the frequency domain

    For each frequency we choose a random value of the power centered on the PSD value, we consider
    that power distribution is gaussian with a width equal to power/4 (rule of thumb, could be improved)

    Then when the power is chosen we choose a random starting phase Phi0 in order to make sure that the
    frequency component is fully randomized.

    Nf is filled like that:

    a[0] should contain the zero frequency term,
    a[1:n//2] should contain the positive-frequency terms,
    a[n//2+1:] should contain the negative-frequency terms,
    
    PSD and Nf(f)**2 are centered around PSD(f)

    '''
 
    def _genNfFromPSD(self):

        # The power at a given frequency is taken around the corresponding PSD value
        # We produce over the full frequency range
        self.__Nfr[0:self.__N//2+1]=npy.random.normal(npy.sqrt(self.__PSD[0:self.__N//2+1]),npy.sqrt(self.__PSD[0:self.__N//2+1]/self.__nsamp))
        self.__Nfi[0:self.__N//2+1]=self.__Nfr[0:self.__N//2+1]

        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)
        
        # The initial phase is randomized
        # randn provides a nuber following a normal law centered on 0 with sigma=1, so increase a
        # bit to make sure you cover all angular values (thus the factor 100)
        
        phi0=100*npy.random.randn(len(self.__Nfr))
        self.__Nfr*=npy.cos(phi0)
        self.__Nfi*=npy.sin(phi0)
        
        # Brutal filter
        self.__Nfr[ifmax:]=0.
        self.__Nfi[ifmax:]=0.
        self.__Nfr[:ifmin]=0.
        self.__Nfi[:ifmin]=0.
 
        # Then we can define the components
        self.__Nf[0:self.__N//2+1].real=self.__Nfr[0:self.__N//2+1]
        self.__Nf[0:self.__N//2+1].imag=self.__Nfi[0:self.__N//2+1]
        self.__Nf[-1:-self.__N//2:-1]=npy.conjugate(self.__Nf[1:self.__N//2])


    '''
    NOISE 5/8
    
    Get noise signal in time domain from signal in frequency domain (inverse FFT)

    https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html

    If whitening option is set to true signal is normalized. The whitened noise should be a gaussian centered on 0 and of width 1.

    '''

    def _genNtFromNf(self):

        self.__Nt=[]                       # Noise in time domain
        
        for j in range(self.__nTtot):      # one array per chunk
            self.__Nt.append([])
        
        # Inverse FFT over the total length (0/1)
        if self.__whiten==0:
            self.__Nt[0] = npy.fft.ifft(self.__Nf[:],norm='ortho').real
        elif self.__whiten==1:
            self.__Nt[0] = npy.fft.ifft(self.__Nf[:]/npy.sqrt(self.__PSD),norm='ortho').real
        elif self.__whiten==2:
            #Run the Zero-L FIR filter
            self.__Nt[0] = self.__nf*signal.lfilter(self.__whitener_MP,1,npy.fft.ifft(self.__Nf[:],norm='ortho').real)
      
        
        self.__Ntraw = self.__Nt[0].copy()
        self.__Nf2=npy.fft.fft(self.__Nt[0],norm='ortho') # Control
  
        #Run the main FIR filter
        #self.__tfilt = self.__nf*signal.lfilter(self.__whitener,1,npy.fft.ifft(self.__Nf[:],norm='ortho').real)


    '''
    NOISE 6/8
    
    Signal resampling
    '''

    def _resample(self):

        Ntref=self.__Nt[0] # The temporal realization at max sampling
                    
        if self.__verb:
            print("Signal Nt has frequency",self.__fe,"and duration",self.__Ttot,"second(s)")

        for j in range(self.__nTtot):
            if self.__verb:
                print("Chunk",j,"has frequency",self.__listfe[j],"and covers",self.__listTtot[j],"second(s)")
            
            #Pick up the data chunk
            ndatapts=int(self.__listTtot[j]*self.__fe)
            nttt=len(Ntref)
            Nt=Ntref[-ndatapts:]
            Ntref=Ntref[:nttt-ndatapts]
            decimate=int(self.__fe/self.__listfe[j])
            self.__Nt[j]=Nt[::int(decimate)]
    
    
    '''
    NOISE 7/8
    
    The full procedure to produce a noise sample once the noise object has been instantiated
    '''

    def getNewSample(self):

        if self.__kindPSD=='realistic':
            self.changePSD('realistic')

        self._genNfFromPSD()               # Noise realisation in frequency domain
        self._genNtFromNf()                # Noise realisation in time domain
        if self.__nTtot > 1:               # If requested, resample the data
            self._resample()
        return self.__Nt.copy()
   
   
    '''
    NOISE 8/8
    
    Plot macros and getters
    '''

    # The main plot (noise in time domain)
    def plotNoise(self):

        listT=[] # Time of the samples accounting for the diffrent freqs
        if self.__nTtot > 1:
            maxT=self.__Ttot
            for j in range(self.__nTtot):
                delta_t=1/self.__listfe[j]
                N=int(self.__listTtot[j]*self.__listfe[j])
                listT.append(npy.arange((maxT-self.__listTtot[j])/delta_t,maxT/delta_t)*delta_t)
                maxT-=self.__listTtot[j]
        else:
            listT.append(self.__T)
            
        
        for j in range(self.__nTtot):
            plt.plot(listT[j], self.__Nt[j],'-',label=f"noise at {self.__listfe[j]} Hz")

        plt.xlabel('t (s)')
        plt.ylabel('h(t)')
        plt.grid(True, which="both", ls="-")
        plt.legend()

    def plotNoiseTW(self):

        npts=len(self.__whitener)
        npts2=len(self.__whitener_MP)

        middle=int(npts/2)

        listT  = npy.arange(npts)*self.__delta_t-(npts/2)*self.__delta_t
        listT2 = npy.arange(npts2)*self.__delta_t
        
        plot1 = plt.subplot2grid((2, 2), (0, 0))
        plot2 = plt.subplot2grid((2, 2), (0, 1))
        plot3 = plt.subplot2grid((2, 2), (1, 0))
        plot4 = plt.subplot2grid((2, 2), (1, 1))

        plot1.plot(listT, self.__whitener,'-')
        plot1.set_title('Normal whitening filter')
        plot2.plot(listT2, self.__whitener_MP,'-')
        plot2.set_title('Low-latency whitening filter')
        plot3.plot(listT[middle-500:middle+500], self.__whitener[middle-500:middle+500],'-')
        plot4.plot(listT2[0:500], self.__whitener_MP[0:500],'-')

        plt.tight_layout()
        plt.show()


    # The 1D projection (useful to check that noise has been correctly whitened
    def plotNoise1D(self,band):

        print("Freq band:",self.__listfe[band],"Hz")
        
        _, bins, _ = plt.hist(self.__Nt[band],bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__Nt[band])
        print("Freq domain whitening:")
        print(f"Width: {sigma}")
        print(f"Mean: {mu}")
        
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, best_fit_line)
        plt.title(f'Time domain noise 1D projection for band at {self.__listfe[band]}Hz')

    # Frequency domain
    def plotTF(self,fmin=None,fmax=None):

        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf[ifmin:ifmax]),'.',label='n_tilde(f)')
        plt.title('Noise realisation in frequency domain')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
             
    # Frequency domain whithened

    def plotTF2(self,fmin=None,fmax=None):

        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf2[ifmin:ifmax]),'-',label='n_tilde(f)')
        #plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf2_ZL[ifmin:ifmax]),'--',label='n_tilde(f)')
        plt.title('Noise realisation in frequency domain (normalized to PSD)')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

    # PSD
    def plotPSD(self,fmin=None,fmax=None):

        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.__PSD[ifmin:ifmax]),'-',label='Sn(f)')
        plt.title('PSD')
        plt.xlabel('f (Hz)')
        plt.ylabel('Sn(f)^(1/2) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

    def plotinvPSD(self,fmin=None,fmax=None):

        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.__invPSD[ifmin:ifmax]),'-',label='Sn(f)')
        plt.title('inverse PSD')
        plt.xlabel('f (Hz)')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

    def getNf(self):
        return self.__Nf

    @property
    def kindPSD(self):
        return self.__kindPSD

    @property
    def PSD(self):
        return self.__PSD

    def getNoise(self):
        return self.__Ntraw

    @property
    def nf(self):
        return self.__nf

    @property
    def whitener_MP(self):
        return self.__whitener_MP

    @property
    def whitener(self):
        return self.__whitener
        
    @property
    def length(self):
        return self.__N
        
    @property
    def T(self):
        return self.__T
        
    @property
    def Ttot(self):
        return self.__Ttot

############################################################################################################################################################################################
if __name__ == "__main__":
    main()
