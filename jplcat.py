# Class to read and store JPL cat file from local database as jplcat object
# Function getIndices returns transition array indices within specified frequency range
#   The indices array is then used to select the relevant transitions at each step
# Function getSmusq calculates S*mu**2
# Function getA calculates Einstein A from S*mu**2
# Function getTau calculates tau from N, T, dv and S*mu**2
# Function getTB calculates antenna brightness temperature from tau
# Function getFlux calculates optically thin flux (erg s^-1 cm^-2) given Nmolec (total emitters), T, distance (cm)

import numpy as np
import re
h = 6.6260755e-27
c = 2.99792e10
k = 1.38062e-16

# Deal with g_up values greater than 999
gHash = {"A":1000,"B":1100,"C":1200,"D":1300,"E":1400,"F":1500,"G":1600,"H":1700,"I":1800,"J":1900,"K":2000,"L":2100,"M":2200}
grex=re.compile("([A-M])(\d+)")

class jplcat:
    def __init__(self,catfile):
    
        data = catfile.readlines()
        self.nTrans = len(data)
    
        self.index=np.zeros(self.nTrans)
        self.freq=np.zeros(self.nTrans)
        self.err=np.zeros(self.nTrans)
        self.lgint=np.zeros(self.nTrans)
        self.dr=np.zeros(self.nTrans)
        self.elo=np.zeros(self.nTrans)
        self.gup=np.zeros(self.nTrans)
        self.tag=np.zeros(self.nTrans)
        self.qnfmt=np.zeros(self.nTrans,dtype='object')
        self.qnup=np.zeros(self.nTrans,dtype='object')
        self.qnlo=np.zeros(self.nTrans,dtype='object')
        
        for i in range (self.nTrans):
            try:
               self.index[i] = i
               self.freq[i] = float(data[i][0:13].strip())
               self.err[i] = float(data[i][13:21].strip())
               self.lgint[i] = float(data[i][21:29].strip())
               self.dr[i] = float(data[i][29:31].strip())
               self.elo[i] = float(data[i][31:41].strip())
               self.gupString = data[i][41:44].strip()
               gm=grex.match(self.gupString)
               if gm:
                  self.gupString = gHash[gm.group(1)] + int(gm.group(2))
               self.gup[i] = float(self.gupString)
               self.tag[i] = float(data[i][44:52].strip())
               self.qnfmt[i] = data[i][52:55]
               self.qnup[i] = data[i][55:67].strip()
               self.qnlo[i] = data[i][67:79].strip()
            except:
               print("WARNING: Error in catalogue data on line %d" % (i+1))

# Function to get transition array indices within specified frequency range
    def getIndices(self,flo,fup):
    # flo and fup represents frequency range in MHz
        return np.logical_and((self.freq >= flo), (self.freq <= fup))
        
# Function to get strongest n transition indices at temperature T within specified frequency range - takes into account variation of beam size and number of molecules in a 1/rho coma
    def getSubIndicesStrongestComa(self,indices,n,T,jpldat):
        subindices = np.zeros(len(self.freq[indices]),dtype=bool)
        Strength = self.getA(jpldat,indices)*self.gup[indices]*(np.exp(h*self.freq[indices]*1e6/(k*T))-1)/(np.exp(self.EuK/T)*self.freq[indices]**2)

        for i in range(n):
           strongest = np.argmax(Strength)
           subindices[strongest] = True
           Strength[strongest] = 0.0
        
        return subindices
     

# Return truth array over all indices      
    def getAllIndices(self):
        return np.ones(self.nTrans,dtype=bool)

# Function to calculate Einstein A from S*mu**2
    def getA(self,jpldat,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
    
        smusq = self.getSmusq(jpldat,indices)
        
        return (64.0*(np.pi**4.)*((self.freq[indices]*1.0e6)**3.)*smusq)/(3.0e36*self.gup[indices]*h*c**3.)


# Function to calculate S*mu**2
# see http://www.ph1.uni-koeln.de/cdms/catalog for calculation details    
    def getSmusq(self,jpldat,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
    
        self.ElK = self.elo[indices] * 1.43883
        self.EuK = self.ElK + (0.047994 * self.freq[indices]/1.e3)
        intensity = 10**self.lgint[indices]
        
        return 2.40251e4 * intensity * jpldat.getQ(300.0) * (1.0/self.freq[indices]) * (1.0/(np.exp(-self.ElK/300.0) - np.exp(-self.EuK/300.0)))

# Function to calculate S*mu**2 when cat file is at a specific temperature
# see http://www.ph1.uni-koeln.de/cdms/catalog for calculation details    
    def getSmusqT(self,T,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
    
        self.ElK = self.elo[indices] * 1.43883
        self.EuK = self.ElK + (0.047994 * self.freq[indices]/1.e3)
        intensity = 10**self.lgint[indices]
        
        return 2.40251e4 * intensity * (1.0/self.freq[indices]) * (1.0/(np.exp(-self.ElK/T) - np.exp(-self.EuK/T)))


# Function to calculate tau from N (cm^-2), T (K), dv (km/s) and S*mu**2
# Formula of Nummelin et al. (2000)
    def getTau(self,N,T,dv,jpldat,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
    
        smusq = self.getSmusq(jpldat,indices)

        const = np.sqrt(np.log(2.)/(16.*np.pi**3.))*64.*np.pi**4./(3.*h) * 10.**-36 * 10.**-5
        return N * (const*smusq*np.exp(-self.EuK/T)*(np.exp((self.EuK-self.ElK)/T)-1)) / (jpldat.getQ(T)*dv)

# Function to calculate tau from N (cm^-2), T (K), dv (km/s) and S*mu**2 when cat file is at a specific temperature
# Formula of Nummelin et al. (2000)
    def getTauT(self,N,T,dv,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
    
        smusq = self.getSmusqT(T,indices)

        const = np.sqrt(np.log(2.)/(16.*np.pi**3.))*64.*np.pi**4./(3.*h) * 10.**-36 * 10.**-5
        return N * (const*smusq*np.exp(-self.EuK/T)*(np.exp((self.EuK-self.ElK)/T)-1)) / dv


# Function to calculate Rayleigh Jeans brightness temperature from tau
# Formula of Nummelin et al. (2000)
    def getTB(self,tau,T,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
        
        return (self.getJ(self.freq[indices]*1.e6,T)-self.getJ(self.freq[indices]*1.e6,2.725)) * (1 - np.exp(-tau))


# Optically thin flux (erg s^-1 cm^-2) given Nmolec (total emitters), T, distance (cm)
    def getFlux(self,Nmolec,T,dist,jpldat,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
    
        A = self.getA(jpldat,indices)

        return h*self.freq[indices]*1.e6 * Nmolec * (A*self.gup[indices]*np.exp(-self.EuK/T)) / (jpldat.getQ(T)*(4.0*np.pi*dist**2.0))


# Optically thin flux (ergs per unit distance**2) per molecule at given T
    def getFluxMol(self,T,distance,jpldat,indices = None):
        if indices is None:
            indices = np.ones(self.nTrans,dtype=bool)
    
        A = self.getA(jpldat,indices)

        return h*self.freq[indices]*1.e6 * (A*self.gup[indices]*np.exp(-self.EuK/T)) / (jpldat.getQ(T)*(4.0*np.pi*distance**2.0))


# Rayleigh Jeans Planck radiation brightness equation
    def getJ(self,freq,T):
        return ((h*freq/k)*(1/(np.exp((h*freq)/(k*T))-1)))
