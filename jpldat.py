# Class to read and store JPL dat file from local database as jpldat object
# Method getQ returns Q(T)

import re
import numpy as np
from scipy.interpolate import interp1d
import pdb

class jpldat:
    def __init__(self,datfile):   
        self.Q = {}
        self.splitter=re.compile('[()]')

        self.lines = datfile.readlines()
    
        for self.line in self.lines:    
            self.data = self.line.split('&')
        
            if (self.data[0].strip() == '\\\\ Species Tag:'):    
                self.molname = self.data[3].strip()
    
            # dipole moments and rot. constants
            elif (self.data[0].strip() == '\\ $\mu_a$ ='):
                self.mua = self.data[1].strip()
                self.A = self.data[3].strip()
    
            elif (self.data[0].strip() == '\\ $\mu_b$ ='):
                self.mub = self.data[1].strip()
                self.B = self.data[3].strip()
    
            elif (self.data[0].strip() == '\\ $\mu_c$ ='):
                self.muc = self.data[1].strip()
                self.C = self.data[3].strip()
    
            # partition functions at various temperatures
            elif (len(self.data)>2):
                if('Q(' in self.data[2]):
                    self.Q[self.splitter.split(self.data[2])[1]] = self.data[3].strip()
                    
            # Zero point partition function
            self.Q['0.0'] = 1.0e-20
            

# Method getQ to return Q(T)
    def getQ(self,T):
       Qs = []
       # Get the Q and T values
       self.QT = np.array(list(map(float, list(self.Q.keys()))))
       self.QQ = np.array(list(map(float, list(self.Q.values()))))
       # Sort by temperature
       inds = self.QT.argsort()
       self.QT = self.QT[inds]
       self.QQ = self.QQ[inds]

       # Spline interpolator for Q
       Qspline=interp1d(self.QT,self.QQ,kind='cubic',bounds_error=True)

       if not type(T) in [np.ndarray,list]:
          Ts = [T]
       else:
          Ts = T
       
       for T in Ts:     
          try:
              Qs.append(float(Qspline(T)))
          except:
          #    print "Temperature lies above catalogue range -> Using linear extrapolation of partition function"
              Qs.append(self.QQ[-1] + (T-self.QT[-1])*((self.QQ[-1]-self.QQ[-2])/(self.QT[-1]-self.QT[-2])))
      
       if (len(Qs) == 1):
          return Qs[0]
       else:
          return np.asarray(Qs)
 
# Method getQ to return Q(T) from external tabulated Q file
    def getTabulatedQ(self,T,Qfilename):
      
        Qfile = open(Qfilename, 'r')
      
        # Load partition function from file (columns: T, Q(T))
        QTab = np.loadtxt(Qfile, unpack=1)

      
      # Bracketing Qs.
        Qlow = 0.
        Tlow = 0.
        Tup = 1e30
        index = 0
        for QT in QTab[0]:
            QTf = float(QT)
            # bracket T
            if ((QTf < T) and (QTf >= Tlow)):
                Tlow = QTf
                Qlow = float(QTab[1][index])
            if ((QTf >= T) and (QTf < Tup)):
                Tup = QTf
                Qup = float(QTab[1][index])
            index+=1
    
    # linear interpolation to get Q(T)
        sort = QTab[0].argsort()
        QTab[0] = QTab[0][sort]
        QTab[1] = QTab[1][sort]
        if T > float(QTab[0][-1]):
            return float(QTab[1][-1]) + ((T - float(QTab[0][-1])) * ((float(QTab[1][-1]) - float(QTab[1][-2])) / (float(QTab[0][-1]) - float(QTab[0][-2]))))
        else:
            return Qlow + (T - Tlow)/(Tup - Tlow) * (Qup - Qlow)
