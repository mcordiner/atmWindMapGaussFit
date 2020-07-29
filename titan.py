import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,interp2d
import os


# Solid angle of Titan + atmosphere
def titanSr(dist,atmos):
   """
   dist is distance in AU
   atmos is depth of atmosphere in km
   """

   from pymodules.constantsCGS import AU
   
   kmdist = dist * AU / 1.0e5
   
   # Solid angle = Area^2/distance^2  
   return np.pi*(2575.0+atmos)**2./kmdist**2.

# Interpolated temperatures as a function of altitude based on Teanby CIRS CH4 retrieval
def getTeanT(scriptDir,alt):
   infile = scriptDir+'/Lat_-27_Tz.txt' #Teanby CIRS limb temperature data from Rev275, lat -27

   z,T=np.loadtxt(infile,unpack=1)
         
   with np.errstate(divide='ignore',invalid='ignore'): #interp1d doesn't love nan values, but it doesn't break anything to keep them in there
      Tfunc = interp1d(z,T,kind='linear',bounds_error = False,fill_value=T[-1])
   
   return Tfunc(alt)
