#! /usr/bin/env python
#
# This is the main Python executable, to be run from the command line (requires Python 3, scipy, astropy)
#
# The basic method is to generate a synthetic wind map based on the supplied latitudinal wind profile (in this case, it's Gaussian).
# Then we take the moment 1 of the synthetic (convolved) data cube and compare it with observed Doppler map to optimize the wind profile. 
# The convolved wind map is flux-weighted according to the molecular emission, based on a supplied vertical abundance profile.

#    Copyright (C) Martin A. Cordiner (cordiner@gmail.com)

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.



import os,sys
import numpy as np
from matplotlib import pyplot as plt 
from scipy.stats import scoreatpercentile,chi2
from mpfit import mpfit
from titan import getTeanT
import jplcat as JPLcat
import jpldat as JPLdat
from readwrite import write2col
from constantsSI import k,amu
from scipy.interpolate import interp1d,interp2d
from astropy.convolution import Gaussian2DKernel, convolve
import pickle
import matplotlib
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse,Circle
from astropy.io import fits as pyfits

global platescale,distance,sublat,ccw,beamxas,beamyas,beampa,vpeak,offset,fwhm,lat,pfix,xkm,ykm,mass

modelscale = 100. # pixel size of wind model (km) - tested OK for CH3CN and HNC
modelsize = 12000./modelscale  # Number of pixels in model (needs to be much bigger than data region to avoid deconvolution issues)
titanRad = 2575.
atmosCut = 1200. # How high to generate model atmosphere (zero flux above this altitude) - tested OK for CH3CN and HNC
T=160.

AU=1.496e11

scriptDir = os.path.dirname(os.path.abspath(__file__))

##########################
#Abundance slope function#
##########################
def abundance(alt1,loga1,alt2,loga2,radii):
   aslope = 10**(((radii-alt1)*(loga2-loga1)/(alt2-alt1)) + loga1)
   aslopecut = np.ma.masked_where(radii<alt1,aslope).filled(fill_value=0.0)
   return np.ma.masked_where(radii>alt2,aslopecut).filled(fill_value=10**loga2)

#######################
#  Gaussian function  #
#######################
gauss = lambda x,x0,fwhm: np.exp(-(x-x0)**2./(2.*(fwhm/2.3548)**2.))   

####################################
#Set up the atmospheric flux model #
####################################

if len(sys.argv) != 2: sys.exit("Usage: atmWindMapGaussFit.py <input_parameter_filename.par>")

# Load parameters
exec(open(sys.argv[1]).read())

dopsigma = (2*k*T/(mass*amu))**0.5

ccw=np.deg2rad(ccw)
sublat=np.deg2rad(sublat)
beamxkm=beamxas*725.27*distance
beamykm=beamyas*725.27*distance

# Atmospheric density interpolator
radius,density=np.loadtxt(scriptDir+'/densities_extrapolated.txt',unpack=1)
densityinterp=interp1d(radius,density,kind='linear',bounds_error=False,fill_value=0.0) 

# Set up the 3D grid axes for atmospheric model
x = ((np.arange(modelsize)-modelsize/2.)+0.5)*modelscale
y = ((np.arange(modelsize)-modelsize/2.)+0.5)*modelscale
z = ((np.arange(modelsize)-modelsize/2.)+0.5)*modelscale
a,b,c=np.meshgrid(x,y,z)

# Generate the radius of each grid point
radii = np.sqrt((a)**2+(b)**2+(c)**2)

# Get the density grid
densityinterpradii=densityinterp(radii)
# Delete shadowed cells
densityinterpradiishadow = np.ma.masked_where((((a**2+b**2)**0.5<titanRad) & (c>0.)),densityinterpradii).filled(fill_value=0.0)
# Get it the right way round
densityinterpradiishadowT=densityinterpradiishadow.transpose((2,0,1))
# Get temperatures as a function of radius
r=np.arange(titanRad,titanRad+5000.,10)
T_r=getTeanT(scriptDir,r-titanRad)

#Get the emission model
catfilename = scriptDir+'/JPL/'+mol+'.cat'
datfilename = scriptDir+'/JPL/'+mol+'.dat'
catfile = open(catfilename,'r')
datfile = open(datfilename,'r')
jplcat = JPLcat.jplcat(catfile)
jpldat = JPLdat.jpldat(datfile)
catfile.close()
datfile.close()
f=jplcat.freq[transition]*1e6

# Get abundance values as a function of (1D) radius
a_r=abundance(alt1+titanRad,loga1,alt2+titanRad,loga2,r)

# Get (abundance * flux) as a function of radius and make into interpolator as a funtion of radius
FluxAbund_r=(jplcat.getFluxMol(T_r,distance*AU,jpldat,indices = transition) * 1e-7) * a_r  # 1e-7 is to convert ergs to Joules  
FluxAbundInterpol=interp1d(r,FluxAbund_r,kind='linear',bounds_error=False,fill_value=0)

# Get the flux from each cell (W/m^2)
S_v = densityinterpradiishadowT * (modelscale*1e5)**3. * FluxAbundInterpol(radii)   # 1e5 is to convert km to cm
#S_v = S_v.transpose((2,0,1))

# Generate the latitude for each cell
# N pole vector
northx = -np.sin(ccw)*np.cos(sublat)
northy = np.cos(ccw)*np.cos(sublat)
northz = np.sin(sublat)
#dot product of north pole vector and cell vectors
dprod = (northx*a + northy*b + northz*-c)/radii
lats = 90. - np.degrees(np.arccos(dprod))

#Rotate the coordinates
# Sky-projected distance of each point from the polar vector
d = (a - (-b * np.tan(ccw))) * np.cos(ccw) 

# Cylindrical polar radii
rho = (d**2 + c**2)**0.5

######################################
# Wind speed as function of latitude #
######################################
def vfun(vpeak,offset,fwhm,l):
   return vpeak*gauss(l,offset,fwhm)

#########################################################################
# Function to generate wind speed as a function of supplied x,y (in km) #
#########################################################################
def getWind(vpeak,offset,fwhm,xnew,ynew):
   # Doppler shift of each cell. 1st term is the projection of the wind vector along the line of sight. 3rd term accounts for the polar tilt
   dop = (d/rho) * vfun(vpeak,offset,fwhm,lats) * np.cos(sublat)
 
   # Get the cube the "right way round" in my head (a,b,c --> x,y,z) (this leads to weird reversed indices in dop for modelCube)
   dop=dop.transpose((2,0,1)) 

   # Set up velocity grid   
   vgrid=np.arange(-1500,1600,100)

   # Set up model data cube
   modelCube = np.zeros([len(vgrid),len(y),len(x)])

   for i in range(len(x)):
      for j in range(len(y)):
         # Get flux sum (integrated through data cube) in each velocity channel
         # Only include relevant pixels
         if ((x[i]**2+y[j]**2)**0.5 <= titanRad + atmosCut):
            k=0
            for v in vgrid:
               modelCube[k][j][i] = np.sum(S_v[:,j,i]*(1.0/dopsigma)*(2.0*np.pi)**-0.5*gauss(v,dop[:,i,j],dopsigma*2.3548))
               k += 1 
   
   # Convolve the model cube with PSF 
   # Note the somewhat peculiar beam angle specification - may need to double-check?
   modelCubec = np.zeros([len(vgrid),len(y),len(x)])
   k=0
   kernel = Gaussian2DKernel(x_stddev=beamxkm/modelscale/2.3548,y_stddev=beamykm/modelscale/2.3548,theta=np.deg2rad(90+beampa))
   for plane in modelCube:
      modelCubec[k] = convolve(plane, kernel)
      k += 1

   # Weighted mean velocities
   modelCubecv = modelCubec * vgrid[:,np.newaxis,np.newaxis]
   vmodel = (np.ma.sum(modelCubecv,axis=0) / np.sum(modelCubec,axis=0)).transpose()
    
   vmodelI=interp2d(x,y,vmodel)
   return vmodelI(xnew,ynew)
   



#####################################################################
# Function to interface with MPFIT and return normalised residuals  #
#####################################################################
def getChisq(p, fjac=None, functkw=None):
   global xkm,ykm,akm,bkm,r,obsWind,mask
   
   rresiduals=[]
   model=[]
   
#  Parameters for this run            
   vpeak,offset,fwhm = p

#  Get the velocity field model
   model = getWind(vpeak,offset,fwhm,xkm,ykm)
   
#  Get the normalized residuals   
   residuals = np.ravel((obsWind - model)*2./(r['v_errup']+r['v_errlo']))/1000.
   residualsnonan = residuals[np.logical_not(np.isnan(residuals))]

   status = 0
   
   return ([status, residualsnonan])


############
#Make Plot #       
############
def doPlot(data,extent,fig,num):
   global titanRad,distance,platescale

   matplotlib.rcParams["contour.negative_linestyle"]='solid'
   cmap = LinearSegmentedColormap.from_list('mycmap', ['darkblue','blue','white','red','darkred'])
   planetgridcolor='black'                                                                         
   
   ax = fig.add_subplot(1,4,num)
   
   kmscale=725.27*distance*platescale

   norm = colors.Normalize(vmin=data.min(),vmax=data.max())
   c=ax.imshow(data,extent=[-extent,extent,-extent,extent],origin='lower',interpolation='nearest',norm=norm,zorder=-21,cmap=cmap)
   levels = np.array([-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) * 25.
   co=ax.contour(data,levels=levels,extent=[-extent,extent,-extent,extent],origin='lower',colors='k',linewidths=0.75,zorder=-20)
   ax.clabel(co,fmt="%d")

   ax.axis([-extent,extent,-extent,extent])
   
   #Titan circle
   titan = Circle((0.0,0.0), titanRad,color=planetgridcolor,linewidth=1.5,zorder=0,alpha=0.25)
   titan.set_facecolor('none')
   ax.add_artist(titan)

   #Latitude lines
   grid = np.linspace(-extent,extent,1024)
   x,y = np.meshgrid(grid,grid)
   z = np.hypot(x, y)
   with np.errstate(divide='ignore',invalid='ignore'): 
           zcoord = np.sqrt((titanRad)**2 - x**2 - y**2)
           dprod = (northx*x + northy*y + northz*zcoord)/(titanRad) #dot product of north pole vector and each vector in model planet
           z_lat = 90. - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid
   ax.contour(z_lat,colors=planetgridcolor,extent=[-extent, extent, -extent, extent],linestyles='dashed',zorder=0,alpha=0.25)

   #Longitude lines
   xma=np.ma.masked_where(np.isnan(zcoord),x)
   yma=np.ma.masked_where(np.isnan(zcoord),y)
   # Projected distances of each x,z point from the polar vector
   projx = (xma - (-yma * np.tan(ccw))) * np.cos(ccw) 
   projz = (zcoord - (yma * np.tan(sublat))) * np.cos(sublat)
   z_long = np.arctan2(projx,projz) #longitudes
   ax.contour(z_long,12,colors=planetgridcolor,extent=[-extent, extent, -extent, extent],linestyles='dotted',zorder=0)

   
###################################################################
#  START OF MAIN PROGRAM                                          #
###################################################################

parinfo=[]
fixed=[]

#  Chi-squared tolerance between successive iterations
ftol = 1.0e-5
#  Parameter value tolerance between successive iterations
xtol = 1.0e-4

#  set up parameters and their constraints
p0=[vpeak,offset,fwhm]
allfix=0

for i in range(len(p0)):
   parinfo.append({'value':p0[i], 'fixed':pfix[i], 'limited':[1,1], 'step':10})
   allfix+=pfix[i]

# vpeak limits
parinfo[0]['limits']=[0.,500]
# offset limits
parinfo[1]['limits']=[-90,90]
# fwhm limits
parinfo[2]['limits']=[10,150]


#Load the observed wind field
r=pickle.load(open(obsWindField,"rb"), encoding='latin-1')
# Doppler-Correct for velocity offset (fixing situation where some velocity errors are (close to) zero due to improper Monte Carlo runs)
r['v_errup'][r['v_errup']<1e-6]=1e30
r['v_errlo'][r['v_errlo']<1e-6]=1e30
#obsWind = (r['v']-np.average(r['v'],weights=2./(r['v_errup']+r['v_errlo']))).transpose() * 1000. 
obsWind = (r['v']).transpose() * 1000.

xkm = ((np.arange(len(r['v']))-len(r['v'])/2.)+0.5)*platescale*distance*725.27
ykm = ((np.arange(len(r['v'][0]))-len(r['v'][0])/2.)+0.5)*platescale*distance*725.27
akm,bkm=np.meshgrid(xkm,ykm)

# Define spatial region to be plotted
if (mol=='HNC'):
   mask = ( (((akm**2+bkm**2)**0.5)>4200) | (((akm**2+bkm**2)**0.5)<2200)  )
else:
   mask = ((((akm**2+bkm**2)**0.5)>4200))

plats=np.arange(-90,91)

print("")

if(showplots):
#  Plot the initial model:
   fig = plt.figure(figsize=(12,3))
   maskedobs = np.ma.array(obsWind,mask=mask)
   doPlot(maskedobs,4200,fig,1)
   plt.title("Observations")
   maskedmodel = np.ma.array(getWind(vpeak,offset,fwhm,xkm,ykm),mask=mask)
   doPlot(maskedmodel,4200,fig,2)
   plt.title("Starting model")
   residuals = maskedobs-maskedmodel
   doPlot(residuals,4200,fig,3)
   plt.title("Residuals")
   ax=fig.add_subplot(1,4,4)
   print("Close the plot window to continue...\n")
   ax.plot(plats,vfun(vpeak,offset,fwhm,plats),'k.')
   plt.show()  
   
if (allfix!=len(p0)):
#  Unless all parameters are fixed, do the fitting
   print("Starting MPFIT...")
   result = mpfit(getChisq, p0, parinfo=parinfo, xtol=xtol,ftol=ftol)
   
#  Print MPFIT status
   npts = len(np.ravel(r['v']))
   DOF = npts-(len(p0)-allfix)
   Chisq_R = float(result.fnorm) / DOF
   print('MPFIT exit status = ', result.status)
   if (result.status <= 0): 
      print('MPFIT', result.errmsg)
      sys.exit(result.status)
   
#  Get final parameters and (Scaled) errors
   pf = result.params
   pferr = result.perror
   print('\nBest-fit Gaussian parameters [v_max,phi_0,FWHM] and 1-sigma errors:')
   print(pf)
   print(np.asarray(pferr) * Chisq_R)
else:
   print("All parameters held fixed:")
   pf = p0
   print(pf)

vpeak,offset,fwhm = pf

finalModel = getWind(vpeak,offset,fwhm,xkm,ykm)

#  Write best-fit model wind field to file
outfilename="atmWindGaussFit."+obsWindField
print("\nWriting best-fit model wind field to "+outfilename)
outfile = open(outfilename, 'wb')
pickle.dump((xkm,ykm,finalModel),outfile)
outfile.close

#  Write best-fit winds as a function of latitude
outfilename="atmWindGaussFit."+os.path.splitext(obsWindField)[0]+".txt"
print("\nWriting best-fit latitudinal wind profile to "+outfilename+"\n")
write2col(plats,vfun(vpeak,offset,fwhm,plats),outfilename)


if(showplots):
#  Plot the final model:
   fig = plt.figure(figsize=(12,3))
   maskedobs = np.ma.array(obsWind,mask=mask)
   doPlot(maskedobs,4200,fig,1)
   plt.title("Observations")
   maskedmodel = np.ma.array(finalModel,mask=mask)
   doPlot(maskedmodel,4200,fig,2)
   plt.title("Final model")
   residuals = maskedobs-maskedmodel
   doPlot(residuals,4200,fig,3)
   plt.title("Residuals")
   ax=fig.add_subplot(1,4,4)
   print("Close the plot window to exit...")
   ax.plot(plats,vfun(vpeak,offset,fwhm,plats),'k.')
   plt.show()  
