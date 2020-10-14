#! /usr/bin/env python
#
# Generate synthetic wind map based on supplied latitudinal wind profile (in this case, it's Gaussian)
# Take the moment 1 of the synthetic (convolved) data cube and compare with observed Doppler map to optimize the wind profile. This is called FW because the resulting (convolved) wind map is properly Flux Weighted according to the molecular emission. This version is called tau because the total emission is calculated by integrating the optical depth equation along the line of sight.

import os,sys
import numpy as np
from matplotlib import pyplot as plt 
from scipy.stats import scoreatpercentile,chi2
from pymodules.mpfit import mpfit
from pymodules.titan import getTeanT
from pymodules import jplcat as JPLcat
from pymodules import jpldat as JPLdat
from pymodules.io import write2col
from pymodules.constantsSI import h,k,amu
from scipy.interpolate import interp1d,interp2d
from astropy.convolution import Gaussian2DKernel, convolve
import pickle
import matplotlib
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse,Circle
from astropy.io import fits as pyfits

import pdb

global platescale,distance,sublat,ccw,beamxas,beamyas,beampa,vpeak,offset,fwhm,lat,pfix,xkm,ykm,mass

modelscale = 100. # pixel size of wind model (km) - tested OK for CH3CN and HNC
modelsize = 12000./modelscale  # Number of pixels in model (needs to be much bigger than data region to avoid deconvolution issues)
titanRad = 2575.
atmosCut = 1200. # How high to generate model atmosphere (zero flux above this altitude) - tested OK for CH3CN and HNC

# Model velocity grid parameters (m/s)
VMIN=-1500.
VMAX=1600.
VSTEP=100.

T_Titan=94. # Titan background temperature
T_CMB=2.73 # Background Temperature

AU=1.496e11

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

if len(sys.argv) != 2: sys.exit("Usage: atmWindMapGaussFit.py atmWindMapGaussFit.par")

# Load parameters
exec(open(sys.argv[1]).read())

ccw=np.deg2rad(ccw)
sublat=np.deg2rad(sublat)
beamxkm=beamxas*725.27*distance
beamykm=beamyas*725.27*distance

# Atmospheric density interpolator
home = os.environ.get('HOME')
radius,density=np.loadtxt(home+'/projects/titan/Krasnopolsky/densities_extrapolated.txt',unpack=1)
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
# Mask shadowed cells (behind and inside Titan and outside Titan's atmosphere)
with np.errstate(invalid='ignore'):
	atmosShadowMask=np.invert((radii>(titanRad+atmosCut)) | (((a**2+b**2)**0.5<titanRad) & (c>-(titanRad**2-a**2-b**2)**0.5)))

densityinterpradiishadow = np.ma.array(densityinterpradii, mask = np.invert(atmosShadowMask)).filled(fill_value=0.0)

# Get temperatures as a function of radius
r=np.arange(titanRad,titanRad+5000.,10)
T_r=getTeanT(r-titanRad)

# Get abundance values as a function of (1D) radius
a_r=abundance(alt1+titanRad,loga1,alt2+titanRad,loga2,r)
a_I=interp1d(r,a_r,kind='linear',bounds_error=False,fill_value=0)

Tgrid=getTeanT(radii-titanRad)
dopsigmaGrid = (2.*k*Tgrid/(mass*amu))**0.5
NGrid=a_I(radii) * densityinterpradiishadow * modelscale * 1e5

#Get the emission model
catfilename = home+'/scripts/JPL/'+mol+'.cat'
datfilename = home+'/scripts/JPL/'+mol+'.dat'
catfile = open(catfilename,'r')
datfile = open(datfilename,'r')
jplcat = JPLcat.jplcat(catfile)
jpldat = JPLdat.jpldat(datfile)
catfile.close()
datfile.close()
f=jplcat.freq[transition]*1e6

tauGrid=jplcat.getTau(NGrid,Tgrid,dopsigmaGrid*2.3548/1000.,jpldat,indices = transition)

# Check what the 2D collapsed images look like 
tauSum=np.sum(tauGrid,axis=2)
#NSum=np.sum(NGrid,axis=0)
#Tmean=np.mean(Tgrid,axis=0)

print("\nPeak optical depth = %8.2e\n" % np.max(tauSum))


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

# Set up velocity grid   
vgrid=np.arange(VMIN,VMAX,VSTEP)


######################################
# Wind speed as function of latitude #
######################################
def vfun(vpeak,offset,fwhm,l):
   return vpeak*gauss(l,offset,fwhm)

####################################
#  Rayleigh Jeans Planck function  #
####################################
J = lambda nu,T: h*nu/(k*(np.exp(h*nu/(k*T))-1))


#########################################################################
# Function to generate wind speed as a function of supplied x,y (in km) #
#########################################################################
def getWind(vpeak,offset,fwhm,xnew,ynew):
   # Doppler shift of each cell. 1st term is the projection of the wind vector along the line of sight. 3rd term accounts for the polar tilt
   dop = (d/rho) * vfun(vpeak,offset,fwhm,lats) * np.cos(sublat)

   # Set up model data cube
   modelCube = np.zeros([len(vgrid),len(y),len(x)])
   
   for i in range(len(x)):
      for j in range(len(y)):
         # Get flux sum (integrated through data cube) in each velocity channel
         # Only include relevant pixels
         if ((x[i]**2+y[j]**2)**0.5 <= titanRad + atmosCut):
         # Set the background level for the raytracing
            if ((x[i]**2+y[j]**2)**0.5 <= titanRad):
               Tbg = T_Titan
            else:
               Tbg = T_CMB
            k=0
				#Extract the tau values and temperatures along the z axis then select only the relevant cells in the contributing atmosphere
				# This could be sped up by moving the masking and reversing of the array z-axes out to the main program
            mask = atmosShadowMask[i,j,:]
            TBits=Tgrid[i,j,:][mask]
            tauBits=tauGrid[i,j,:][mask]
            dopBits=dop[i,j,:][mask]
            dopsigmaBits=dopsigmaGrid[i,j,:][mask]
            for v in vgrid:
               # Get the Doppler-shifted opacities corresponding to the current velocity channel (assuming a thermal distribution)
               tauBitsv=tauBits*gauss(v,dopBits,dopsigmaBits*2.3548)
               emission = 0.
               continuum = Tbg
               # Integrate the opacity equation along the line of sight 
               # Speed up by moving the array reversal operation to main
               for tauBitv,TBit in zip(tauBitsv[::-1],TBits[::-1]):
                  rad = J(f,TBit)-J(f,continuum)
                  emission += rad * (1.-np.exp(-tauBitv))
                  continuum = emission + Tbg
               modelCube[k,j,i] = emission
               k += 1 
   
   # Convolve the model cube with PSF 
   modelCubec = np.zeros([len(vgrid),len(y),len(x)])
   k=0
   kernel = Gaussian2DKernel(x_stddev=beamxkm/modelscale/2.3548,y_stddev=beamykm/modelscale/2.3548,theta=np.deg2rad(-beampa))
   for plane in modelCube:
      modelCubec[k] = convolve(plane, kernel)
      k += 1

   hdu=pyfits.PrimaryHDU(modelCubec.transpose(0,2,1))
   hdu.header['CDELT1'] =  platescale/(60.*60.)
   hdu.header['CDELT2'] =  platescale/(60.*60.)
   hdu.header['CDELT3'] =  VSTEP
   hdu.header['CTYPE1'] =  'RA---SIN'
   hdu.header['CTYPE2'] =  'DEC--SIN'
   hdu.header['CTYPE3'] =  'VELO-LSR'
   hdu.header['CRPIX1'] =  float((modelsize-1)/2+1)
   hdu.header['CRPIX2'] =  float((modelsize-1)/2+1)
   hdu.header['CRPIX3'] =  1.
   hdu.header['CRVAL1'] =  0.
   hdu.header['CRVAL2'] =  0.
   hdu.header['CRVAL3'] =  VMIN
   hdu.header['CUNIT3'] = 'M/S'
   hdu.header['BUNIT' ] =    'K'
   hdu.header['RESTFRQ'] =  f
   hdu.header['SPECSYS'] = 'LSRK'
   hdu.header['RADESYS'] = 'FK5'
   hdu.header['TELESCOP'] = 'ALMA'

   hdulist = pyfits.HDUList([hdu])
   hdulist.writeto('ModelCubeK.fits',overwrite=True)


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
   residuals = np.ravel((obsWind - model)*2./(r['v_errup']+r['v_errlo']))/1000
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
   levels = np.array([-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) * 25
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
#  MAIN PROGRAM                                                   #
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
r=pickle.load(open("ltefitalma.results.pickle","rb"), encoding='latin-1')
# Doppler-Correct for velocity offset (fixing situation where velocity errors are (close to) zero due to improper Monte Carlo runs)
r['v_errup'][r['v_errup']<1e-6]=1e30
r['v_errlo'][r['v_errlo']<1e-6]=1e30
obsWind = (r['v']-np.average(r['v'],weights=2./(r['v_errup']+r['v_errlo']))).transpose() * 1000.
#obsWind = (r['v']).transpose() * 1000.

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

# if(showplots):
#   #Plot the initial model:
# 
#    fig = plt.figure(figsize=(12,3))
#    maskedobs = np.ma.array(obsWind,mask=mask)
#    doPlot(maskedobs,4200,fig,1)
#    plt.title("Observations")
#    maskedmodel = np.ma.array(getWind(vpeak,offset,fwhm,xkm,ykm),mask=mask)
#    doPlot(maskedmodel,4200,fig,2)
#    plt.title("Starting model")
#    residuals = maskedobs-maskedmodel
#    maskedtau = np.ma.array(tauSum,mask=mask)
#    doPlot(maskedtau,4200,fig,3)
#    plt.title("$\\tau$")
#    ax=fig.add_subplot(1,4,4)
#    ax.plot(plats,vfun(vpeak,offset,fwhm,plats),'k.')
#    plt.show()  
   
if (allfix!=len(p0)):
#  Unless all parameters are fixed, do the fitting
   print("Starting MPFIT...")
   result = mpfit(getChisq, p0, parinfo=parinfo, xtol=xtol,ftol=ftol)
   
#  Print statistics and MPFIT status

   npts = len(np.ravel(r['v']))
   DOF = npts-(len(p0)-allfix)
   Chisq_R = float(result.fnorm) / DOF
   print('Reduced Chisq Xr = %.4f'% (Chisq_R))
   print('P (probability that model is different from data due to chance) = %.3f' % chi2.sf(result.fnorm,DOF))

   print('MPFIT exit status = ', result.status)
   if (result.status <= 0): 
      print('MPFIT', result.errmsg)
      sys.exit(result.status)
   
#  Get final parameters and (Scaled) errors
   pf = result.params
   pferr = result.perror
   print('\nBest-fit parameters:')
   print(pf)
   print(np.asarray(pferr) * Chisq_R)
else:
   print("All parameters held fixed:")
   pf = p0
   print(pf)

vpeak,offset,fwhm = pf

finalModel = getWind(vpeak,offset,fwhm,xkm,ykm)

#Write best-fit model wind field to file
outfilename="atmWindGaussFit."+obsWindField
print("\nWriting best-fit model wind field to "+outfilename)
outfile = open(outfilename, 'wb')
pickle.dump((xkm,ykm,finalModel),outfile)
outfile.close

#Write best-fit winds as a function of latitude
outfilename="atmWindGaussFit."+os.path.splitext(obsWindField)[0]+".txt"
print("\nWriting best-fit latitudinal wind profile to "+outfilename)
write2col(plats,vfun(vpeak,offset,fwhm,plats),outfilename)


if(showplots):
   print("Close the plot window(s) to continue...")
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
   ax.plot(plats,vfun(vpeak,offset,fwhm,plats),'k.')
   plt.show() 

	
