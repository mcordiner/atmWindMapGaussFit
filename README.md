# atmWindMapGaussFit
Python executable code to generate best-fitting Gaussian wind speed profile (as a function of latitude) based on supplied 2D Doppler wind map (on a sky-projected cartesian grid).

A synthetic Doppler wind map is generated using on the supplied latitudinal wind profile - parameterised by the gaussian peak velocity, FWHM and latitudinal offset. 

A 3D grid of spectral opacities and radial velocities is constructed based on the supplied vertical abundance profile and an initial guess for the wind velocity profile (as a function of latitude). This grid is then raytraced to produce a spectral-spatial data cube, which is then convolved with a telescope point-spread function. The mean Doppler shift of the emitted flux from each pixel in the convolved (synthetic) data cube is generated and compared with the observed Doppler map using a least squares fit, to optimize the latitudinal wind profile. 

Run this code directly from the command line, using supplied example_input files as a template:

e.g.> ./atmWindMapGaussFitFWtau.py <input_parameter_filename.par>

Example input parameter files are in the example_input/ directory.

The input parameter file must contain a reference to a python pickle, which contains the following dictionary keys and values (1) 'v': a 2D numpy array of observed line-of-sight wind speeds as a function of image pixel, (2) 'v_errup' and 'v_errlo': 1-sigma upper and lower error margins, respectively, on each velocity measurement.

Best-fitting Gaussian wind profile parameters (and their 1-sigma errors) are printed to the terminal after the code completes. The synthetic (convolved) 2D wind field is written to a Python pickle in the current working directory, and the best-fitting latitudinal wind speed profile is written to a text file.

The code requires Python 3, scipy and astropy to be installed.

************************************************************************************************************************************************************
If you use this code in your research, please cite Cordiner et al. (2020, ApJL, 904, L12, "Detection of Dynamical Instability in Titan's Thermospheric Jet")
************************************************************************************************************************************************************
