# atmWindMapGaussFit
Python executable code to generate best-fitting Gaussian wind speed profile (as a function of latitude) based on supplied 2D wind map.

A synthetic Doppler wind map is generated using on the supplied latitudinal wind profile - parameterised by the gaussian peak velocity, FWHM and latitudinal offset. 

The mean velocity along the line-of-sight through each pixel in the synthetic (convolved) data cube is generated and compared with the observed Doppler map using a least squares fit, to optimize the wind profile. The convolved wind map is flux-weighted according to the molecular emission, based on the supplied vertical abundance profile.

Run this code directly from the command line, using supplied example_input files as a template

e.g. atmWindMapGaussFit.py <input_parameter_filename.par>

The input parameter file must contain a reference to a python pickle, which contains the following dictionaries (1) 'v': an array of observed line-of-sight wind speeds as a function of image pixel, (2) 'v_errup' and 'v_errlo': 1-sigma upper and lower error margins, respectively, on each velocity measurement.

The code requires Python 3, scipy and astropy to be installed.