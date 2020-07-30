# atmWindMapGaussFit
Python executable code to generate best-fitting Gaussian wind speed profile (as a function of latitude) based on supplied 2D Doppler wind map (on a sky-projected cartesian grid).

A synthetic Doppler wind map is generated using on the supplied latitudinal wind profile - parameterised by the gaussian peak velocity, FWHM and latitudinal offset. 

The mean velocity along the line-of-sight through each pixel in the synthetic (convolved) data cube is generated and compared with the observed Doppler map using a least squares fit, to optimize the wind profile. The convolved wind map is flux-weighted according to the molecular emission, based on the supplied vertical abundance profile.

Run this code directly from the command line, using supplied example_input files as a template:

e.g.> ./atmWindMapGaussFit.py <input_parameter_filename.par>

Example input parameter files are in the example_input/ directory.

The input parameter file must contain a reference to a python pickle, which contains the following dictionary keys and values (1) 'v': a 2D numpy array of observed line-of-sight wind speeds as a function of image pixel, (2) 'v_errup' and 'v_errlo': 1-sigma upper and lower error margins, respectively, on each velocity measurement.

Best-fitting Gaussian wind profile parameters (and their 1-sigma errors) are printed to the terminal after the code completes. The synthetic (convolved) 2D wind field is written to a Python pickle in the current working directory, and the best-fitting latitudinal wind speed profile is written to a text file.

The code requires Python 3, scipy and astropy to be installed.

IF YOU USE THIS CODE IN YOUR RESEARCH, PLEASE CITE CORDINER ET AL. (2020, SUBMITTED TO GRL)
