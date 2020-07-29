import numpy as np

# Write two-column spectrum
def write2col(data1,data2,outfile):
    f = open(outfile,'w')
    for i in range(len(data1)):
        f.write("%12.8f  %10.3e\n" %(data1[i], data2[i]))
    f.close()

# Write three-column spectrum
def write3col(data1,data2,data3,outfile):
    f = open(outfile,'w')
    for i in range(len(data1)):
        f.write("%12.8f  %10.3e  %s\n" %(data1[i], data2[i], str(data3[i])))
    f.close()


# Read 2-column spectrum
def read2col(infile):
    f = open(infile, 'r')
    return np.loadtxt(f, unpack=1)

