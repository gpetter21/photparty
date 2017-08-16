#The goal of this script is to read fits files, locate stars, and return magnitudes for those stars
#Background sky level is calculated using random median sampling of square arrays
#A specific inset of the data array is taken if desired
#Pixel values are summed row-wise and column-wise and background level is used to determine which rows and columns contain stars
#A list of matched star coordinates is prepared
#Star magnitudes are obtained using square apertures after being background subtracted for the region specific to the star
#By Teresa Symons 2016

#from config import *

#Import math, plotting, extraneous functions, and fits file handling:
import numpy as np
import numpy.ma as ma
import random as rand
import os
from astropy.io import fits
from background import background
from binsum import binsum
from starlocate import starlocate
from starmed import starmed
from starphot import starphot
from astropy.table import Table
import matplotlib.patches as patches
import matplotlib.pylab as plt


parameters = {}
raw_params = open('photparty.param', 'r')
param_lines = raw_params.readlines()
for entry in param_lines:
    if entry[0] != '#' and entry[0] != '\n' and entry[0] != '\r':
        tempstring = entry.split(':')
        param_name = str(tempstring[0])
        param_value = str(tempstring[1].strip())
        parameters[param_name] = param_value
raw_params.close()

path = parameters['path']
exptimekword = parameters['exptimekword']
exptime = int(parameters['exptime'])
filterkword = parameters['filterkword']
filter = parameters['filter']
airmasskword = parameters['airmasskword']
airmass = int(parameters['airmass'])
gainkword = parameters['gainkword']
gain = int(parameters['gain'])
backsize = int(parameters['backsize'])
backnum = int(parameters['backnum'])
framearea = parameters['framearea']
xlow = int(parameters['xlow'])
ylow = int(parameters['ylow'])
xhigh = int(parameters['xhigh'])
yhigh = int(parameters['yhigh'])
uplim = int(parameters['uplim'])
lowsig = int(parameters['lowsig'])
sig = int(parameters['sig'])
plotdetect = parameters['plotdetect']
plotstars = parameters['plotstars']
boxhw = int(parameters['boxhw'])
defectcol = list(map(int, parameters['defectcol'].split()))
defectrow = list(map(int, parameters['defectrow'].split()))

#Create list of files to run based on defined path, ignoring all files that are not fit or fits
files = [f for f in os.listdir(path) if any([f.endswith('fit'), f.endswith('fits')]) if not f.startswith('.')]

#Alternatively, run list of files in a specified text file
#In this case files will output to the location of the main script
#files = [line.rstrip() for line in open('files.txt')]

#Run for all files in list
for i in files:
    #Define names for output files based on names of original files
    (name, ext) = os.path.splitext(i)
    newname = path+'/'+name+'info.txt'
    datname = path+'/'+name+'dat.txt'
    #Open output files
    f = open(newname,'w')
    df = open(datname,'w')
    #Open fits file
    filepath = path+'/'+i
    image = fits.open(filepath)

    #Retrieve data, exposure time, airmass, and filter from fits file:
    Data = image[0].data
    maskedData = Data
    if len(defectcol) > 0:
        maskarray = np.zeros_like(Data)
        maskarray[:, defectcol] = 1
        maskedData = ma.masked_array(Data, maskarray)
    if len(defectrow) > 0:
        maskarray = np.zeros_like(Data)
        maskarray[defectrow] = 1
        maskedData = ma.masked_array(Data, maskarray)

    if exptimekword == 'NONE':
        etime = int(exptime)
    else:
        etime = image[0].header[exptimekword]
    if filterkword == 'NONE':
        filter = filter
    else:
        filter = image[0].header[filterkword]
    if airmasskword == 'NONE':
        airmass = int(airmass)
    else:
        airmass = image[0].header[airmasskword]
    if gainkword == 'NONE':
        gain = int(gain)
    else:
        gain = image[0].header[gainkword]

    #Compute background sky level through random median sampling:
    #Inputs: data array, nxn size of random subarray to use for sampling, and number of desired sampling iterations
    back, skyvals = background(maskedData,backsize,backnum)

    #Create desired inset of total data array:
    if framearea == 'half':
        #Find midpoint of image data
        mid = len(Data)/2
        #Take an inset of data that is half of area
        inset = Data[round(mid/2):3*round(mid/2),round(mid/2):3*round(mid/2)]
        xlow = 0
        xhigh = 0
        ylow = 0
        yhigh = 0
    if framearea == 'whole':
        #Use entire data array as inset
        inset = Data
        mid = 0
        xlow = 0
        xhigh = 0
        ylow = 0
        yhigh = 0
    if framearea == 'custom':
        inset = Data[ylow-1:yhigh-1,xlow-1:xhigh-1]
        mid = 0
    # Replace defective columns and rows with local medians
    # The median is taken for 5 rows above and 5 below the defective row/col, and the std dev is calculated as well.
    # Each pixel of the defective row/col is replaced by this median +/- a multiple of the sigma
    # If defective pixel is near edge of frame, will look further in opposite direction when getting median etc.
    buffersize = 5
    lowwidth, upwidth = buffersize, buffersize
    if len(defectcol) > 0:
        for col in defectcol:
            if framearea == 'half':
                col -= 511
            elif framearea == 'custom':
                col -= (xlow -1)
            if col < 0 or col > len(inset):
                continue
            for i in range(buffersize):
                if col + upwidth > len(inset[0]):
                    upwidth -= 1
                    lowwidth += 1
                elif col - lowwidth < 0:
                    lowwidth -= 1
                    upwidth += 1
            under = inset[:, (col - lowwidth):col]
            over = inset[:, (col + 1):(col + 1 + upwidth)]
            tot = np.hstack((under, over))
            median = np.median(tot)
            std = np.std(tot)
            for i in range(len(inset)):
                inset[i][col] = (median + std * rand.uniform(-3, 3))
    if len(defectrow) > 0:
        for row in defectrow:
            if framearea == 'half':
                row -= 511
            elif framearea == 'custom':
                row -= (ylow -1)
            if row < 0 or row > len(inset[0]):
                continue
            for i in range(buffersize):
                if row + upwidth > len(inset):
                    upwidth -= 1
                    lowwidth += 1
                elif row - lowwidth < 0:
                    lowwidth -= 1
                    upwidth += 1
            under = inset[(row - lowwidth):row]
            over = inset[(row + 1):(row + 1 + upwidth)]
            tot = np.vstack((under, over))
            median = np.median(tot)
            std = np.std(tot)
            for i in range(len(inset[0])):
                inset[row][i] = (median + std * rand.uniform(-3, 3))

    #Blanket removal of bad pixels above 45000 and 3*standard deviation below 0:
    inset[inset>uplim] = 0
    std = np.std(inset)
    inset[inset<-lowsig*std] = 0

    #Calculate sky background for specific inset:
    #Inputs: inset data array, nxn size of random subarray used for sampling, number of desired sampling iterations
    insetback, insetskyvals = background(inset,backsize,backnum)
    #Compute summed row and column values for desired array by number of bins:
    #Inputs: inset data array, number of bins desired
    rowsum, colsum = binsum(inset,1)

    #Locate values in summed row and column vectors that are greater than desired sigma level above background:
    #Inputs: Data array, background level variable, desired sigma detection level, summed row vector, summed column vector
    starrow, starcol, backsum, std, sigma = starlocate(inset,insetback,sig,rowsum,colsum)
    if starrow == []:
        print('Error: No stars found in '+name+' by row - check threshold.')
    if starcol == []:
        print('Error: No stars found in '+name+' by column - check threshold.')

    # Plot summed row and column values with detection level marked:
    if plotdetect == 'on':
        plt.plot(rowsum)
        plt.plot((0, len(rowsum)), (backsum + sigma * std, backsum + sigma * std))
        plt.title('Summed Rows' + '-' + name)
        plt.xlabel('Row Index in Data Inset')
        plt.ylabel('Summed Row Value')
        plt.show()
        plt.plot(colsum)
        plt.plot((0, len(colsum)), (backsum + sigma * std, backsum + sigma * std))
        plt.title('Summed Columns' + '-' + name)
        plt.xlabel('Column Index in Data Inset')
        plt.ylabel('Summed Column Value')
        plt.show()

    #Take indices of detected star pixels and divide into sublists by individual star:
    #Return sublists of star indices, number of stars, and median pixel of each star
    #Pair star center row with star center column and return total number of pairs found
    #Inputs: vectors containing indices of detected star pixels for row and column and inset data array
    rowloc, colloc, numstarr, numstarc, rowmed, colmed, starpoints, adjstarpoints = starmed(starrow,starcol,inset,mid,xlow,ylow)

    #Take list of star coordinates and find summed pixel values within a square aperture of desired size:
    #Also find background values for each star and subtract them from star values
    #Convert background-subtracted star values into fluxes and then magnitudes
    #Inputs: half-width of nxn square aperture, inset data array, vector containing star coordinates, exposure time
    boxsum, starback, backsub, flux, mags, hw, magerr = starphot(boxhw, inset, starpoints, etime, gain, name)

    #Output data table to file:
    n = len(mags)
    tname = [i]*n
    tfilter = [filter]*n
    tairmass = [airmass]*n
    tetime = [etime]*n
    x = [x for [x,y] in adjstarpoints]
    y = [y for [x,y] in adjstarpoints]
    t = Table([tname, tfilter, tairmass, tetime, x, y, mags, magerr], names=('File_Name', 'Filter', 'Airmass', 'Exposure_Time', 'X', 'Y', 'Magnitude', 'Mag_Err'))
    t.write(df,format='ascii')

    #Plot fits image with square apertures for detected stars overlaid
    if plotstars == 'on':
        fig, ax = plt.subplots(1)
        ax.imshow(Data, cmap='Greys',vmin=0,vmax=10)
        for i in range(0,len(x)):
            rect = patches.Rectangle(((x[i]-1-hw),(y[i]-1-hw)), 2*hw, 2*hw, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        if framearea == 'custom':
            rect = patches.Rectangle((xlow-1,ylow-1), xhigh-xlow, yhigh-ylow, linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        plt.title(name)
        plt.show()


    #Printing of values to check program function to file
    print('Exposure time:', file=f)
    print(etime, file=f)
    print('Filter:', file=f)
    print(filter, file=f)
    print('Airmass:', file=f)
    print(airmass, file=f)
    print('Sky background level:', file=f)
    print(back, file=f)
    print('Shape of inset array:', file=f)
    print(np.shape(inset), file=f)
    print('Inset background sky level:', file=f)
    print(insetback, file=f)
    print('Summed background value for one row/column:', file=f)
    print(backsum, file=f)
    print('Standard deviation of inset:', file=f)
    print(std, file=f)
    print('Detection level in sigma:', file=f)
    print(sigma, file=f)
    print('Indices of detected stars:', file=f)
    print(starrow, file=f)
    print(starcol, file=f)
    print('Indices of detected stars divided into sublist by star:', file=f)
    print(rowloc, file=f)
    print(colloc, file=f)
    print('Number of stars found by row/column:', file=f)
    print(numstarr, file=f)
    print(numstarc, file=f)
    print('Median pixel of each star by row/column:', file=f)
    print(rowmed, file=f)
    print(colmed, file=f)
    print('Paired indices of star centers:', file=f)
    print(starpoints, file=f)
    print('Total number of stars found:', file=f)
    print(len(starpoints), file=f)
    print('Coordinates of stars (x,y):', file=f)
    print(adjstarpoints, file=f)
    print('Width/Height of boxes:', file=f)
    print(hw * 2, file=f)
    print('Pixel sums for boxes around each star:', file=f)
    print(boxsum, file=f)
    print('Background values for each star:', file=f)
    print(starback, file=f)
    print('Background subtracted star values:', file=f)
    print(backsub, file=f)
    print('Flux of stars:', file=f)
    print(flux, file=f)
    print('Magnitudes for each star:', file=f)
    print(mags, file=f)