import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy import ndimage as ndi
import numpy.lib.stride_tricks as np_tricks
import bottleneck as btl
import scipy.interpolate as interp
import scipy.stats as sta
import itertools as itt
import medusa.local_activation.nonlinear_parameters as mnl # :)
import medusa.local_activation.spectral_parameteres as msp # :)
import pickle as pkl
import xgboost as xgb
import os
import plotly.express as px
import tensorflow as tf

#__________________________________________________________________________
#__ HELPERS _______________________________________________________________
def seconds2index(input,sr):
    return(np.round(input*sr).astype(int))

def absSquared(vector):
    return np.power(np.real(vector),2)+np.power(np.imag(vector),2)

def getWindowSampleCount(windowDuration,samplerate):
    windowSampleCount=int(windowDuration*samplerate)
    if windowSampleCount % 2 == 0:  #<- force odd number of samples per window
        windowSampleCount = windowSampleCount+1
    return windowSampleCount

def window2half(window):
    if window % 2 == 0:         #<- check window is odd
        raise ValueError("[window] parameter: only odd numbers allowed")
    halfWindow=int(window/2)    #<- int crops the decimal part, arithmetically halfWindow=(window-1)/2
    return halfWindow

def padVectorBegin(vector,amount,method):
    if not method in ['closest','zero','nan']:
        raise ValueError("[method] parameter: only 'closest', 'zero' or 'nan allowed")
    if method=='closest':
        vector=np.concatenate( (np.ones((amount,))*vector[0] ,vector) )
    elif method=='zeros':
        vector=np.concatenate( (np.zeros((amount,)), vector) )
    elif method=='nan':
        vector=np.concatenate( (np.zeros((amount,))*np.nan, vector) )
    return vector

def padVectorEnd(vector,amount,method):
    if not method in ['closest','zero','nan']:
        raise ValueError("[method] parameter: only 'closest', 'zero' or 'nan allowed")
    if method=='closest':
        vector=np.concatenate( (vector, np.ones((amount,))*vector[-1]) )
    elif method=='zeros':
        vector=np.concatenate( (vector, np.zeros((amount,))) )
    elif method=='nan':
        vector=np.concatenate( (vector, np.zeros((amount,))*np.nan) )
    return vector

def padVectorBothSides(vector,amount,method):
    if not method in ['closest','zero','nan']:
        raise ValueError("[method] parameter: only 'closest', 'zero' or 'nan allowed")
    else:
        vector=padVectorBegin(vector,amount,method)
        vector=padVectorEnd(vector,amount,method)
    return vector

def loadPickle(filepath):
    cFile = open(filepath, 'rb')
    content = pkl.load(cFile)
    cFile.close()
    return content

def dumpPickle(filepath,content):
    with open(filepath, 'wb') as file:
        pkl.dump(content, file)
#_________________________________________________________________________


#__________________________________________________________________________
#__ MATHEMATICS ___________________________________________________________
def deltaSignChanges(vector,window):
    #-> Credits to Mark Byers here: https://stackoverflow.com/questions/2936834/python-counting-sign-changes
    lambdaSignChanges = lambda segment: len(list(itt.groupby(segment, lambda x: x > 0)))-1
    #<-
    halfWindow=window2half(window)
    vector=padVectorBothSides(vector,halfWindow,'closest')
    view=np_tricks.sliding_window_view(vector,(window,))
    delta=list(itt.starmap(lambdaSignChanges,zip(view)))
    return np.array(delta)

#using bottleneck-------------------------------------------------------------------
def moveStatCentered(vector,window,statFunction,minCount):
    #Wrapper to use bottleneck functions at our style
    halfWindow=window2half(window)
    n=len(vector)
    vector=padVectorEnd(vector,halfWindow,'nan')    #<- because bottleneck acts like padding with nan only the begining
    out=statFunction(vector,window,min_count=minCount)
    return out[halfWindow:n+halfWindow] #<- because we want to recover the original length with the window centered

def moveMedianCentered(vector,window,minCount=3):
    statFunction=btl.move.move_median
    return moveStatCentered(vector,window,statFunction,minCount)

def moveMeanCentered(vector,window,minCount=3):
    statFunction=btl.move.move_mean
    return moveStatCentered(vector,window,statFunction,minCount)

def moveStdCentered(vector,window,minCount=3):
    statFunction=btl.move.move_std
    return moveStatCentered(vector,window,statFunction,minCount)

def moveVarCentered(vector,window,minCount=3):
    statFunction=btl.move.move_var
    return moveStatCentered(vector,window,statFunction,minCount)

#using scipy-------------------------------------------------------------------
def movingSkew(vector,window):
    #General variables
    halfWindow=window2half(window)
    #Lambda definition
    lambdaSkew=lambda segment: sta.skew(segment,bias=True,nan_policy='omit')
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'nan')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(window,))
    mskew=list(itt.starmap(lambdaSkew,zip(view)))
    return np.array(mskew).flatten()

def movingKurt(vector,window):
    #General variables
    halfWindow=window2half(window)
    #Lambda definition
    lambdaKurt=lambda segment: sta.kurtosis(segment,bias=True,nan_policy='omit')
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'nan')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(window,))
    mkurt=list(itt.starmap(lambdaKurt,zip(view)))
    return np.array(mkurt).flatten()
#__________________________________________________________________________

#__________________________________________________________________________
#__ FILTERING _____________________________________________________________
def filterBand(vector,filterFrequencies,samplerate,filterOrder=4):
    if filterFrequencies=='broadband':
        return vector
    else:
        if filterFrequencies[0]==0:
            sos = sg.butter(filterOrder, filterFrequencies[1], btype='low', analog=False, output='sos',fs=samplerate)
        else:    
            sos = sg.butter(filterOrder, filterFrequencies, btype='bandpass', analog=False, output='sos',fs=samplerate)
        filtered = sg.sosfiltfilt(sos, vector)
        return filtered
#__________________________________________________________________________

#__________________________________________________________________________
#__ BAND STATISTIC ________________________________________________________
def bandStatistic(vector,windowDuration,filterFrequencies,movingStatFunction,samplerate,filterOrder=4):
    vector = filterBand(vector,filterFrequencies,samplerate,filterOrder)
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    out = movingStatFunction(vector,windowSampleCount)
    return out

def bandSkew(vector,windowDuration,filterFrequencies,samplerate,filterOrder=4):
    movingStatFunction=movingSkew
    return bandStatistic(vector,windowDuration,filterFrequencies,movingStatFunction,samplerate,filterOrder)

def bandKurt(vector,windowDuration,filterFrequencies,samplerate,filterOrder=4):
    movingStatFunction=movingKurt
    return bandStatistic(vector,windowDuration,filterFrequencies,movingStatFunction,samplerate,filterOrder)

def bandStd(vector,windowDuration,filterFrequencies,samplerate,filterOrder=4):
    movingStatFunction=moveStdCentered
    return bandStatistic(vector,windowDuration,filterFrequencies,movingStatFunction,samplerate,filterOrder)
#__________________________________________________________________________

#__________________________________________________________________________
#__ TIME DOMAIN INPUT _____________________________________________________
def getSegmentBandRP(segment,frequencyBand,samplerate,resolution=1024):
    aux=tapperAndGetPSD(segment,samplerate,resolution,returnFreqs=True)
    roi=(aux[0]>=frequencyBand[0])&(aux[0]<=frequencyBand[1])
    psd=aux[1]
    psd=psd/np.sum(psd)
    rp=np.sum(psd[roi])
    return rp

def bandRelativePower(vector,frequencyBand,windowDuration,samplerate,resolution=1024):  #target band must be explicited
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    halfWindow=window2half(windowSampleCount)
    #Lambda definition
    lambdaRP=lambda segment: getSegmentBandRP(segment,frequencyBand,samplerate,resolution)
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'closest')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(windowSampleCount,))
    se=list(itt.starmap(lambdaRP,zip(view)))
    return np.array(se).flatten()

def centralTendencyMeasure(vector,windowDuration,samplerate,r=0.1):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    halfWindow=window2half(windowSampleCount)
    #Lambda definition
    lambdaCTM=lambda segment: mnl.central_tendency_measure(segment[...,np.newaxis],r=r)[0]
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'closest')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(windowSampleCount,))
    se=list(itt.starmap(lambdaCTM,zip(view)))
    return np.array(se).flatten()

def tapperVector(vector):   #if other methods added, keep hamming as default
    tap=np.hamming(len(vector))
    return vector*tap

def tapperAndGetPSD(vector,samplerate,resolution=1024,returnFreqs=False): #keep hamming as default tapping
    tappered=tapperVector(vector)
    if not returnFreqs:
        return sg.periodogram(tappered,samplerate,nfft=resolution,scaling='density')[1]
    else:
        return sg.periodogram(tappered,samplerate,nfft=resolution,scaling='density')


def spectralEntropy(vector,frequencyBand,windowDuration,samplerate,resolution=1024):  #target band must be explicited
    """ It's an spectral characteristic but should be called with time domain input to use the sliding logic """
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    halfWindow=window2half(windowSampleCount)
    #Lambda definition
    lambdaSpecE=lambda segment: msp.shannon_spectral_entropy(tapperAndGetPSD(segment,samplerate,resolution)[np.newaxis,...,np.newaxis],samplerate,frequencyBand)[0,0]
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'closest')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(windowSampleCount,))
    se=list(itt.starmap(lambdaSpecE,zip(view)))
    return np.array(se).flatten()

def medianFrequency(vector,frequencyBand,windowDuration,samplerate,resolution=1024):  #target band must be explicited
    """ It's an spectral characteristic but should be called with time domain input to use the sliding logic """
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    halfWindow=window2half(windowSampleCount)
    #Lambda definition
    lambdaSpecE=lambda segment: getMedianPSDIndex(tapperAndGetPSD(segment,samplerate,resolution),samplerate,frequencyBand)[0,0]
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'closest')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(windowSampleCount,))
    se=list(itt.starmap(lambdaSpecE,zip(view)))
    return np.array(se).flatten()

def sampleEntropy(vector,windowDuration,samplerate,m=2,rFactor=0.25):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    halfWindow=window2half(windowSampleCount)
    #std calculation
    stdVector=moveStdCentered(vector,windowSampleCount)
    #Lambda definition
    lambdaSampE=lambda segment, stdValue: mnl.sample_entropy(segment,m=m,r=rFactor*stdValue)
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'closest')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(windowSampleCount,))
    se=list(itt.starmap(lambdaSampE,zip(view,stdVector)))
    return np.array(se).flatten()

def lempelZivComplexity(vector,windowDuration,samplerate):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    halfWindow=window2half(windowSampleCount)
    #Median calculation
    medianVector=moveMedianCentered(vector,windowSampleCount)
    #Lambda definition
    lambdalZiv=lambda segment,median: mnl.__lz_algorithm((segment>median))
    #Vector padding
    vector=padVectorBothSides(vector,halfWindow,'closest')
    #View iteration and lambda application using starmap
    view=np_tricks.sliding_window_view(vector,(windowSampleCount,))
    lzc=list(itt.starmap(lambdalZiv,zip(view,medianVector)))
    return np.array(lzc).flatten()

def bandRatioRMS(vector,filterFrequencies,windowDuration,samplerate,filterOrder=4):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    #Filtering
    filtered = filterBand(vector, filterFrequencies, samplerate)
    #Band RMS ratio
    bandRatio=np.sqrt(moveMeanCentered(filtered**2,windowSampleCount))/np.sqrt(moveMeanCentered(vector**2,windowSampleCount))
    return bandRatio

def sigmaindex(vector,windowDuration,samplerate,filterOrder=4,f1Frequencies=[4,10],f2Frequencies=[20,40],f3Frequencies=[12.5,15]):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    #Filter signals
    f1sos = sg.butter(filterOrder, f1Frequencies, btype='bandpass', analog=False, output='sos',fs=samplerate)
    f1 = sg.sosfiltfilt(f1sos, vector)
    f2sos = sg.butter(filterOrder, f2Frequencies, btype='bandpass', analog=False, output='sos',fs=samplerate)
    f2 = sg.sosfiltfilt(f2sos, vector)
    f3sos = sg.butter(filterOrder, f3Frequencies, btype='bandpass', analog=False, output='sos',fs=samplerate)
    f3 = sg.sosfiltfilt(f3sos, vector)
    #Sigma Index
    sigmaindex=moveMeanCentered(np.abs(f3),windowSampleCount)/(moveMeanCentered(np.abs(f2),windowSampleCount)+moveMeanCentered(np.abs(f1),windowSampleCount))
    return sigmaindex

def petrosianFractalDimension(vector,timevector,windowDuration,samplerate):
    """ Petrosian, A. (1995). Kolmogorov complexity of finite sequences and recognition of different preictal EEG patterns.
    Proceedings Eighth IEEE Symposium on Computer-Based Medical Systems, 212–217. https://doi.org/10.1109/CBMS.1995.465426 """
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    #Gradient
    vectorGradient=np.gradient(vector,timevector)
    #Delta
    delta=deltaSignChanges(vectorGradient,windowSampleCount)
    #Petrosian Fractal Dimension
    pfd=np.log10(windowSampleCount)/(np.log10(windowSampleCount)+np.log10(windowSampleCount/(windowSampleCount+0.4*delta)))
    return pfd

def hjortActivity(vector,windowDuration,samplerate):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    #Activity
    activity=moveVarCentered(vector,windowSampleCount)
    return activity

def hjortMobility(vector,timevector,windowDuration,samplerate):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    #Gradient
    vectorGradient=np.gradient(vector,timevector)
    #Mobility
    mobility=np.sqrt(moveVarCentered(vectorGradient,windowSampleCount)/moveVarCentered(vector,windowSampleCount))
    return mobility

def hjortComplexity(vector,timevector,windowDuration,samplerate):
    #Gradient
    vectorGradient=np.gradient(vector,timevector)
    #Complexity
    complexity=hjortMobility(vectorGradient,timevector,windowDuration,samplerate)/hjortMobility(vector,timevector,windowDuration,samplerate)
    return complexity

def envelopeHilbert(vector,filterFrequencies,samplerate,filterOrder=4):
    filtered=filterBand(vector,filterFrequencies,samplerate,filterOrder)
    hilbert=sg.hilbert(filtered)
    return np.abs(hilbert)

def eodSymmetry(vector,windowDuration,samplerate):
    #General variables
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    maxHalfWidth=window2half(windowSampleCount)

    #Define matrices to expand the vector
    eMatrix=np.zeros((len(vector),maxHalfWidth),dtype=float)
    oMatrix=np.zeros((len(vector),maxHalfWidth),dtype=float)
    for i in range(maxHalfWidth):   #<- loop trough the expansion index of the matrices ------- CAN WE AVOID THIS?
        halfWidth=i+1
        #Vector padding to control the convolutions
        aux=padVectorBothSides(vector,halfWidth,'closest')
        #Fill the even expansion
        eKernel=np.zeros((2*halfWidth+1,))
        eKernel[0]=0.5
        eKernel[-1]=0.5
        eMatrix[:,i]=np.convolve(aux,eKernel,mode='valid')
        #Fill the odd expansion
        oKernel=np.zeros((2*halfWidth+1,))
        oKernel[0]=-0.5
        oKernel[-1]=0.5
        oMatrix[:,i]=np.convolve(aux,oKernel,mode='valid')
    
    #Calculate even-odd modulus squared
    evenModulusSquared=2*np.sum((eMatrix)**2,1)+vector**2
    oddModulusSquared=2*np.sum((oMatrix)**2,1)

    #Calculate symmetry-antysymmetry
    symmetry=np.sqrt( (evenModulusSquared)/(evenModulusSquared+oddModulusSquared) )
    antysymmetry=np.sqrt( (oddModulusSquared)/(evenModulusSquared+oddModulusSquared) )
    #Calculate symmetryness fuzzy descriptor [0,1]
    sygma=np.arcsin(symmetry)/(np.pi/2)

    return symmetry,antysymmetry,sygma
#_________________________________________________________________________


#__________________________________________________________________________
#__ FREQUENCY DOMAIN INPUT_________________________________________________
def getPSD(vector,returnFactor=False):
    vector=absSquared(vector)
    if returnFactor:
        factor=np.sum(vector)
        return vector/factor, factor
    else:
        return vector/np.sum(vector)

def getMedianPSDIndex(vector,method='firstOver'): #<------ this could be inside MATHEMATICS section, think on it!
    #some checks
    if not method in ['firstOver','lastUnder']:
        raise ValueError("[method] parameter: only 'fisrtOver' or 'lastUnder' allowed")
    vector=np.cumsum(vector)
    if not abs(1-vector[-1])<0.1:
        raise ValueError("[vector] validation: input vector must be normalized")
    
    if method=='firstOver':
        vector=vector>=0.5
        return np.argwhere(vector)[0][0]
    elif method=='lastUnder':
        vector=vector<=0.5
        return np.argwhere(vector)[-1][0]
#_________________________________________________________________________


#_________________________________________________________________________
#__ TIME-WINDOW METRICS __________________________________________________  DEPRECATED!!!!!!!
def F1(gtVector,predVector):
    tp=np.sum(gtVector*predVector)
    fp=np.sum((1-gtVector)*predVector)
    fn=np.sum(gtVector*(1-predVector))
    return 2*tp/(2*tp+fp+fn)

def MCC(gtVector,predVector):
    tp=np.sum(gtVector*predVector)
    fp=np.sum((1-gtVector)*predVector)
    tn=np.sum((1-gtVector)*(1-predVector))
    fn=np.sum(gtVector*(1-predVector))
    return (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

def binaryIoU(vector1,vector2):
    return np.sum(vector1*vector2)/np.sum(np.maximum(vector1,vector2))

def IoUfromLabel(label,annotations,verbose=1):  #<-   DEPRECATED!!!!!!! see METRICS section below
    isIntersect=annotations.apply(
        lambda row: ((label.startInd>=row.startInd)&(label.startInd<row.stopInd)) | ((row.startInd>=label.startInd)&(row.startInd<label.stopInd))
        ,axis=1)
    aux=np.sum(isIntersect)
    if aux==0:
        IoU=0
    else:
        if (aux>1)&(verbose>0):   #<----------- beware of labels with more than 1 intersecting detection
            print("WARNING: number of intersections is "+str(aux))
        allIntersections=np.where(isIntersect)[0]
        intersectingIndexes=np.array([])
        for i in allIntersections:
            intersecting=annotations.loc[i]
            tempIndexes=np.arange(intersecting.startInd,intersecting.stopInd)
            intersectingIndexes=np.concatenate((intersectingIndexes,tempIndexes))
        labelIndexes=np.arange(label.startInd,label.stopInd)
        IoU=len(np.intersect1d(labelIndexes,intersectingIndexes))/len(np.union1d(labelIndexes,intersectingIndexes))
    return IoU
#_________________________________________________________________________


#_________________________________________________________________________
#__ SPINDLE DETECTION 1 __________________________________________________
""" Ray, Laura, Stéphane Sockeel, Melissa Soon, Arnaud Bore, Ayako Myhr, Bobby Stojanoski, Rhodri Cusack, Adrian Owen, Julien Doyon, and Stuart Fogel.
“Expert and Crowd-Sourced Validation of an Individualized Sleep Spindle Detection Method Employing Complex Demodulation and Individualized Normalization.”
Frontiers in Human Neuroscience 9 (2015). https://www.frontiersin.org/articles/10.3389/fnhum.2015.00507. """

def smoothingByTriangles(vector,ratio):
    #kernel definition
    aux=np.arange(ratio,0,-1)
    kernel=np.append(np.flip(aux[1:]),aux)
    kernel=kernel/sum(kernel)
    #border criterium
    vector=np.concatenate(( np.ones((len(aux)-1,))*vector[0], vector, np.ones((len(aux)-1,))*vector[-1] ))
    #apply convolution
    vector=np.convolve(vector, kernel, mode='valid')
    return vector

def applySpindleThresholds(response,detectThres,frontierThres):
    if (type(frontierThres)==float)|(type(frontierThres)==int):
        output=applySingleSpindleThreshold(response,detectThres,frontierThres)
    elif len(frontierThres)==2:
        output=applyCombinedSpindleThreshold(response,detectThres,frontierThres)
    else:
        raise ValueError("[frontierThres] parameter: float value, int value or list of two of this elements allowed")
    return output

def applySingleSpindleThreshold(response,detectThres,frontierThres):
    overDetection=response>detectThres
    overFrontier=ndi.label(response>frontierThres)[0]
    regionLabels=np.setdiff1d(np.unique(overDetection*overFrontier),0)
    regions=np.in1d(overFrontier, regionLabels)
    regions=ndi.label(regions)[0]
    objects=ndi.find_objects(regions)
    startInd=np.apply_along_axis(lambda x: x[0].start,1,objects)
    stopInd=np.apply_along_axis(lambda x: x[0].stop,1,objects)
    spindleId=np.arange(len(regionLabels))

    output=pd.DataFrame({
        'spindleId':spindleId,
        'startInd':startInd,
        'stopInd':stopInd
    })

    output['peakInd']=output.apply(lambda row: row.startInd+np.argmax(response[row.startInd:row.stopInd]), axis=1)
    output['peakValue']=response[output.peakInd]

    return output

def applyCombinedSpindleThreshold(response,detectThres,frontierThres):
    sortedThres=np.sort(frontierThres)
    overDetection=response>detectThres

    #smaller frontierThres 
    overFrontier0=ndi.label(response>sortedThres[0])[0]
    regionLabels0=np.setdiff1d(np.unique(overDetection*overFrontier0),0)
    regions0=np.in1d(overFrontier0, regionLabels0)
    regions0=ndi.label(regions0)[0]
    objects0=ndi.find_objects(regions0)
    startInd0=np.apply_along_axis(lambda x: x[0].start,1,objects0)
    stopInd0=np.apply_along_axis(lambda x: x[0].stop,1,objects0)
    spindleId=np.arange(len(regionLabels0))

    #larger frontierThres 
    overFrontier1=ndi.label(response>sortedThres[1])[0]
    regionLabels1=np.setdiff1d(np.unique(overDetection*overFrontier1),0)
    regions1=np.in1d(overFrontier1, regionLabels1)
    regions1=ndi.label(regions1)[0]
    regions1=(regions1!=0)*regions0     #<------- Main step
    objects1=ndi.find_objects(regions1)
    startInd1=np.apply_along_axis(lambda x: x[0].start,1,objects1)
    stopInd1=np.apply_along_axis(lambda x: x[0].stop,1,objects1)

    if (sortedThres==frontierThres).all():
        output=pd.DataFrame({
            'spindleId':spindleId,
            'startInd':startInd0,
            'stopInd':stopInd1
        })
    else:
        output=pd.DataFrame({
            'spindleId':spindleId,
            'startInd':startInd1,
            'stopInd':stopInd0
        })

    output['peakInd']=output.apply(lambda row: row.startInd+np.argmax(response[row.startInd:row.stopInd]), axis=1)
    output['peakValue']=response[output.peakInd]

    return output

def detectSpindles_Ray2015(signal,samplerate,centralFreq=13.5,filterWidth=5,filterOrder=4,
windowDuration=60,detectThres=2.33,frontierThres=0.1,minSpindleDuration=0.49):
    """ do not modify default parameters unless very sure of what you're doing """
    filterHalfWidth=0.5*filterWidth
    # general variables used ------------------------------->
    npoints = len(signal)
    totaltime = (npoints-1)/samplerate
    timepoints=np.linspace(0,totaltime,npoints)
    # <------------------------------------------------------

    # 1 - preprocess (CD and butterworth filter)
    signal=signal*np.exp(-2*np.pi*complex(0,1)*centralFreq*timepoints)
    b,a = sg.butter(filterOrder, filterHalfWidth, btype='lowpass', analog=False, output='ba',fs=samplerate)
    signal = sg.filtfilt(b, a, signal, padtype='odd', padlen=3*(max(len(b),len(a))-1))    #Matlab's padding: #credits to this dicussion: https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
 
    # 2 - smoothing
    signal = smoothingByTriangles(signal,samplerate/centralFreq)

    #Missing from original implementation -> bad data to NaN (must be supported but done previously to this kind of functions)
    #TBD: add a mask vector (binary) as input that could be included in the calculations sensitive to noise

    # 3 - z-score (on the modulus squared)
    signal=(np.power(np.real(signal),2)+np.power(np.imag(signal),2))
    windowSampleCount=getWindowSampleCount(windowDuration,samplerate)
    mavg=moveMeanCentered(signal,windowSampleCount)
    mstd=moveStdCentered(signal,windowSampleCount)
    signal=(signal-mavg)/mstd
    # 4 - tresholding
    out=applySpindleThresholds(signal,detectThres,frontierThres)
    # 5 - discard spindles of less than .5s
    out['duration']=(out['stopInd']-out['startInd'])/samplerate
    out=out[out.duration > minSpindleDuration]

    #Missing from original implementation -> delay between spindles
    #I wouldn't do it, you can use these detections at your will and discard afterwards based on distances

    return out.reset_index()
#_________________________________________________________________________


#_________________________________________________________________________
#__ DATABASE MANAGEMENT __________________________________________________
# NEEDS REFINEMENT AND CONVERGE THE MULTIPLE DATABASES
def loadMASSSpindles(path,returnSignals=False,forceSamplerate=0,onlySpindlesFilteredN2=False):
    if returnSignals & (forceSamplerate>0):
        raise ValueError("[returnSignals, forceSamplerate] parameters: resample functionality not availabe at signal load, please leave one of the parameters with default value")

    #signalsMetadata
    signalsMetadata=pd.read_csv(path+'\\signals\\signalsMetadata.csv')
    signalsMetadata['subjectId']=signalsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)
    signalsMetadata['isOriginalSamplerate']=True
    #-------------------------------------->
    signalsMetadata['database']='MASS'
    #<--------------------------------------
    if forceSamplerate>0:   #use to load the information for a particular samplerate
        signalsMetadata['samplerate']=forceSamplerate
        signalsMetadata['isOriginalSamplerate']=False

    #annotations
    if onlySpindlesFilteredN2:
        annotations=pd.read_csv(path+'\\annotations\\spindlesFilteredN2.csv')
        annotations['subjectId']=annotations.apply(
            lambda row: str(row.subjectId).zfill(4),axis=1)
        annotations['labelerId']=annotations.apply(
            lambda row: str(row.labelerId).zfill(4),axis=1)
    else:  
        annotations=pd.read_csv(path+'\\annotations\\annotations.csv')
        annotations['subjectId']=annotations.apply(
            lambda row: str(row.subjectId).zfill(4),axis=1)
        annotations['labelerId']=annotations.apply(
            lambda row: str(row.labelerId).zfill(4),axis=1)
    
    #add stop and index colums
    annotations=annotations.merge(signalsMetadata[['subjectId','samplerate']],how='left',on='subjectId')
    annotations['stopTime']=annotations.apply(
        lambda row: row.startTime+row.duration , axis=1)
    annotations['startInd']=annotations.apply(
        lambda row: seconds2index(row.startTime,row.samplerate) , axis=1)
    annotations['stopInd']=annotations.apply(
        lambda row: seconds2index(row.stopTime,row.samplerate) , axis=1)

    if returnSignals:
        #load signals from pickle
        signals={}
        for index, row in signalsMetadata.iterrows():
            signalpath=path+"/signals/"+row.file
            signals[row.subjectId]=loadPickle(signalpath)
        return signals, annotations, signalsMetadata

    else:
        return annotations, signalsMetadata

def loadDREAMSSpindles(path,equalize=True):
    #signalsMetadata
    signalsMetadata=pd.read_csv(path+'\\signals\\signalsMetadata.csv')
    signalsMetadata['subjectId']=signalsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)

    #subjectsMetadata
    subjectsMetadata=pd.read_csv(path+'\\signals\\subjectsMetadata.csv')
    subjectsMetadata['subjectId']=subjectsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)

    #load signals from .txt
    signals={}
    for index, row in signalsMetadata.iterrows():
        signals[row.subjectId]=np.loadtxt(path+'\\signals\\'+row.filename,skiprows=1)

    if equalize:
        #------------------->
        masterSamplerate=200
        masterDuration=1800.0
        #<-------------------
        originalSignals=signals.copy()
        timepoints=np.linspace(0.0,masterDuration,int(masterDuration)*masterSamplerate)
        for key, signal in originalSignals.items():
            metadata=signalsMetadata.loc[np.where(signalsMetadata.subjectId==key)[0][0]]
            signalDuration=float(len(signal))/metadata.samplerate
            if metadata.samplerate!=masterSamplerate:
                print("SubjectId: "+metadata.subjectId+"--------------")
                print("resampling from "+str(metadata.samplerate)+" to "+str(masterSamplerate))
                thisTimepoints=np.linspace(0.0,signalDuration,len(signal))
                aux=interp.interp1d(thisTimepoints,signal,kind='linear')
                signals[key]=aux(timepoints)
            if signalDuration > masterDuration:
                print("SubjectId: "+metadata.subjectId+"--------------")
                print("duration discrepancy, removing last "+str(round(signalDuration-masterDuration,3))+" seconds")
                signals[key]=signal[:int(masterDuration)*masterSamplerate]
        #update metadata
        signalsMetadata.samplerate=masterSamplerate
    signalsMetadata['database']='DREAMS'


    #spindle annotations
    annotationsMetadata=pd.read_csv(path+'\\annotations\\annotationsMetadata.csv')
    annotationsMetadata['subjectId']=annotationsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)
    annotationsMetadata['labelerId']=annotationsMetadata.apply(
        lambda row: str(row.labelerId).zfill(4),axis=1)

    #populate annotations metadata
    annotations=pd.DataFrame(columns=['startTime','duration','channel','subjectId','labelerId','type'])
    for index,row in annotationsMetadata.iterrows():
        aux=pd.read_csv(path+'\\annotations\\'+row.filename,index_col=False)
        aux['channel']=row.channel
        aux['subjectId']=row.subjectId
        aux['labelerId']=row.labelerId
        aux['type']='spindle'
        annotations=pd.concat((annotations,aux))

    #add stop and index colums
    annotations=annotations.merge(signalsMetadata[['subjectId','samplerate']],how='left',on='subjectId')
    annotations['stopTime']=annotations.apply(
        lambda row: row.startTime+row.duration , axis=1)
    annotations['startInd']=annotations.apply(
        lambda row: seconds2index(row.startTime,row.samplerate) , axis=1)
    annotations['stopInd']=annotations.apply(
        lambda row: seconds2index(row.stopTime,row.samplerate) , axis=1)
    annotations=annotations.reset_index()

    return signals, annotations, signalsMetadata

def loadDREAMSSpindlesDemo(path,equalize=True):
    #signalsMetadata
    signalsMetadata=pd.read_csv(path+'\\signalsMetadata.csv')
    signalsMetadata['subjectId']=signalsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)

    #subjectsMetadata
    subjectsMetadata=pd.read_csv(path+'\\subjectsMetadata.csv')
    subjectsMetadata['subjectId']=subjectsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)

    #load signals from .txt
    signals={}
    for index, row in signalsMetadata.iterrows():
        signals[row.subjectId]=np.loadtxt(path+'\\DREAMS\\'+row.filename,skiprows=1)

    if equalize:
        #------------------->
        masterSamplerate=200
        masterDuration=1800.0
        #<-------------------
        originalSignals=signals.copy()
        timepoints=np.linspace(0.0,masterDuration,int(masterDuration)*masterSamplerate)
        for key, signal in originalSignals.items():
            metadata=signalsMetadata.loc[np.where(signalsMetadata.subjectId==key)[0][0]]
            signalDuration=float(len(signal))/metadata.samplerate
            if metadata.samplerate!=masterSamplerate:
                print("SubjectId: "+metadata.subjectId+"--------------")
                print("resampling from "+str(metadata.samplerate)+" to "+str(masterSamplerate))
                thisTimepoints=np.linspace(0.0,signalDuration,len(signal))
                aux=interp.interp1d(thisTimepoints,signal,kind='linear')
                signals[key]=aux(timepoints)
            if signalDuration > masterDuration:
                print("SubjectId: "+metadata.subjectId+"--------------")
                print("duration discrepancy, removing last "+str(round(signalDuration-masterDuration,3))+" seconds")
                signals[key]=signal[:int(masterDuration)*masterSamplerate]
        #update metadata
        signalsMetadata.samplerate=masterSamplerate
    signalsMetadata['database']='DREAMS'


    #spindle annotations
    annotationsMetadata=pd.read_csv(path+'\\annotationsMetadata.csv')
    annotationsMetadata['subjectId']=annotationsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)
    annotationsMetadata['labelerId']=annotationsMetadata.apply(
        lambda row: str(row.labelerId).zfill(4),axis=1)

    #populate annotations metadata
    annotations=pd.DataFrame(columns=['startTime','duration','channel','subjectId','labelerId','type'])
    for index,row in annotationsMetadata.iterrows():
        aux=pd.read_csv(path+'\\DREAMS\\'+row.filename,index_col=False,skiprows=1,sep='\t',names=["startTime","duration"])
        aux['channel']=row.channel
        aux['subjectId']=row.subjectId
        aux['labelerId']=row.labelerId
        aux['type']='spindle'
        annotations=pd.concat((annotations,aux))

    #add stop and index colums
    annotations=annotations.merge(signalsMetadata[['subjectId','samplerate']],how='left',on='subjectId')
    annotations['stopTime']=annotations.apply(
        lambda row: row.startTime+row.duration , axis=1)
    annotations['startInd']=annotations.apply(
        lambda row: seconds2index(row.startTime,row.samplerate) , axis=1)
    annotations['stopInd']=annotations.apply(
        lambda row: seconds2index(row.stopTime,row.samplerate) , axis=1)
    annotations=annotations.reset_index()

    return signals, annotations, signalsMetadata

def loadCOGNITIONSpindles(path,returnSignals=False):
    #signalsMetadata
    signalsMetadata=pd.read_csv(path+'\\signals\\signalsMetadata.csv')
    signalsMetadata['subjectId']=signalsMetadata.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)
    signalsMetadata['isOriginalSamplerate']=True
    #-------------------------------------->
    signalsMetadata['database']='COGNITION'
    #<--------------------------------------
    signalsMetadata['samplerate']=200   #<- samplerate harcoded to 200 Hz
    signalsMetadata['isOriginalSamplerate']=False

    #annotations
    annotations=pd.read_csv(path+'\\annotations\\annotations.csv')
    annotations['subjectId']=annotations.apply(
        lambda row: str(row.subjectId).zfill(4),axis=1)
    annotations['labelerId']=annotations.apply(
        lambda row: str(row.labelerId).zfill(4),axis=1)

    #add stop and index colums
    annotations=annotations.merge(signalsMetadata[['subjectId','samplerate']],how='left',on='subjectId')
    annotations['stopTime']=annotations.apply(
        lambda row: row.startTime+row.duration , axis=1)
    annotations['startInd']=annotations.apply(
        lambda row: seconds2index(row.startTime,row.samplerate) , axis=1)
    annotations['stopInd']=annotations.apply(
        lambda row: seconds2index(row.stopTime,row.samplerate) , axis=1)

    if returnSignals:
        #load signals from pickle --------- RESAMPLE TO 200Hz HARCODED
        signals={}
        for index, row in signalsMetadata.iterrows():
            signalpath=path+"/signals/"+row.filename
            signals[row.subjectId]=np.loadtxt(signalpath)
            signals[row.subjectId]=sg.resample_poly(signals[row.subjectId],up=2,down=5)

        return signals, annotations, signalsMetadata

    else:
        return annotations, signalsMetadata
    
def loadCOGNITIONSpindles_v2(path,returnSignals=False):
    #signalsMetadata
    signalsMetadata=pd.read_csv(path+'\\signals\\signalsMetadata.csv')
    signalsMetadata['isOriginalSamplerate']=False
    #-------------------------------------->
    signalsMetadata['database']='COGNITION_v2'
    #<--------------------------------------

    #annotations
    annotations=pd.read_csv(path+'\\annotations\\annotations.csv')
    
    #add stop and index colums
    annotations['samplerate']=200
    annotations['stopTime']=annotations.apply(
        lambda row: row.startTime+row.duration , axis=1)
    annotations['startInd']=annotations.apply(
        lambda row: seconds2index(row.startTime,row.samplerate) , axis=1)
    annotations['stopInd']=annotations.apply(
        lambda row: seconds2index(row.stopTime,row.samplerate) , axis=1)

    if returnSignals:
        #load signals from pickle --------- RESAMPLED TO 200Hz
        signals={}
        for subject in np.unique(signalsMetadata.subjectId):
            signals[subject]={}
            for channel in np.unique(signalsMetadata.channel):
                signals[subject][channel]=np.array([])
        for index, row in signalsMetadata.iterrows():
            signalpath=path+"/signals/"+row.filename
            signals[row.subjectId][row.channel]=loadPickle(signalpath)

        return signals, annotations, signalsMetadata

    else:
        return annotations, signalsMetadata
    
def loadCOGNITIONHypnogram(subject,cognipath):
    return loadPickle(cognipath+'/stages/'+subject+'.pkl')
#_________________________________________________________________________

#_________________________________________________________________________
#__ FEATURE & LABEL MANAGEMENT ___________________________________________
def labelIntersectionIoU(label,annotations,verbose=0):
    isIntersect=annotations.apply(
        lambda row: ((label.startInd>=row.startInd)&(label.startInd<row.stopInd)) | ((row.startInd>=label.startInd)&(row.startInd<label.stopInd))
        ,axis=1)
    labelIntersect=None
    #output lenght should be equal 1 or 0 (at least now)
    aux=np.sum(isIntersect)
    if aux==0:
        IoU=0
    else:
        if (aux>1)&(verbose>0):   #<----------- beware of labels with more than 1 intersecting detection
            print("Warning: multiple overlap")
            print(aux)
        allIntersections=np.where(isIntersect)[0]
        intersectingIndexes=np.array([])
        intersection=0                        #indices of the intersection selected
        labelIndexes=np.arange(label.startInd,label.stopInd)    #indices of the label we are checking
        for i in allIntersections:
            intersecting=annotations.loc[i]
            tempIndexes=np.arange(intersecting.startInd,intersecting.stopInd)
            tempIntersection=len(np.intersect1d(labelIndexes,tempIndexes))
            if (tempIntersection>intersection): #select only maximal overlap
                intersectingIndexes=tempIndexes
                labelIntersect=i
        IoU=len(np.intersect1d(labelIndexes,intersectingIndexes))/len(np.union1d(labelIndexes,intersectingIndexes))
    return IoU, labelIntersect

def byEventEvaluation(gtAnnotations,finalAnnotations,thres_IoU=0.3):
    gtAnnotations[['IoU','intersectsWith']]=gtAnnotations.apply(
        lambda row: labelIntersectionIoU(row,finalAnnotations)
        ,axis=1, result_type ='expand')
    # handle detections intersecting with multiple gt annotations -------------------------------------->
    nanindex=np.isnan(gtAnnotations.intersectsWith)
    np.unique(gtAnnotations.intersectsWith)
    notnan=np.array(gtAnnotations.index)[-nanindex]
    uniqueIntersect=np.unique(gtAnnotations.iloc[notnan].intersectsWith)
    if len(gtAnnotations.iloc[notnan])>len(uniqueIntersect):
        for val in uniqueIntersect:
            aux=gtAnnotations[gtAnnotations.intersectsWith==val].copy()
            if len(aux)>1:
                keep=aux.index[np.argmax(aux.IoU)]  #<- keep maximum IoU only
                for ind_aux,row in aux.iterrows():
                    if ind_aux!=keep:
                        gtAnnotations.at[ind_aux,"intersectsWith"]=np.NaN
    #<---------------------------------------------------------------------------------------------------
    annotationsNoOverlap=np.array(gtAnnotations[gtAnnotations.intersectsWith.isna()].index)
    gtAnnotationsOverlap=gtAnnotations.iloc[np.setdiff1d(np.array(gtAnnotations.index),annotationsNoOverlap)]
    detectionsOverThreshold=np.array(gtAnnotationsOverlap[gtAnnotationsOverlap.IoU>=thres_IoU].intersectsWith)
    detectionsBelowThreshold=np.array(gtAnnotationsOverlap[gtAnnotationsOverlap.IoU<thres_IoU].intersectsWith)
    detectionsConsidered=np.union1d(detectionsOverThreshold,detectionsBelowThreshold)
    detecionsNotConsidered=np.setdiff1d(
        np.array(finalAnnotations.index),
        detectionsConsidered
    )
    #detection below threshold are a false positive and a false negative!
    tp=len(detectionsOverThreshold)
    fp=len(detecionsNotConsidered)+len(detectionsBelowThreshold)
    fn=len(annotationsNoOverlap)+len(detectionsBelowThreshold)
    return tp,fp,fn

def getConfidence(label,annotations):
    isIntersect=annotations.apply(
        lambda row: ((label.startInd>=row.startInd)&(label.startInd<row.stopInd)) | ((row.startInd>=label.startInd)&(row.startInd<label.stopInd))
        ,axis=1)

    #output lenght should be equal 1 or 0 (at least now)
    aux=np.sum(isIntersect)
    if aux==0:
        raise Exception("Data inconsistency: all predictions must intersect with some candidate")
    else:
        allIntersections=np.where(isIntersect)[0]
        intersectingIndexes=np.array([])
        for i in allIntersections:
            intersecting=annotations.loc[i]
            tempIndexes=np.arange(intersecting.startInd,intersecting.stopInd)
            intersectingIndexes=np.concatenate((intersectingIndexes,tempIndexes))
        labelIndexes=np.arange(label.startInd,label.stopInd)
        #double check:
        union=np.union1d(labelIndexes,intersectingIndexes)
        if len(np.setdiff1d(labelIndexes,union))>0:
            raise Exception("Data inconsistency: all intersections must be contained in the prediction")
        confidence=len(np.intersect1d(labelIndexes,intersectingIndexes))/len(labelIndexes)
    return confidence

def labelingProcess(labelVector,maxTimeClose,minDuration,samplerate,verbose=0):
    aux=ndi.label(labelVector)
    preCandidates=ndi.find_objects(aux[0])
    if len(preCandidates)>0:    #consider the case of no precandidates at all
        if verbose>0:
            print("Number of raw candidates: "+str(len(preCandidates)))
        #1. Join candidates separated by less than the threshold
        kernelLength=int(maxTimeClose*samplerate)
        if kernelLength>0:
            kernel=np.ones((kernelLength,))
            labelVector=ndi.binary_closing(labelVector,kernel)
        #2. Discard candidates under minimum duration
        aux=ndi.label(labelVector)
        candidates=ndi.find_objects(aux[0])
        labelVector=np.zeros_like(labelVector)
        if len(candidates)>0:
            durations=np.apply_along_axis(lambda x: (x[0].stop-x[0].start)/samplerate,1,candidates)
            detections=[candidates[i] for i in np.where(durations>minDuration)[0]]
            for detection in detections:
                labelVector[detection]=1
            if verbose>0:
                print("Number of detections: "+str(len(detections)))
    return labelVector

def getFilepath(window,subject,characteristic,bandName,samplerate,featurespath):
    filename=str(window)+'_'+subject+'_'+characteristic+'_'+bandName+'.fd'
    filepath=featurespath+'/'+str(samplerate)+'fs/'+str(window)+'win/'+subject+'/'+filename
    return filepath

def saveFeature(vector,window,subject,characteristic,bandName,samplerate,featurespath):
    filepath=getFilepath(window,subject,characteristic,bandName,samplerate,featurespath)
    if not os.path.exists(filepath):
        print('saving feature: '+filepath)
        dumpPickle(filepath,vector)

def loadFeature(window,subject,characteristic,bandName,samplerate,featurespath):
    filename=str(window)+'_'+subject+'_'+characteristic+'_'+bandName+'.fd'
    filepath=featurespath+'/'+str(samplerate)+'fs/'+str(window)+'win/'+subject+'/'+filename
    vector=loadPickle(filepath)
    return vector

def loadFeatureMatrix(subjectList,featureSelection,signalsMetadata,samplerate,datapath):
    featureSelection=featureSelection.reset_index(drop=True)
    # operate on the signal lengths of the subjects selected
    thisSignals=signalsMetadata[signalsMetadata.subjectId.isin(subjectList)].copy()
    thisSignals['excerptDimension']=thisSignals.apply(
        lambda row: int(row.duration*samplerate),
        axis=1)
    # initialise the feature matrix
    featureMatrix=np.zeros((np.sum(thisSignals.excerptDimension),len(featureSelection)))
    # fill the matrix
    for i,feature in featureSelection.iterrows():   #iterate the featrures
        characteristic=feature['characteristic']
        bandName=feature['bandName']
        window=feature['window']
        featureValue=np.zeros_like(featureMatrix[:,i])  #initialise a row
        auxStartInd=0
        for j, row in thisSignals.iterrows():   #iterate the signals selected to fill the row
            subject=row['subjectId']
            excerptDim=row['excerptDimension']
            featurespath=datapath+"/"+row.database+"/features"
            featureValue[auxStartInd:auxStartInd+excerptDim]=loadFeature(str(window),subject,characteristic,bandName,str(samplerate),featurespath)
            auxStartInd=auxStartInd+excerptDim
        featureMatrix[:,i]=featureValue #   fill a row
    return featureMatrix

def excerptAnnotationsToLabels(excerptAnnotations,excerptDimension):
    excerptLabels=np.zeros((excerptDimension,))
    for index,row in excerptAnnotations.iterrows():
        excerptLabels[int(row.startInd):int(row.stopInd)]=1 # int casting to avoid exceptions when converting back and forth
    return excerptLabels

def loadLabelsVector(subjectList,annotations,signalsMetadata,samplerate):
    #WARNING: annotations must be filtered or union criterium will be used
    # operate on the signal lengths of the subjects selected
    thisSignals=signalsMetadata[signalsMetadata.subjectId.isin(subjectList)].copy()
    thisSignals['excerptDimension']=thisSignals.apply(
        lambda row: int(row.duration*samplerate),
        axis=1)
    # initialise vector of labels
    labelsVector=np.zeros((np.sum(thisSignals['excerptDimension']),))
    auxStartInd=0
    # iterate the signals
    for j, row in thisSignals.iterrows():   #iterate the signals selected to fill the row
        subject=row['subjectId']
        excerptDim=row['excerptDimension']
        thisAnnotations=annotations[annotations.subjectId==subject] #filter the annotations
        labelsVector[auxStartInd:auxStartInd+excerptDim]=excerptAnnotationsToLabels(thisAnnotations,excerptDim) # process all annotations of a given subject
        auxStartInd=auxStartInd+excerptDim
    return labelsVector

def labelVectorToAnnotations(labelVector,samplerate):
    aux=ndi.label(labelVector)
    detections=ndi.find_objects(aux[0])
    startInds=np.apply_along_axis(lambda x: x[0].start,1,detections)
    stopInds=np.apply_along_axis(lambda x: x[0].stop,1,detections)
    durations=np.apply_along_axis(lambda x: (x[0].stop-x[0].start)/samplerate,1,detections)
    outputAnnotations=pd.DataFrame({
        "startInd":startInds,
        "stopInd":stopInds,
        "duration":durations
    })
    return outputAnnotations
#_________________________________________________________________________

#_________________________________________________________________________
#__ EXPERIMENT & MODEL MANAGEMENT ________________________________________
def loadExperiment(experimentId,datapath):
    experimentModels=loadPickle(datapath+"/experiments/"+experimentId+"/experimentModels.pkl")
    if os.path.exists(datapath+"/experiments/"+experimentId+"/featureSelection.pkl"):
        featureSelection=loadPickle(datapath+"/experiments/"+experimentId+"/featureSelection.pkl")
        return experimentModels, featureSelection
    else:
        return experimentModels

def loadModel(modelId,experimentId,datapath):   #TBD: this should be generic
    model=xgb.XGBClassifier()
    model.load_model(datapath+"/experiments/"+experimentId+"/"+modelId+".json")
    return model

def loadBooster(modelId,experimentId,datapath):
    model=xgb.Booster()
    model.load_model(datapath+"/experiments/"+experimentId+"/"+modelId+".json")
    return model
#_________________________________________________________________________

#_________________________________________________________________________
#__ METRICS ______________________________________________________________
def getIou(coordA,coordB):
    if (coordA[1]<coordB[0])|(coordB[1]<coordA[0]): #NOT INTERSECTION
        iou=0
    else:   #INTERSECTION
        if (coordA[0]<coordB[0])&(coordA[1]>coordB[1]):     #B included in A:
            inter=coordB[1]-coordB[0]
            union=coordA[1]-coordA[0]
        elif (coordB[0]<coordA[0])&(coordB[1]>coordA[1]):   #A included in B
            inter=coordA[1]-coordA[0]
            union=coordB[1]-coordB[0]
        else:
            inter=np.min((coordA[1]-coordB[0],coordB[1]-coordA[0]))
            union=np.max((coordA[1]-coordB[0],coordB[1]-coordA[0]))
        iou=inter/union
    return iou

def getIoUmatrix(annotations,detections):
    #get the coords
    gtCoords=zip(annotations.startInd,annotations.stopInd)
    outCoords=zip(detections.startInd,detections.stopInd)
    #calculate the iou vector
    iouVector=np.array(list(itt.starmap(getIou,itt.product(gtCoords,outCoords))))
    #reshape to a matrix
    iouMatrix=iouVector.reshape(len(annotations),len(detections))
    return iouMatrix

def IoUmatrixToF1(iouMatrix,thresIoU=0.3):
    #binarize
    binarized=iouMatrix>thresIoU
    #calculateF1
    outF1=(np.sum(np.max(binarized,axis=0))+np.sum(np.max(binarized,axis=1)))/(iouMatrix.shape[0]+iouMatrix.shape[1])
    return outF1

def annotationPairToMetrics(annotations,detections,thresIoU=0.3):
    #get the coords
    gtCoords=zip(annotations.startInd,annotations.stopInd)
    outCoords=zip(detections.startInd,detections.stopInd)
    #calculate the iou vector
    iouVector=np.array(list(itt.starmap(getIou,itt.product(gtCoords,outCoords))))
    #reshape to a matrix
    iouMatrix=iouVector.reshape(len(annotations),len(detections))
    #binarize
    binarized=iouMatrix>thresIoU
    #calculateF1
    outF1=(np.sum(np.max(binarized,axis=0))+np.sum(np.max(binarized,axis=1)))/(len(annotations)+len(detections))
    recall=np.sum(np.max(binarized,axis=1))/len(annotations)
    precision=np.sum(np.max(binarized,axis=0))/len(detections)
    return outF1,recall,precision

def getMetricTables(annotations,detections,thresIoU=0.3):
    #get the coords
    gtCoords=zip(annotations.startInd,annotations.stopInd)
    outCoords=zip(detections.startInd,detections.stopInd)
    #calculate the iou vector
    iouVector=np.array(list(itt.starmap(getIou,itt.product(gtCoords,outCoords))))
    #reshape to a matrix
    iouMatrix=iouVector.reshape(len(annotations),len(detections))
    #create tables
    index0=np.apply_along_axis(np.argmax,0,iouMatrix)
    iou0=np.apply_along_axis(np.max,0,iouMatrix)
    index1=np.apply_along_axis(np.argmax,1,iouMatrix)
    iou1=np.apply_along_axis(np.max,1,iouMatrix)
    
    tableOut=pd.DataFrame({
        'indexGT':index0,
        'iou':iou0
    })

    tableGT=pd.DataFrame({
        'indexOut':index1,
        'iou':iou1
    })

    tableOut['type']='out'
    tableGT['type']='gt'
    tableOut['indexOut']=tableOut.index
    tableGT['indexGT']=tableGT.index
    #Correct external indexes of objects not overlapping
    tableOut.loc[tableOut.iou==0,'indexGT']='NA'
    tableGT.loc[tableGT.iou==0,'indexOut']='NA'
    #set tps
    tableOut['tp']=tableOut.iou>thresIoU
    tableOut['fp']=tableOut.iou<=thresIoU
    tableGT['tp']=tableGT.iou>thresIoU
    tableGT['fn']=tableGT.iou<=thresIoU

    return tableOut,tableGT

def annotationPairToGraph(annotations,detections,thresIoU=0.3):
    #get the coords
    gtCoords=zip(annotations.startInd,annotations.stopInd)
    outCoords=zip(detections.startInd,detections.stopInd)
    #calculate the iou vector
    iouVector=np.array(list(itt.starmap(getIou,itt.product(gtCoords,outCoords))))
    #reshape to a matrix
    iouMatrix=iouVector.reshape(len(annotations),len(detections))
    #create tables
    index0=np.apply_along_axis(np.argmax,0,iouMatrix)
    iou0=np.apply_along_axis(np.max,0,iouMatrix)
    index1=np.apply_along_axis(np.argmax,1,iouMatrix)
    iou1=np.apply_along_axis(np.max,1,iouMatrix)
    
    tableOut=pd.DataFrame({
        'indexGT':index0,
        'iou':iou0
    })

    tableGT=pd.DataFrame({
        'indexOut':index1,
        'iou':iou1
    })

    tableOut['type']='out'
    tableGT['type']='gt'
    tableOut['indexOut']=tableOut.index
    tableGT['indexGT']=tableGT.index
    #Correct external indexes of objects not overlapping
    tableOut.loc[tableOut.iou==0,'indexGT']='NA'
    tableGT.loc[tableGT.iou==0,'indexOut']='NA'
    #set tps
    tableOut['tp']=tableOut.iou>thresIoU
    tableOut['fp']=tableOut.iou<=thresIoU
    tableGT['tp']=tableGT.iou>thresIoU
    tableGT['fn']=tableGT.iou<=thresIoU
    #calculate metrics
    recall=np.sum(tableGT['tp'])/len(tableGT)
    precision=np.sum(tableOut['tp'])/len(tableOut)
    f1=(np.sum(tableGT['tp'])+np.sum(tableOut['tp']))/(len(tableGT)+len(tableOut))
    #concatenate tables
    appended=pd.concat(objs=(tableOut,tableGT),axis=0)
    #modify values
    appended['x']=appended['indexGT']
    appended['y']=appended['indexOut']
    appended.loc[((appended.type=='out')&(~ appended.tp)),'x']=-10
    appended.loc[((appended.type=='gt')&(~ appended.tp)),'y']=-10
    appended['size']=1
    appended.loc[appended.type=='out','size']=3
    #create the graph
    minTPIoU=np.min(appended[appended.tp].iou)
    fig=px.scatter(appended,x='x',y='y',color='iou',symbol='type',
    opacity=0.8,symbol_map={'out':'circle-open','gt':'circle'},size='size',
    color_continuous_scale=
        ((0.0, 'rgb(40,40,40)'),
        (0.000001, 'rgb(28,227,255)'),
        (0.14, 'rgb(56,199,255)'),
        (0.29, 'rgb(85,170,255)'),
        (0.42, 'rgb(113,142,255)'),
        (0.57, 'rgb(142,113,255)'),
        (0.71, 'rgb(170,85,255)'),
        (0.86, 'rgb(199,56,255)'),
        (1.0, 'rgb(227,28,255)')),
    range_x=(-20,len(tableGT)+10),range_y=(-20,len(tableOut)+10),
    title='by-Event evaluation summary<br><sup>F1(@IoU>'+str(thresIoU)+')='+str(round(f1,4))+' | minimum TP IoU: '+str(round(minTPIoU,4))+'</sup>',
    hover_data={'x':False,
    'y':False,
    'tp':False,
    'fp':False,
    'fn':False,
    'size':False,
    'type':False,
    'iou':':.4f', # customize hover for column of y attribute
    'indexGT':True,
    'indexOut':True
    })
    for t in fig.data:
        t.marker.line.width = 2
    fig.update_xaxes(title_text=str(len(tableGT))+' ANNOTATIONS | recall(@IoU>'+str(thresIoU)+')= '+str(round(recall,4)))
    fig.update_yaxes(title_text=str(len(tableOut))+' DETECTIONS | precision(@IoU>'+str(thresIoU)+')= '+str(round(precision,4)))
    fig.add_vline(x=-5,line_dash='dash')
    fig.add_hline(y=-5,line_dash='dash')
    #----------------------------------------------------------------------->
    # https://stackoverflow.com/questions/61827165/plotly-how-to-handle-overlapping-colorbar-and-legends
    # @vestland answer
    """ fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
                                            ticks="outside",
                                            ticksuffix=" bills")) """
    # @bitbang answer
    fig.update_layout(legend_orientation="h")
    #<----------------------------------------------------------------------
    return fig
#_________________________________________________________________________
