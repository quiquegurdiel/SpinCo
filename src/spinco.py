import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy import ndimage as ndi
import numpy.lib.stride_tricks as np_tricks
import bottleneck as btl
import scipy.interpolate as interp
import scipy.stats as sta
import itertools
import medusa.local_activation.nonlinear_parameters as mnl # :)
import medusa.local_activation.spectral_parameteres as msp # :)
import pickle as pkl

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
#_________________________________________________________________________


#__________________________________________________________________________
#__ MATHEMATICS ___________________________________________________________
def deltaSignChanges(vector,window):
    #-> Credits to Mark Byers here: https://stackoverflow.com/questions/2936834/python-counting-sign-changes
    lambdaSignChanges = lambda segment: len(list(itertools.groupby(segment, lambda x: x > 0)))-1
    #<-
    halfWindow=window2half(window)
    vector=padVectorBothSides(vector,halfWindow,'closest')
    view=np_tricks.sliding_window_view(vector,(window,))
    delta=list(itertools.starmap(lambdaSignChanges,zip(view)))
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
    mskew=list(itertools.starmap(lambdaSkew,zip(view)))
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
    mkurt=list(itertools.starmap(lambdaKurt,zip(view)))
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
    se=list(itertools.starmap(lambdaCTM,zip(view)))
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
    se=list(itertools.starmap(lambdaSpecE,zip(view)))
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
    se=list(itertools.starmap(lambdaSpecE,zip(view)))
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
    se=list(itertools.starmap(lambdaSampE,zip(view,stdVector)))
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
    lzc=list(itertools.starmap(lambdalZiv,zip(view,medianVector)))
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
#__ TIME-WINDOW METRICS __________________________________________________
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

def IoUfromLabel(label,annotations,verbose=1):  #<- TBD, optimise this, extremelly unefficient
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
#__ DREAMS DB ____________________________________________________________
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

    return signals, annotations, signalsMetadata, subjectsMetadata, annotationsMetadata
#_________________________________________________________________________

#_________________________________________________________________________
#__ FEATURE & LABEL MANAGEMENT ___________________________________________
def saveFeature(vector,window,subject,characteristic,bandName,samplerate,featurespath):
    filename=str(window)+'_'+subject+'_'+characteristic+'_'+bandName+'.fd'
    filepath=featurespath+'/'+str(samplerate)+'fs/'+str(window)+'win/'+subject+'/'+filename
    with open(filepath, 'wb') as file:
        pkl.dump(vector, file)

def loadFeature(window,subject,characteristic,bandName,samplerate,featurespath):
    filename=str(window)+'_'+subject+'_'+characteristic+'_'+bandName+'.fd'
    filepath=featurespath+'/'+str(samplerate)+'fs/'+str(window)+'win/'+subject+'/'+filename
    cFile = open(filepath, 'rb')
    vector = pkl.load(cFile)
    cFile.close()
    return vector

def loadFeatureMatrix (subjectList,featuresDataframe,excerptDimension,window,samplerate,featurespath):
    featureMatrix=np.zeros((len(subjectList)*excerptDimension,len(featuresDataframe)))
    for i,row in featuresDataframe.iterrows():
        characteristic=row['characteristic']
        bandName=row['bandName']
        featureValue=np.zeros_like(featureMatrix[:,0])  #initialise a row
        for j in range(len(subjectList)):
            subject=subjectList[j]
            featureValue[excerptDimension*j:excerptDimension*(j+1)]=loadFeature(str(window),subject,characteristic,bandName,str(samplerate),featurespath)
        featureMatrix[:,i]=featureValue
    return featureMatrix

def excerptAnnotationsToLabels(excerptAnnotations,excerptDimension):
    excerptLables=np.zeros((excerptDimension,))
    for index,row in excerptAnnotations.iterrows():
        excerptLables[int(row.startInd):int(row.stopInd)]=1 #avoid exceptions when converting back and forth
    return excerptLables

def loadLabelsVector(subjectList,annotations,excerptDimension):
    labelsVector=np.zeros((len(subjectList)*excerptDimension,))
    for j in range(len(subjectList)):
        subject=subjectList[j]
        thisAnnotations=annotations[annotations.subjectId==subject].reset_index()
        labelsVector[excerptDimension*j:excerptDimension*(j+1)]=excerptAnnotationsToLabels(thisAnnotations,excerptDimension)
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
