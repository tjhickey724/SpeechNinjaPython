import sounddevice as sd
import math
import matplotlib.pyplot as plt
import numpy as np

def record():
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 2
    print("start talking");
    myrecording = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE,
                     channels=CHANNELS, blocking=True, dtype='float64')
    print("stop talking");
    #for x in myrecording:
    #	print(str(x[0])+" "+str(x[1]));
    #let's find the maximum value
    maxval=0;
    for x in myrecording:
        if abs(x[0])>maxval:
		          maxval= abs(x[0]);

    #print(maxval);

    # calculate the cutoff value
    cutoff = maxval*0.05;

    #initialize counter variable
    i=1000

    # find the first index where the value is above cutoff
    while abs(myrecording[i][0]) < cutoff:
	       i = i+1

    start = i;

    i=len(myrecording)-1;
    # find the last index where the value is above cutoff
    while abs(myrecording[i][0]) < cutoff:
	       i = i-1
    last = i;

    print("start = "+str(start));
    print("end = "+str(last));

    duration = last-start;
    # find RMS of middle third, from start + duration/3 to start + 2*duration/3
    m1 = start + duration//3
    m2 = start + 2*duration//3
    sum=0;
    for i in range(m1,m2):
	       val = myrecording[i][0];
	       sum = sum + val*val;
    sum = sum/(duration//3);
    rms = math.sqrt(sum)

    print("rms = "+str(rms));

    leftside=[]
    for x in range(0,len(myrecording)-1):
	       leftside.append(myrecording[x][0])

    return leftside[start:last]


def bandpassfilter2(flo,fhi,tb):
    """ returns a bandpass filter """
    #fL = 0.1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    #fH = 0.4  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    fL = flo/44100.0;
    fH = fhi/44100.0;
    b = tb/44100.0; #0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (n - (N - 1) / 2.))
    hlpf *= np.blackman(N)
    hlpf = hlpf / np.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (n - (N - 1) / 2.))
    hhpf *= np.blackman(N)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = -hhpf
    hhpf[(N - 1) // 2] += 1
    bp = np.convolve(hlpf,hhpf);
    print(len(bp));
    print(bp);
    return bp;

def applyBandPassFilter2(flo,fhi,tb,signal):
    bp = bandpassfilter2(flo,fhi,tb)
    signal2 = np.convolve(pad(signal,len(bp)),bp);
    N=int((len(signal2)-len(signal))/2);
    ss = signal2[N:-N];
    print("N = ");
    print(N);
    #print(signal);
    #print(signal2);
    #print(ss);
    #print([flo,fhi]);
    return signal2[N:-N];

def highpassfilter2(flo,fhi,tb):
    """ returns a bandpass filter """
    #fL = 0.1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    #fH = 0.4  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    fL = flo/44100.0;
    fH = fhi/44100.0;
    b = tb/44100.0; #0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    # Compute a low-pass filter with cutoff frequency fH.
    hlpf = np.sinc(2 * fH * (n - (N - 1) / 2.))
    hlpf *= np.blackman(N)
    hlpf = hlpf / np.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency fL.
    hhpf = np.sinc(2 * fL * (n - (N - 1) / 2.))
    hhpf *= np.blackman(N)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = -hhpf
    hhpf[(N - 1) // 2] += 1
    bp = hhpf; #np.convolve(hlpf,hhpf);
    print(len(bp));
    print(bp);
    return bp;

def applyHighPassFilter2(flo,fhi,tb,signal):
    bp = highpassfilter2(flo,fhi,tb)
    signal2 = np.convolve(pad(signal,len(bp)),bp);
    N=int((len(signal2)-len(signal))/2);
    ss = signal2[N:-N];
    print("N = ");
    print(N);
    #print(signal);
    #print(signal2);
    #print(ss);
    #print([flo,fhi]);
    return signal2[N:-N];

def lowpassfilter(cutoff_freq):
    fc = cutoff_freq/(0.5*44100) #0.1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.

    n = np.arange(N)

    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2.))

    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))

    # Multiply sinc filter with window.
    h = h * w

    # Normalize to get unity gain.
    h = h / np.sum(h)
    hlowpass = list(h);
    #print(len(hlowpass));
    return hlowpass

def applyLowPassFilter(cutoff_freq,signal):
    return np.convolve(signal,lowpassfilter(cutoff_freq));

def applyHighPassFilter(cutoff_freq,signal):
    return np.convolve(signal,highpassfilter(cutoff_freq));

def applyBandPassFilter(cutoff_lo,cutoff_hi,signal):
    a = np.convolve(signal,lowpassfilter(cutoff_lo));
    b = np.convolve(a,highPassFilter(cutoff_hi));
    return b;

def applyLowPassFilter(f,signal):
    result = np.convolve(signal,f);
    return result;

def highpassfilter(cutoff_freq):
    fc = cutoff_freq/(0.5*44100)#0.1  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
    b = 0.08  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.

    n = np.arange(N)

    # Compute sinc filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2.))

    # Compute Blackman window.
    w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        0.08 * np.cos(4 * np.pi * n / (N - 1))

    # Multiply sinc filter with window.
    h = h * w

    # Normalize to get unity gain.
    h = h / np.sum(h)

    # create a high pass filter
    h = -h

    h[int((N - 1) / 2)] += 1
    hhighpass = h;
    #print(len(hhighpass));
    return hhighpass



def getRMS(leftsidefinal):
    #let's find the maximum value
    maxval=0;
    for x in leftsidefinal:
        if abs(x)>maxval:
		          maxval= abs(x);

    # calculate the cutoff value
    cutoff = maxval*0.05;

    #initialize counter variable
    i=1000

    # find the first index where the value is above cutoff
    while abs(leftsidefinal[i]) < cutoff:
	       i = i+1

    start = i;

    i=len(leftsidefinal)-1;
    # find the last index where the value is above cutoff
    while abs(leftsidefinal[i]) < cutoff:
	       i = i-1
    last = i;

    duration = last-start;
    # find RMS of middle third, from start + duration/3 to start + 2*duration/3
    m1 = start + duration//3
    m2 = start + 2*duration//3
    sum=0;
    for i in range(m1,m2):
	       val = leftsidefinal[i];
	       sum = sum + val*val;
    sum = sum/(duration//3);
    rms = math.sqrt(sum)
    return rms

def main():
    leftside=record()
    hlowpass1000=lowpassfilter(1000)
    hhighpass400=highpassfilter(400)
    hlowpass400=lowpassfilter(400)
    hhighpass20=highpassfilter(20)
    hlowpass20=lowpassfilter(20)

    #original
    plt.subplot(2,3,1);
    plt.plot(leftside);

    #lowpassNasal
    plt.subplot(2,3,2);
    leftsidelow1000 = np.convolve(leftside,hlowpass1000);
    plt.plot(leftsidelow1000);

    #passfinalNasal
    plt.subplot(2,3,3);
    nasal= np.convolve(leftsidelow1000,hhighpass400);
    plt.plot(nasal);

    #lowpassAcoustic
    plt.subplot(2,3,5)
    leftsidelowAcoustic = np.convolve(leftside,hlowpass400);
    plt.plot(leftsidelowAcoustic);

    #passfinalAcoustic
    plt.subplot(2,3,6);
    acoustic= np.convolve(leftsidelowAcoustic,hhighpass20);
    plt.plot(acoustic);

    #lowpass frequency
    #plt.subplot(2,4,6);
    #plt.plot(hlowpassNasal);

    #highpass frequency
    #plt.subplot(2,4,7);
    #plt.plot(hhighpass400);
    print("rmsNasal = "+str(getRMS(nasal)));
    print("rmsAcoustic = "+str(getRMS(acoustic)));

    plt.show()

def f(k,t):
    """ f(k,t) - evaluates a sin function of period k at time t """
    return math.sin(2*math.pi*k*t);

def g(t):
    """ g(t) sums four sine functions of periods 1, 400, 800, 5000 at time t """
    #return f(1,t) + 0.4*f(400,t)+0.1*f(800,t)+0.04*f(5000,t);
    return f(10,t) + f(400,t)+f(800,t)+f(5000,t);

def signal1(N):
    """ signal1(N) - creates a signal of length N using the function g """
    result=[];
    for x in range(0,N):
        result.append(g(x/44100.0));
    return result;

def signal0(N):
    """ signal1(N) - creates a signal of length N using the function g """
    result=[];
    for x in range(0,N):
        result.append(f(10,(x/44100.0)));
    return result;

def subtract(a,b):
    """ subtract(a,b) - returns the array a-b"""
    c = [];
    N = len(a);
    for x in range(0,N):
        c.append(a[x]-b[x]);
    return c;

def clamp(lo,hi,a):
    """ subtract(a,b) - returns the array a-b"""
    c = [];
    N = len(a);
    for x in range(0,N):
        c.append(max(lo,min(hi,a[x])));
    return c;

def subtract1(a,b):
    """ subtract1(a,b) - returns the array a-20*b"""
    c = [];
    N = len(a);
    for x in range(0,N):
        c.append(a[x]-20*b[x]);
    return c;

def addList(a,b):
    """ addList(a,b) - returns a+b where a,b are viewed as vectors. """
    c=[];
    for x in range(0,len(a)):
        c.append(a[x]+b[x]);
    return c;

def pad(signal,k):
    """ pad(signal,k) -- returns a list obtained by adding k empty values before and after the signal
    """
    buffer1 = np.empty(k);
    list1 = buffer1.tolist();
    list1.extend(signal);
    list1.extend(buffer1.tolist());
    return list1;

def test1():
    N=4410;
    M=4410;
    signal= signal1(M);
    sig0 = signal0(M);
    sig400 = applyBandPassFilter2(200,600,100,signal);
    sig800 = applyBandPassFilter2(600,1000,100,signal);
    sig2000= applyBandPassFilter2(1000,10000,100,signal);
    sighp200 = clamp(-10,10,applyHighPassFilter2(200,10000,100,signal));
    siglp200 = clamp(-10,10,subtract(signal,sighp200));
    sig0Err =  subtract(sig0,siglp200);

    #hp = applyHighPassFilter(1000,signal);
    #lp = subtract1(signal,hp);
    print("plot signal");
    # Plot the original signal2
    rows=2;cols=4
    plt.subplot(rows,cols,1);
    plt.plot(signal);
    print("plot sig400");
    # isolate the 400 Hz signal and plot
    plt.subplot(rows,cols,2);

    plt.plot(sig400);
    print("plot sig800");
    # isolate the 800 Hz signal and plot
    plt.subplot(rows,cols,3);

    plt.plot(sig800);

    print('plot sig2000');
    # isolate the 5000 Hz signal and plot
    plt.subplot(rows,cols,4);
    plt.plot(sig2000);

    print('plot sig400+sig800+sig2000');
    # add the 400, 800, and 5000 signals back and plot
    plt.subplot(rows,cols,5);
    #sig00 = clamp(-5,5,addList(sig400,addList(sig800,sig2000)));
    #plt.plot(sig00);
    plt.plot(sig0Err);

    print("plot sig001");
    plt.subplot(rows,cols,6);
    #sig001 = clamp(-5,5,subtract(signal,sig0));
    #plt.plot(sig001);
    plt.plot(sig0);

    plt.subplot(rows,cols,7);

    plt.plot(sighp200);

    plt.subplot(rows,cols,8);
    plt.plot(siglp200);
    plt.show();

print(subtract([1,2,3,4,5],[3,3,3,0,0]))

def clampIt(signal):
    return clamp(-1,1,signal);

def test2():
    signal=record();
    sig0 = clampIt(applyBandPassFilter2(100,1000,100,signal));
    sighp = clampIt(applyHighPassFilter2(500,10000,100,signal));
    siglp = clampIt(subtract(signal,sighp));
    rows=4
    cols=1
    plt.subplot(rows,cols,1);
    plt.plot(signal);

    plt.subplot(rows,cols,2);
    plt.plot(sig0);

    plt.subplot(rows,cols,3);
    plt.plot(sighp);

    plt.subplot(rows,cols,4);
    plt.plot(siglp);

    plt.show();

def main2():
    leftside=record()
    hlowpass1000=lowpassfilter(1000)
    hhighpass400=highpassfilter(400)
    hlowpass400=lowpassfilter(400)
    hhighpass20=highpassfilter(20)
    hlowpass20=lowpassfilter(20)

    #original
    plt.subplot(2,3,1);
    plt.plot(leftside);

    #lowpassNasal
    plt.subplot(2,3,2);
    leftsidelow1000 = np.convolve(leftside,hlowpass1000);
    plt.plot(leftsidelow1000);

    #passfinalNasal
    plt.subplot(2,3,3);
    nasal= np.convolve(leftsidelow1000,hhighpass400);
    plt.plot(nasal);

    #lowpassAcoustic
    plt.subplot(2,3,5)
    leftsidelowAcoustic = np.convolve(leftside,hlowpass400);
    plt.plot(leftsidelowAcoustic);

    #passfinalAcoustic
    plt.subplot(2,3,6);
    acoustic= np.convolve(leftsidelowAcoustic,hhighpass20);
    plt.plot(acoustic);

    #lowpass frequency
    #plt.subplot(2,4,6);
    #plt.plot(hlowpassNasal);

    #highpass frequency
    #plt.subplot(2,4,7);
    #plt.plot(hhighpass400);
    print("rmsNasal = "+str(getRMS(nasal)));
    print("rmsAcoustic = "+str(getRMS(acoustic)));

    plt.show()

test2()
#main()
