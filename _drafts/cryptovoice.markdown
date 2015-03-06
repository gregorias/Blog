---
layout: post
title:  "Lessons in writing and testing an audio modem in Python"
date:   2015-03-05 20:00:00
uses_mathjax: true
tags: blog mathematics programming ft
---
In one of my current prototype projects I need to encode voice into binary data
and then later transfer it through a voice channel like Skype. This requires me
to push my limits and learn new things which I will describe here. This post
will explain technologies I use, rationale behind them, and give basic advice on
how to use them. First I will give a quick overview behind them, in the next
section I will describe what basic theory around modulation and demodulation,
then I will describe how to combine Cython and profiling to build a flexible and
efficient Python application. In the last section I will talk about pulseaudio
and how it allows me to perform user testing without using external device.

Overview
========

Modulation and demodulation
===========================
The module that I building encodes and decodes voice. The encoder has one input:
a continuous stream of voice encoded in audio samples. On the output it plays
audio that is transmitted by an independent voice communicator. The decoder
does the reverse.

What makes this task non-trivial is that in the middle the voice is encrypted,
thereby producing a seemingly random bytestring. This task includes a few
challenges:

* How one should modulate the bytestring so that voice communicators do not
  distort it?
* Given that voice channels have limited capacity how I should encode the human
  voice so that it fits?
* How can I effeciently demodulate the modulated signal?

Basics of audio signals
-----------------------
The sound that we hear is a result of a mechanical wave of pressure and
displacement propagating through air. As such we can represent any sound as
wave. Wave as a continuous signal needs to be discretized to be representable in
computer memory and there are two ways to go about it.

First we can represent a wave as a sum of its basal frequencies. It is a
mathematical fact that:

> Every continuous function on closed interval can be represented as a
> (potentially infinite) sum of basic frequencies, that is:

$$
f(x) : [0,1] \rightarrow \mathbb{R}
$$

$$
f(x) = \sum_{i=0}^\infty \left(A_i \sin(ix) + B_i \cos(ix)\right)
$$

Given that we might keep a table of coefficients $$A_i, B_i$$ for all audible
freuencies. Humans can hear up to 20kHZ, so the size of this table goes up to
20000 thousand for each time frame.

This representation may allow for easier reproduction of signal through
mechanical means (combine all those basal frequencies and you are done), but is
impractical.

The second raw representation uses samples of the wave taken at specified
intervals. For example a typical .wav file, which uses this format, stores an
amplitude of the wave for every $$\frac{1}{44100}$$ part of a second. The
frequency of sampling is called a **sampling rate**. The sampling rate used in
telephony is 8000Hz. The reasoning behind the value of the sampling rate 

Both representations are equivalent in the sense that one can transform from one
to the other. From basal frequencies we can produce sampling by simply
calculating the wave value at given points. To get to the other way around we
need to use the discrete Fourier transform.

Discrete Fourier transform
--------------------------
Fourier transform is a general mathematical tool which transform a wave function
into a function of fundamental frequencies. That is the resulting function maps
each of its argument to its power inside the argument. The discrete version of
the Fourier transform maps list of samples of the wave to values of fundamental
frequencies of the wave that generated them. The formula looks as this:

$$X_k = \sum_{i=0}^{\left\lceil\frac{N}{2}\right\rceil} x_k e^{-i2\pi k n / N}$$

Where $$x_k$$ are samples and $$X_k$$ is a complex number representing
fundamental coefficients in the form of: $$\frac{B_k}{2} + \frac{A_k}{2} i$$ if
$$k > 0$$ and $$X_0 = B_0$$. Note that I am using
$$\left\lceil\frac{N}{2}\right\rceil$$ as the upper limit of the sum instead of
often seen $$N-1$$. That's because further numbers do not give any more
information. Naively calculating the DFT takes $$O\left(N^2\right)$$ time.
Fortunately there exists a fast Fourier transform algorithm which takes
$$O\left(N \log N\right)$$.

Originally I planned to make an entire post about the transform, but I don't
think I have so much time I hand plus there are obviously much better resources
available on the internet. I recommend [this site][fft].

The sampling rate that we want is determined by Nyquist-Shannon theorem

> If a function x(t) contains no frequencies higher than B Hz, it is completely
> determined by giving its ordinates at a series of points spaced 1/(2B) seconds
> apart.

That means that if we want reliably represent frequencies up to B Hz we need
have twice that sampling rate and that's enough. Since telephony was meant to
transmit voice and human voice happens in a range up to 300Hz then 8000 sampling
rate was more than enough.

Modulating data
---------------

Now we know enough to understand important tradeoffs of signal modulation. Our
task now is to encode binary data into a voice channel in such a way that the
channel's internal encoding will not distort the signal too much. Here I used
[Hermes][hermes]. It's a research protocol meant to encode data inside
telephony. The authors use relative frequency shift keying (RFSK) for
modulation. FSK is a modulation technique where we encode each bit as a signal
pulse of specific frequency. In Hermes we encode each bit as a relative change
of frequency, specifically 0 is a drop in frequency and 1 is a rise. In order to
not go outside the voice band we expand each original bit into 01 or 10
sequence. Combined with the fact that the telephony can reliably carry signal
only up to 4000 Hz and that each pulse needs to take at least $$\frac{1}{4000}$$
of a second this gives as a theoretical maximal throughput of 2000 kbps. Even so
authors have achieved sufficient reliability at 1200 kbps throughput.

The throughput required by telephony to carry voice is 64kpbs (8 bits per sample
at 8000 samples per second). Comparing it with 1200 kbps, we can get with
Hermes, it seems awfully lot and it is. It took me some time to find the
[state-of-the-art codec][codec2] which encodes voice with bitrate as low as
exactly 1200kbps. Talk about luck. To use it inside Python I had to use Cython
to provide a Python interface to codec2. The library is available on [my
Github][pycodec2]. I'm planning on releasing it on PyPi soon.

To be honest I have never used Foreign Function Interface inside any language,
which seems weird now. I had tried writing this prototype inside Haskell
before, but resigned due to lack of some libraries I would find useful. Now
after overcoming my reluctance to FFI I would simply write such a wrapper for
Haskell. End of diversion.

Demodulation
------------

So far I have described how the voice is encoded into a binary data and how then
this data is modulated. We are now left with possibly the most difficult part of
the process: demodulation. In the modulation phase we just sample the pulses of
different frequency. So in the demodulation phase we need to do something along
the lines of discovering when a pulse begins and what's its frequency or at
least whether it has risen or dropped.

### Using FFT

Initially I was ambitious and tried to do it correctly. I have researched what
kind of method would Digital Signal Processing books propose. Most of the
methods there I found impractical, because they either focused on transforming
the signal or seemed to be designed to be implemented in hardware. In the end I
used FFT to generate a spectrogram of input signal. A spectrogram is a result of
FFT mapped over time. The simplest way to get a spectrogram is to perform FFT
over an interesting window of samples. Additionally that window of samples is
multiplied by a Hamming function which is meant to mask any error at the
beginning or end of the pulse.

<figure>
<img src="/images/2015/03/Hamming.png" />
<figcaption>Hamming function</figcaption>
</figure>

How do I recognize begining of the pulse? In short I don't. I sweep the incoming
signal and let upper layers check how many errors they get. If the error rate is
small then I use that window for decoding until error rate goes up.

Aside from low efficiency of this method it is also inaccurate. At the rate
1200kbps my base frequency is 2400Hz, at sampling rate 44100 I then get
$$\frac{44100}{2400} \approx 18$$ samples per window. That means that if I
perform FFT on that window that each coefficient is responsible for a frequency
window of size $$\frac{44100}{2 \cdot 18} = 1225$$. That means I can not simply
find dominating frequency in the resulting table. I need to use some kind
weighted sum for comparison.

In the end I decided to abandon this method. It was too imprecise and
inefficient. Below I attack a snippet of code performing spectrogram calcualtion
(without some optimizations)

{% highlight python %}
def calculateSpectogram(audioSamples
    , frameDuration):
  """Given audio samples and duration of frame in seconds calculate a
  spectrogram matrix. The returning matrix has the result of successive DFTs in
  each column"""
  samples = audioSamples.samples
  sampleRate = audioSamples.sampleRate
  frameSize = round(sampleRate * frameDuration)
  frameCount = round(len(samples) / frameSize)
  cutOffPoint = round(MAX_AVAILABLE_FREQUENCY * frameDuration) + 2
  ftTable = []
  w = np.hamming(frameSize)
  currentFrame = 0
  while round((currentFrame + 1) * frameDuration * sampleRate) <= len(samples):
    begSampleIdx = round(currentFrame * frameDuration * sampleRate)
    endSampleIdx = begSampleIdx + frameSize
    windowedSample = samples[begSampleIdx : endSampleIdx] * w
    ftTable.append(sumFFTTable(np.array(abs(np.fft.fft(windowedSample)))/frameSize))
    ftTable[-1] = ftTable[-1][:cutOffPoint]
    ftTable[-1] = weightedSum(ftTable[-1])
    currentFrame += 1

  return np.array(ftTable)

def generateFrequencyExponent(frequency, sampleCount):
  exponents = np.linspace(0.0, 1.0, sampleCount + 1)[:sampleCount] \
              * 2 * math.pi \
              * frequency \
              * (0.0 + 1.0j)
  return math.e ** exponents

def myFFT(samples, cutOffPoint):
  resultLength = min(len(samples), cutOffPoint)
  result = np.array([0.0] * resultLength)
  for freqIdx in range(resultLength):
    pass
    #result[freqIdx] = generateFrequencyExponent(

# The two functions below are used for calculating a dominating frequency at
# each window using crude weighted sum

def sumFFTTable(fftTable):
  summedTable = [fftTable[0]]
  for i in range(1, int((len(fftTable) + 1)/2)):
    summedTable.append(fftTable[i] + fftTable[-i])
  return summedTable

def weightedSum(table):
  summed = 0.0
  tableSum = sum(table)
  for (i, elem) in enumerate(table):
    summed += (i  + 1) * elem
  return summed
{% endhighlight %}

### Zero counting

The most primitive method of estimating frequency is to estimate where the wave
function would have value zero and calculate the distances between consecutive
zeros. This method does not give us distribution of funcdamental frequencies and
is prone to data and numerical errors. However in this application, since the
only thing we need is the relation of frequencies in consecutive frames it turns
out to be quite precise. In fact, more precise than the previous one.

Additionally it is more efficient and fits better into the sweeping framework.
We can calculate the differences between zerocrossing beforehand and then
perform the sweeping method on the stream of those differences. We avoid
performing costly FFT for each missed sample or window.

Lesson? Don't discount simple ideas, because they are primitive. 

Cython
======

Profiling
=========

Pulseaudio
==========

[fft]: http://www.thefouriertransform.com/
[hermes]: https://cs.nyu.edu/~jchen/publications/com31a-dhananjay.pdf
[codec2]: http://www.rowetel.com/blog/?page_id=452
[pycodec2]: https://github.com/gregorias/pycodec2
