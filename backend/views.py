from django.views.generic import TemplateView
from django.views.decorators.cache import never_cache

from django.http import HttpResponse
from django.http import JsonResponse

import matplotlib.image as mpimg

import numpy as np
from numpy import sum, isrealobj, sqrt
from numpy.random import standard_normal

from scipy.io import wavfile
import scipy.io
from scipy import signal, stats

from PIL import Image

from io import BytesIO

import math

import os.path

import json

import base64

BASE = os.path.dirname(os.path.abspath(__file__))

# Serve Single Page Application
index = never_cache(TemplateView.as_view(template_name='index.html'))

IMAGE_PATH = os.path.join(BASE, 'resources/wm.bmp')
AUDIO_PATH = os.path.join(BASE, 'resources/100grand.wav')

def time_domain(self):

    response = timeDomainEmbed()
    if response:
        return JsonResponse(response, safe=False)

def timeDomainEmbed():
    watermarkImage = mpimg.imread(IMAGE_PATH)
    flattenWatermarkImage = np.array(watermarkImage).flatten(order='C')
    byteWatermarkImage = integerToBinary(flattenWatermarkImage)
    sampleRate, data = wavfile.read(AUDIO_PATH)
    audioBytes = data[1:524288]
    originalAudioBytes = audioBytes.copy()

    zeros = [253, 251, 239, 247, 239]
    ones = [2, 4, 16, 8, 16]
    counterSS = 1
    byteCount = 2
    bitCount = 1

    x = range(0, 524288, 4)

    for i in x:
        R = np.remainder(counterSS, 5); 
        if(R == 0):
            R = 5
        if(byteWatermarkImage[byteCount - 1, bitCount - 1] == 1):
            audioBytes[i] = audioBytes[i] | ones[R - 1]
        if(byteWatermarkImage[byteCount - 1, bitCount - 1] == 0):
            audioBytes[i] = audioBytes[i] & zeros[R - 1]
        counterSS = counterSS + 1
        bitCount = bitCount + 1
        if(bitCount == 9):
            bitCount = 1
            byteCount = byteCount + 1

    response = {}
    
    values = timeDomainDetect(audioBytes, originalAudioBytes, watermarkImage, ones, "no_attack")
    response["NO_ATTACK_SNR"] = values[1]
    response["NO_ATTACK_RO"] = values[2]

    values = timeDomainDetect(LowPass(audioBytes, 0.5), originalAudioBytes, watermarkImage, ones, "low_pass")
    response["LOW_PASS_SNR"] = values[1]
    response["LOW_PASS_RO"] = values[2]

    values = timeDomainDetect(Shearing(audioBytes), originalAudioBytes, watermarkImage, ones, "shearing")
    response["SHEARING_SNR"] = values[1]
    response["SHEARING_RO"] = values[2]

    values = timeDomainDetect(AWGN(audioBytes, -9), originalAudioBytes, watermarkImage, ones, "awgn")
    response["AWGN_SNR"] = values[1]
    response["AWGN_RO"] = values[2]

    return response
    
def timeDomainDetect(audioBytes, originalAudioBytes, watermarkImage, ones, attack):
    counterSS = 1
    byteCount = 1
    bitCount = 1
    s = (16384, 8)
    receivedBool = np.zeros(s)

    x = range(1, 524288, 4)

    for i in x:
        R = np.remainder(counterSS, 5); 
        if(R == 0):
            R = 5
        if((audioBytes[i - 1] & ones[R - 1]) == ones[R - 1]):
            receivedBool[byteCount - 1, bitCount - 1] = 1
        else:
            receivedBool[byteCount - 1, bitCount - 1] = 0
        counterSS = counterSS + 1
        bitCount = bitCount + 1
        if(bitCount == 9):
            bitCount = 1
            byteCount = byteCount + 1

    size = (16384,1)
    pixelsArray = np.zeros(size)
    x = range(1, 16384, 1)
    for i in x:
        pixelsArray[i - 1] = (receivedBool[i - 1, 0] * 128) + (receivedBool[i - 1, 1] * 64) + (receivedBool[i - 1, 2] * 32) + (receivedBool[i - 1, 3] * 16) + (receivedBool[i - 1, 4] * 8) + (receivedBool[i - 1, 5] * 4) + (receivedBool[i - 1, 6] * 2) + (receivedBool[i - 1, 7])

    pixelsArray.resize(128, 128)

    img = Image.fromarray(pixelsArray).convert('LA')
    img.save(os.path.join(BASE, '../public/static/time_domain_' + attack + '.png'))
    return ["time_domain_" + attack + ".png", computeSNR(audioBytes, originalAudioBytes), compareImages(watermarkImage, pixelsArray)]

def LowPass(audioBytes, value):
    b, a = signal.butter(2, value)
    audioBytes = signal.lfilter(b, a, audioBytes)
    audioBytes = audioBytes.astype(int)
    return audioBytes

def Shearing(audioBytes):
    x = range(10000, 18000, 1)
    for i in x:
        audioBytes[i] = 0
    return audioBytes

def AWGN(s, SNRdB, L=1):
    gamma = 10 ** (SNRdB/10) #SNR to linear scale
    if s.ndim == 1: # if s is single dimensional vector
        P = L * sum(abs(s) ** 2) / len(s) # Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P = L * sum(sum(abs(s) ** 2)) / len(s) # if s is a matrix [MxN]
    N0 = P / gamma # Find the noise spectral density
    if isrealobj(s): # check if input is real/complex object type
        n = sqrt(N0 / 2) * standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0 / 2) * (standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    r = r.astype(int)
    return r

def computeSNR(audioBytes, originalAudioBytes):
    firstPart = 0
    secondPart = 0

    for i in range(len(audioBytes)):
        firstPart = (float(originalAudioBytes[i]) * float(originalAudioBytes[i])) + firstPart
        secondPart = ((float(audioBytes[i]) - float(originalAudioBytes[i])) * (float(audioBytes[i]) - float(originalAudioBytes[i]))) + secondPart

    snr = 10 * (math.log(firstPart / secondPart))
    return snr

def integerToBinary(flattenWatermarkImage):
    byteWatermarkImage = [[0, 0, 0, 0, 0, 0, 0, 0]]
    for pixel in flattenWatermarkImage:
        #this will print a in binary
        bnr = bin(pixel).replace('0b','')
        x = bnr[::-1] #this reverses an array
        while len(x) < 8:
            x += '0'
        bnr = x[::-1]
        bnr = [int(x) for x in str(bnr)] 
        byteWatermarkImage = np.concatenate((byteWatermarkImage, [bnr]), axis=0)
    return byteWatermarkImage

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compareImages(imageA, imageB):
    return mse(imageA, imageB)

def wavelet(self):

    response = waveletEmbed()
    if response:
        return JsonResponse(response, safe=False)

def waveletEmbed():
    watermarkImage = mpimg.imread(IMAGE_PATH)
    watermarkImageOriginal = watermarkImage.copy()
    flattenWatermarkImage = watermarkImage.reshape(1, 16384)
    size = (16384, 1)
    newPixels = np.zeros(size)
    loopRange = range(1, 16384, 1)
    for i in loopRange:
        newPixels[i - 1] = float(flattenWatermarkImage[0][i - 1])/25.5
    sampleRate, data = wavfile.read(AUDIO_PATH)
    audioBytes = data[0:65535]

    L1 = np.zeros((32768, 1))
    L2 = np.zeros((16384, 1))
    H1 = np.zeros((32768, 1))
    H2 = np.zeros((16384, 1))
    normFactor = 2

    loopRange = range(1, 32768, 1)
    for i in loopRange:
        H1[i - 1] = ((float(audioBytes[2 * (i - 1)]) - float(audioBytes[(2 * (i - 1)) + 1])) / normFactor)
        L1[i - 1] = ((float(audioBytes[2 * (i - 1)]) + float(audioBytes[(2 * (i - 1)) + 1])) / normFactor)

    loopRange = range(1, 16384, 1)
    for i in loopRange:
        H2[i - 1] = ((L1[2 * (i - 1)] - L1[(2 * (i - 1)) + 1]) / normFactor)
        L2[i - 1] = ((L1[2 * (i - 1)] + L1[(2 * (i - 1)) + 1]) / normFactor)

    originalWavelet = np.zeros((65536, 1))
    watermarkedWavelet = np.zeros((65536, 1))

    loopRange = range(1, 16384, 1)

    for i in loopRange:
        originalWavelet[i - 1] = L2[i - 1]

    loopRange = range(1, 16384, 1)

    for i in loopRange:
        originalWavelet[(i - 1) + 16383] = H2[i - 1]

    loopRange = range(1, 32768, 1)

    for i in loopRange:
        originalWavelet[(i - 1) + 32767] = H1[i - 1]

    watermarkedWavelet = originalWavelet.copy()

    count = 1
    loopRange = range(1, 65536, 4)

    for i in loopRange:
        watermarkedWavelet[i - 1] = originalWavelet[i - 1] + newPixels[count - 1]
        count = count + 1

    response = {}
    
    values = waveletDetect(watermarkedWavelet, originalWavelet, watermarkImage, "no_attack")
    response["NO_ATTACK_SNR"] = values[1]
    response["NO_ATTACK_RO"] = values[2]

    values = waveletDetect(LowPass(watermarkedWavelet, 0.99), originalWavelet, watermarkImage, "low_pass")
    response["LOW_PASS_SNR"] = values[1]
    response["LOW_PASS_RO"] = values[2]

    values = waveletDetect(Shearing(watermarkedWavelet), originalWavelet, watermarkImage, "shearing")
    response["SHEARING_SNR"] = values[1]
    response["SHEARING_RO"] = values[2]

    values = waveletDetect(AWGN(watermarkedWavelet, -9), originalWavelet, watermarkImage, "awgn")
    response["AWGN_SNR"] = values[1]
    response["AWGN_RO"] = values[2]

    return response

def waveletDetect(watermarkedWavelet, originalWavelet, watermarkImage, attack):

    count2 = 1
    extractedPixels = np.zeros((16384, 1))

    loopRange = range(1, 65536, 4)
    for i in loopRange:
        extractedPixels[count2 - 1] = watermarkedWavelet[i - 1] - originalWavelet[i - 1]
        count2 = count2 + 1

    loopRange = range(1, 16384, 1)
    for  i in loopRange:
        extractedPixels[i - 1] = extractedPixels[i - 1] * 25.5

    size = (128, 128)
    pixelsMatrix = extractedPixels.reshape(size)

    img = Image.fromarray(pixelsMatrix).convert('LA')
    img.save(os.path.join(BASE, '../public/static/wavelet_' + attack + '.png'))
    return ["wavelet_" + attack + ".png", computeSNR(watermarkedWavelet, originalWavelet), compareImages(watermarkImage, pixelsMatrix)]