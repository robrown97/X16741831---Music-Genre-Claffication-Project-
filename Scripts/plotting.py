import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import os
from matplotlib import cm
from python_speech_features import mfcc
import scipy.io.wavfile as wav

dir = ('C:/Users/Robert/Desktop/College/Project/Data/')
os.chdir(dir)

def plot_waveform():
    for fname in os.listdir(dir):

        save = '../Data/temp'+fname+'.waveform.png'
        f = dir+str(fname)

        spf = wave.open(f, "r")
        signal = spf.readframes(-1)
        signal = np.frombuffer(signal, "Int16")

        plt.figure(1)
        plt.title("Waveform : " + fname)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.plot(signal)

        plt.show()
        plt.savefig(save, dpi=100)

def plot_mfcc():
    for fname in os.listdir(dir):
        save = '../Data/temp'+fname+'.mfcc.png'

        rate,sig = wav.read(fname)
        mfcc_feat = mfcc(sig,rate)

        plt.figure(1)
        plt.title("MFCC : " + fname)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.plot(mfcc_feat)
        
        plt.show()
        plt.savefig(save, dpi=100)

plot_waveform()






