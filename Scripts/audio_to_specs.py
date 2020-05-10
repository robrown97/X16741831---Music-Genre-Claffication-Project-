
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os 
import numpy as np

dir = ('C:/Users/Robert/Desktop/College/Project/Data/Raw/')
sr = 22050 #44.1kHz

for fname in os.listdir(dir):
    f = dir+str(fname)

    save_path = fname[:-4]+".png"
    specname = ('C:/Users/Robert/Desktop/College/Project/Data/Specs/' + str(save_path))
    plt.figure(figsize=(2.56,1.28), dpi=1000) 
    plt.axis('off') # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

    y, sr = librosa.load(f, sr=sr) #specify sample rate
    print(sr)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max),  cmap='jet')
  
    plt.xlim(left=0,right=30)
    plt.savefig(fname = specname, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()

