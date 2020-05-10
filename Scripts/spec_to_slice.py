from PIL import Image
import os.path
import re

specPath = "C:/Users/Robert/Desktop/College/Project/Data/Specs/"
slicesPath = "C:/Users/Robert/Desktop/College/Project/Data/Slice/"
desiredSize = 15

#Slice spectrograms 
def slices_from_spec(desiredSize):
    for filename in os.listdir(specPath):
        if filename.endswith(".png"):
            slice_spec(filename,desiredSize)

#Creates slices from spectrogram
def slice_spec(filename, desiredSize):
    file = re.sub("[0-9]","", filename) #remove numbers
    genre = file.split(".")[0] #blues.png --> blues

    img = Image.open(specPath+filename)

    width, height = img.size
    nbSamples = 10
    width - desiredSize

    #Create path if not existing
    slicePath = slicesPath+"{}/".format(genre)
    if not os.path.exists(os.path.dirname(slicePath)):
        try:
            os.makedirs(os.path.dirname(slicePath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    #For each sample
    for i in range(nbSamples):
        startPixel = i*desiredSize
        imgTmp = img.crop((startPixel, 0, startPixel + desiredSize, desiredSize + 113))
        imgTmp.save(slicesPath+"{}/{}_{}.png".format(genre,filename[:-4],i))

slices_from_spec(desiredSize)   