import numpy as np
from PIL import Image 
import os
import random
from shutil import copyfile
from multiprocessing import Pool

def generateKernel(sigma, width):
    kernel = np.zeros((width,width))
    temp = width//2
    for i in range(width):
        for j in range(width):
            x = -temp+i
            y = -temp+j
            kernel[i,j] = (1/(2*np.pi*(sigma**2)))*np.exp(-1*(x**2+y**2)/(2*sigma**2))
    return kernel

def lowpass(path, kernal):
    kernal_width = kernal.shape[0]
    pad = kernal_width//2
    img = Image.open(path)
    img = np.array(img)
    
    imgout = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype="uint8")
    
    img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    
    for h in range(imgout.shape[0]):
        for w in range(imgout.shape[1]):
            for c in range(imgout.shape[2]):
                imgSlice = img[h:h+kernal_width, w:w+kernal_width, c]
                conv = np.multiply(imgSlice, kernal)
                imgout[h][w][c] = np.sum(conv)

    return imgout

def lowpath(path, sigma, width):
    kernal = generateKernel(sigma, width)
    img = lowpass(path, kernal)
    
    # imgreal = Image.open(path)
    # imgreal = np.array(imgreal)
    # img = imgreal - img
    img = Image.fromarray(np.uint8(img))
    
    return img

def highpath(path, sigma, width):
    kernal = generateKernel(sigma, width)
    img = lowpass(path, kernal)
    
    imgreal = Image.open(path)
    imgreal = np.array(imgreal)
    img = imgreal - img
    img = Image.fromarray(np.uint8(img))
    
    return img
    
def trainset(folder):
    
    os.mkdir(highpass_path+'/'+folder)
    for i, img in enumerate(os.listdir(test_set+'/'+folder), 0):
        high_low = random.sample([True, False], 1)[0]
        sigma = random.uniform(0.25, 1.75)
        width = random.sample([2, 3, 4, 5], 1)[0]
        copyfile(test_set+'/'+folder+'/'+img, highpass_path+'/'+folder+'/'+'new'+img)
        if high_low:
            new_img = lowpath(test_set+'/'+folder+'/'+img, sigma, width)
            new_img.save(highpass_path+'/'+folder+'/'+img)
        else:
            new_img = highpath(test_set+'/'+folder+'/'+img, sigma, width)
            new_img.save(highpass_path+'/'+folder+'/'+img)
        if i == 2500:
            print("{} half done.".format(folder), end=' ')
    print("{} done.".format(folder))

test_set = r'/content/drive/My Drive/CIFAR-10 Test/Trainset'
highpass_path = r'/content/drive/My Drive/CIFAR-10 Test/Trainset3'
os.mkdir(highpass_path)

for folder in os.listdir(test_set):
  trainset(folder)