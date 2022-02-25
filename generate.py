import numpy as np
from PIL import Image 
import os

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

def high_path(path, width):
    kernal = generateKernel(0.5, width)
    img = lowpass(path, kernal)
    
    imgreal = Image.open(path)
    imgreal = np.array(imgreal)
    img = imgreal - img
    img = Image.fromarray(np.uint8(img))
    
    return img

def low_path(path, width):
    kernal = generateKernel(1, width)
    img = lowpass(path, kernal)
    
    img = Image.fromarray(np.uint8(img))
    
    return img
    
if __name__ == "__main__":
    test_set = r"/content/content/CIFAR-10/test_set"
    os.mkdir("/content/CIFAR_10_Filtered_Set")
    
    for i in [3, 5, 7]:
        highpass_path = r"/content/CIFAR_10_Filtered_Set/highpass_1_"+str(i)
        lowpass_path = r"/content/CIFAR_10_Filtered_Set/lowpass_1_"+str(i)
        os.mkdir(highpass_path)
        os.mkdir(lowpass_path)
        
        for fol_id, folder in enumerate(os.listdir(test_set), 1):
            os.mkdir(highpass_path+'/'+folder)
            os.mkdir(lowpass_path+'/'+folder)
            for img in os.listdir(test_set+'/'+folder):
                new_img = high_path(test_set+'/'+folder+'/'+img, i)
                new_img.save(highpass_path+'/'+folder+'/'+img)

                new_img = low_path(test_set+'/'+folder+'/'+img, i)
                new_img.save(lowpass_path+'/'+folder+'/'+img)
            print("Folder No: {}, Folder Name: {} --- Complete".format(fol_id, folder))
        print('{} complete.'.format(i))