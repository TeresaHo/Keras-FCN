from __future__ import print_function
import sys
sys.path.append('..')
import h5py
import os
import numpy as np
#from fuel.datasets.hdf5 import H5PYDataset
#import cv2
import argparse
import glob
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
from PIL import Image

"""
parser = argparse.ArgumentParser('create a hdf5 file from lmdb or image directory')
parser.add_argument('--dataset_dir', dest='dataset_dir', help='the file or directory that stores the image collection', type=str ,default=None)
parser.add_argument('--width', dest='width', help='image size: width x width', type=int, default=32)
parser.add_argument('--mode', dest='mode', help='how the image collection is stored (mnist, lmdb, dir)', type=str, default='dir')
parser.add_argument('--channel', dest='channel', help='the number of image channels', type=int, default=3)
parser.add_argument('--hdf5_file', dest='hdf5_file', help='output hdf5 file', type=str ,default='faces.hdf5')
args = parser.parse_args()

if not args.dataset_dir:  #if the model_file is not specified
        args.dataset_dir = "Airplane"
width = args.width
"""
def ProcessImage(img, channel=3):  # [assumption]:  image is x, w, 3 with uint8
    if channel == 1:
        img = 255- cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1, 384, 384, 1)
    else:
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(1, 384, 384, 3)
    return img

def loaddataset(filepath):
    imgList = os.listdir(filepath)
    imgList.sort()
    nImgs = len(imgList)
    print(imgList)
    print("# of images :"+str(nImgs))

    imgs = []
    for id, file in enumerate(imgList):
        if id % 10 == 0:
            print('read %d/%d image' % (id, nImgs))
        img = image.load_img(os.path.join(filepath, file), target_size=(224, 224))
        x = image.img_to_array(img)
        #x = np.expand_dims(x, axis=0)
        #print(x.shape)
        x = preprocess_input(x)
        imgs.append(x)
    nImgs = len(imgs)
    imgs = np.stack(imgs, axis=0)
    #imgs = preprocess_input(imgs)
    print(imgs.shape)
    return imgs


if __name__ == '__main__':
    imgs = loaddataset("Car89")
    print(isinstance(imgs, np.ndarray))
    print(imgs.shape)
    #x = np.array(imgs)
    #print(imgs.dtype)
    #print(imgs.shape)
    """
    img_path = 'Airplane100/'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)
    """