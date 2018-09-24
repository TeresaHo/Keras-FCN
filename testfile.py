"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,3"
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

from models import *
from utils.loss_function import *
from utils.metrics import *
from utils.SegDataGenerator import *
import time
import tensorflow as tf
import keras.backend as K


def loaddataset(train_file_path,data_dir,label_dir):
    data_files = []
    label_files = []
    fp = open(train_file_path)
    lines = fp.readlines()
    fp.close()
    nb_samples = len(lines)
    print(nb_samples)
    for line in lines:
        line = line.strip('\n')
        data_files.append(line + '.jpg')
        label_files.append(line + '.png')
    print(len(label_files))
    
    imgs = []
    labels = []
    for data, label in zip(data_files,label_files):
        img = image.load_img(os.path.join(data_dir, data), target_size=(384, 384))
        label_filepath = os.path.join(label_dir, label)
        label = Image.open(label_filepath)
        label = label.resize((384,384), Image.ANTIALIAS)

        x = image.img_to_array(img)
        y = image.img_to_array(label).astype(int)
        #x = np.expand_dims(x, axis=0)
        #print(x.shape)
        x = preprocess_input(x)
        imgs.append(x)
        labels.append(y)
    nImgs = len(imgs)
    imgs = np.stack(imgs, axis=0)
    labels = np.stack(labels, axis=0)
    print(imgs.shape)
    print(labels.shape)
    print(labels[0,200])
    
    return imgs,labels

if __name__ == '__main__':
    
    train_file_path = os.path.expanduser('/4TB/ccho/VOC2012/ImageSets/Segmentation/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
    # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
    val_file_path   = os.path.expanduser('/4TB/ccho/VOC2012/ImageSets/Segmentation/val.txt')
    data_dir        = os.path.expanduser('/4TB/ccho/VOC2012/JPEGImages')
    label_dir       = os.path.expanduser('/4TB/ccho/VOC2012/SegmentationClass')
    data_suffix='.jpg'
    label_suffix='.png'
    classes = 21
    model_name = 'myFCN_Vgg16_8s'
    #X,Y = loaddataset(train_file_path,data_dir,label_dir)
    print(os.path.abspath(os.path.join(os.getcwd(), "../../..","4TB/ccho/Data")))
    path_to  = os.path.abspath(os.path.join(os.getcwd(), "../../..","4TB/ccho"))
    save_path = os.path.abspath(os.path.join(path_to, 'Models/' + model_name))
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    
    a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
    fu = np.array([3])
    print(a.shape)
    print(fu.shape)
    a = K.variable(value=a)
    b = tf.tile(a,multiples=[1, 5])
    b = tf.reshape(b,(5,5,3))
    fu = K.variable(value=fu)
    #print(b.shape)
    c = tf.concat([a,a,a,a,a],0)
    c = tf.reshape(c,(5,5,3))
    #print(c.shape)
    d = b-c
    e = K.sum(d,axis=-1)
    e = K.sum(e,axis=-1)
    e = tf.reshape(e,(5,1))
    f = tf.concat([e,e],1)
    fu = tf.concat([fu,fu,fu,fu,fu],0)
    
    
    
    logits = K.variable(value=val)
    for i in range(3):
        a = tf.ones((5,1))
        b = tf.concat([b,a],1)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    #print(sess.run(b))
    print(sess.run(d))
    print(d.shape)
    print(sess.run(e))
    print(e.shape)
    print(sess.run(fu))
    print(fu.shape)
"""
import torch
import torch.nn.functional as F     # 激励函数都在这
import torchvision

class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  
print(torch.cuda.is_available())
    