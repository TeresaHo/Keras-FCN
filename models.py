import numpy as np
import matplotlib.pyplot as plt
from pylab import *

import sys
#from keras_contrib.applications import densenet
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.applications.resnet50 import ResNet50
from keras.models import *
from keras.initializers import Constant
from keras.optimizers import SGD, Adam
#from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import tensorflow as tf

from utils.get_weights_path import *
from utils.basics import *
from utils.resnet_helpers import *
from utils.BilinearUpSampling import *

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
"""
def top(x, input_shape, classes, activation, weight_decay):

    x = Conv2D(classes, (1, 1), activation='linear',
               padding='same', kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)

    if K.image_data_format() == 'channels_first':
        channel, row, col = input_shape
    else:
        row, col, channel = input_shape

    # TODO(ahundt) this is modified for the sigmoid case! also use loss_shape
    if activation is 'sigmoid':
        x = Reshape((row * col * classes,))(x)

    return x


def FCN_Vgg16_32s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(size=(32, 32))(x)

    model = Model(img_input, x)

    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model


def AtrousFCN_Vgg16_16s(input_shape=None, weight_decay=0., batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', dilation_rate=(2, 2),
                      name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', strides=(1, 1), kernel_regularizer=l2(weight_decay))(x)

    x = BilinearUpSampling2D(target_size=tuple(image_size))(x)

    model = Model(img_input, x)

    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    return model
"""



def FCN_8s_unsupervised(input_shape=None, weight_decay=None, batch_momentum=0.9, batch_shape=None, classes=1):
    
    img_input = Input(shape=input_shape)
    image_size = input_shape[0:2]
    
    h = image_size[0]
    w = image_size[1]
    
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',kernel_regularizer=l2(weight_decay))(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',kernel_regularizer=l2(weight_decay))(pool3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',kernel_regularizer=l2(weight_decay))(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',kernel_regularizer=l2(weight_decay))(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
    
    x = Conv2D(4096, (7, 7), activation='relu',padding='same', name='fc1', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu',padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), name='classifyLayer', activation='sigmoid', padding='valid', strides=(1, 1),kernel_regularizer=l2(weight_decay))(x)
    
    lastup = BilinearUpSampling2D(target_size=tuple((h/16,w/16)))(x)
    conv4 = Conv2D(classes,kernel_size=(1,1),padding = "same",activation=None,kernel_initializer='zero',name = "conv4",kernel_regularizer=l2(weight_decay))(pool4)
    summed = Add()([conv4,lastup])
    up4 = BilinearUpSampling2D(target_size=tuple((h/8,w/8)))(summed)
    conv3 = Conv2D(classes,kernel_size=(1,1),padding = "same",activation=None,kernel_initializer='zero',name = "conv3",kernel_regularizer=l2(weight_decay))(pool3)
    summed = Add()([conv3,up4])
    temmap = BilinearUpSampling2D(target_size=tuple(image_size))(summed)
    #add sigmoid
    co_map = Activation('sigmoid',name='map')(temmap)

    co_map = Lambda(stackmap, name = 'stackmap')(co_map)
    ################
    #add classifier#
    ################
    #addmean
    img = Lambda(AddMean, name = 'addmean')(img_input)
    #img map multiply
    
    
    #img_b = Lambda(BHighLight,  name='highlightlayer2')([img, co_map])
    
    img_o = Multiply()([img,co_map])
    bg_map = Lambda(minusmap, name = 'minusmap')(co_map)
    img_b = Multiply()([img,bg_map])
    
    #substract
    img_o = Lambda(SubMean, name = 'submean')(img_o)
    img_b = Lambda(SubMean, name = 'submean1')(img_b)

    #feature extractor
    extractor = ResNet50(weights = 'imagenet', include_top = False, pooling = 'avg',input_shape=(h,w,3))
    extractor.trainable = False
    
    o_feature = extractor(img_o)
    b_feature = extractor(img_b)

    

    logits = Lambda(co_attention_loss,name='loss')([o_feature,b_feature])
    
    model = Model(inputs=img_input, outputs= logits ,name='Generator')
    
    
    #model.add_loss(loss)
    #optimizer = Adam(lr=0.00001)
    #model.compile(optimizer=optimizer,loss=[lambda y_true,y_pred: y_pred ])
    #model.compile(optimizer=optimizer,loss=None)

    ###### load weights
    #weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                        WEIGHTS_PATH_NO_TOP,cache_subdir='models')
    #model.load_weights(weights_path, by_name=True)
    #######
    # load transfer weights
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    model.summary()
    
    return model


def co_attention_loss(args):
    o_feature = args[0]
    b_feature = args[1]
    ext_feature = K.tile(o_feature,[1, 5])
    ext_feature = K.reshape(ext_feature,(5,5,2048))
    tem_feature = K.concatenate([o_feature,o_feature,o_feature,o_feature,o_feature],0)
    tem_feature = K.reshape(tem_feature,(5,5,2048))

    pos_dis = K.sum(K.square(tem_feature - ext_feature),axis=-1)
    pos_dis = K.sum(pos_dis,axis=-1)/(2048*4)
    neg_dis = K.sum(K.sum(K.square(o_feature-b_feature),axis=-1))
    neg_dis = K.stack([neg_dis,neg_dis,neg_dis,neg_dis,neg_dis],0)
    neg_dis = neg_dis/(2048*8)

    pos_dis = K.reshape(pos_dis,(5,1))
    neg_dis = K.reshape(neg_dis,(5,1))
    logits = K.concatenate([pos_dis,neg_dis],1)
    
    return logits

def stackmap(args):
    temmap = args
    co_map = K.concatenate([temmap,temmap,temmap],3)
    return co_map
def minusmap(args):
    temmap = args
    co_map = 1-temmap
    return co_map
def HighLight(args):
    img_input = args[0]
    co_map = args[1]
    img_o = img_input * co_map
                   
    return img_o
def BHighLight(args):
    img_input = args[0]
    co_map = args[1]
    co_map = 1-co_map
    img_o = img_input * co_map          
    return img_o

def AddMean(args):
    img_input = args
    r = img_input[:,:, :, 0] + 103.939
    g = img_input[:,:, :, 1] + 116.779
    b = img_input[:,:, :, 2] + 123.68
    img = K.stack([r,g,b] , axis=-1)
    return img

def SubMean(args):
    img_input = args
    r = img_input[:,:, :, 0] - 103.939
    g = img_input[:,:, :, 1] - 116.779
    b = img_input[:,:, :, 2] - 123.68
    img = K.stack([r,g,b] , axis=-1)
    return img

def FCN_Vgg16_8s(input_shape=None, weight_decay=None, batch_momentum=0.9, batch_shape=None, classes=21):
    if batch_shape:
        img_input = Input(batch_shape=batch_shape)
        image_size = batch_shape[1:3]
    else:
        img_input = Input(shape=input_shape)
        image_size = input_shape[0:2]
    
    h = image_size[0]
    w = image_size[1]
    
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',kernel_regularizer=l2(weight_decay))(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',kernel_regularizer=l2(weight_decay))(pool3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',kernel_regularizer=l2(weight_decay))(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',kernel_regularizer=l2(weight_decay))(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    
    
    x = Conv2D(4096, (7, 7), activation='relu',padding='same', name='fc1',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu',padding='same', name='fc2',kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    #classifying layer
    x = Conv2D(classes, (1, 1), name='classifylayer',activation=None, kernel_initializer='zero', padding='valid', strides=(1, 1),kernel_regularizer=l2(weight_decay))(x)
    
    #######Upsampling using Deconvolution2D (weights for upsampling will be learned)
    ##upsamplimg last layer
    # x = Deconvolution2D(classes,kernel_size=(4,4),kernel_initializer='zero',use_bias=False,strides = (2,2),padding = "valid",activation=None)(x)
    # x = Cropping2D(cropping=((0,2),(0,2)))(x)
    ##Conv to be applied on Pool4
    # conv4 = Conv2D(classes,kernel_size=(1,1),kernel_initializer='zero',padding = "same",activation=None, name = "conv4")(pool4)
    ##Add pool4 and last layer
    # Summed1 = Add()([conv4,x])
    ##upsample summed1
    # x1 = Deconvolution2D(classes,kernel_size=(4,4),kernel_initializer='zero',use_bias=False,strides = (2,2),padding = "valid",activation=None,name = "score4")(Summed1)
    # x1 = Cropping2D(cropping=((0,2),(0,2)))(x1)
    ##Conv to be applied on Pool3
    # conv3 = Conv2D(classes,kernel_size=(1,1),kernel_initializer='zero',padding = "same",activation=None, name = "conv3")(pool3)
    ##Add pool3 and summed1
    # Summed2 = Add()([conv3,x1])
    # Up = Deconvolution2D(classes,kernel_size=(16,16),strides = (8,8),kernel_initializer='zero',
                        #padding = "valid",activation = None,name = "upsample")(Summed1)
    # temmap = Cropping2D(cropping = ((0,8),(0,8)))(Up)
    ###############
    temmap = BilinearUpSampling2D(target_size=tuple(image_size))(Summed2)
    #co_map = Activation('sigmoid',name='map')(temmap)
    ###
    lastup = BilinearUpSampling2D(target_size=tuple((h/16,w/16)))(x)
    conv4 = Conv2D(classes,kernel_size=(1,1),padding = "same",activation=None,kernel_initializer='zero', name = "conv4",kernel_regularizer=l2(weight_decay))(pool4)
    summed = Add()([conv4,lastup])
    up4 = BilinearUpSampling2D(target_size=tuple((h/8,w/8)))(summed)
    conv3 = Conv2D(classes,kernel_size=(1,1),padding = "same",activation=None,kernel_initializer='zero', name = "conv3",kernel_regularizer=l2(weight_decay))(pool3)
    summed = Add()([conv3,up4])
    temmap = BilinearUpSampling2D(target_size=tuple(image_size))(summed)
    
    model = Model(inputs=img_input, outputs= temmap ,name='generator')
    #load weights
    ####no transfer learning on fc6 fc7 layers
    #weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                      #WEIGHTS_PATH_NO_TOP,cache_subdir='models')
    ######
    weights_path = os.path.expanduser(os.path.join('~', '.keras/models/fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5'))
    model.load_weights(weights_path, by_name=True)
    print("load transfer weights")
    model.summary()
    
    return model