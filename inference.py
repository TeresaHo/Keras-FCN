import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.models import load_model
import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input

from models import *


def inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=True, save_dir=None,
              label_suffix='.png',
              data_suffix='.jpg'):

    path_to  = os.path.abspath(os.path.join(os.getcwd(), "../../..","4TB/ccho"))
    save_path = os.path.abspath(os.path.join(path_to, 'Models/' + model_name))
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'save_dir')
    batch_shape = (1, ) + image_size + (3, )
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, weight_file)
    # model_path = os.path.join(current_dir, 'model_weights/fcn_atrous/model_change.hdf5')
    # model = FCN_Resnet50_32s((480,480,3))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name](batch_shape=batch_shape, input_shape=(384, 384, 3))
    model.load_weights(checkpoint_path, by_name=True)
    eva_model = Model(inputs = model.input, outputs = model.get_layer('map').output)
    eva_model.load_weights(checkpoint_path, by_name=True)
    model.summary()

    results = []
    total = 0
    for img_num in image_list:
        #img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total,img_num))
        image = Image.open('%s/%s' % (data_dir, img_num))
        image = img_to_array(image)  # , data_format='default')
        
        #label = Image.open('%s/%s%s' % (label_dir, img_num, label_suffix))
        #label_size = label.size

        img_h, img_w = image.shape[0:2]

        # long_side = max(img_h, img_w, image_size[0], image_size[1])
        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        #image = np.lib.pad(image, ((pad_h/2, pad_h - pad_h/2), (pad_w/2, pad_w - pad_w/2), (0, 0)), 'constant', constant_values=0.)
        image = np.lib.pad(image, ((math.ceil(pad_h/2), math.floor(pad_h - pad_h/2)), (math.ceil(pad_w/2), math.floor(pad_w - pad_w/2)), (0, 0)), 'constant', constant_values=0.)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        
        result = eva_model.predict(image, batch_size=1)
        print("predict shape"+str(result.shape))
        if total==1 or total==80:
            print(result)
            print(np.mean(result))
        
        result = np.squeeze(result)
        if total==1 or total==80:
            print(result)
            print(np.mean(result))
        result = np.squeeze(result)
        print(result.shape)
        rescaled = (result*255).astype(np.uint8)
        result_img = Image.fromarray(rescaled,mode='L')
        
        # result_img = result_img.resize(label_size, resample=Image.BILINEAR)
        result_img = result_img.crop((pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
        
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, str(total) + '.png'))
        
    return results

if __name__ == '__main__':
    # model_name = 'AtrousFCN_Resnet50_16s'
    # model_name = 'Atrous_DenseNet'
    model_name = 'FCN_Vgg16_8s'
    weight_file = 'checkpoint_weights.hdf5'
    image_size = (384, 384)
    val_file_path   = os.path.expanduser('/4TB/ccho/VOC2012/ImageSets/Segmentation/val.txt')
    #data_dir        = os.path.expanduser('/4TB/ccho/VOC2012/JPEGImages')
    label_dir       = os.path.expanduser('/4TB/ccho/VOC2012/SegmentationClass')
    data_dir = os.path.expanduser('~/FCN/Car100')
    
    image_list = os.listdir(data_dir)
    image_list.sort()
    
    #fp = open(val_file_path)
    #image_list = fp.readlines()
    #fp.close()
    
    results = inference(model_name, weight_file, image_size, image_list, data_dir, label_dir)
    
"""  
def inference(model_name, weight_file, image_size, image_list, data_dir, label_dir, return_results=True, save_dir=None,
              label_suffix='.png',
              data_suffix='.jpg'):
    path_to  = os.path.abspath(os.path.join(os.getcwd(), "../../..","4TB/ccho"))
    save_path = os.path.abspath(os.path.join(path_to, 'Models/' + model_name))
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(current_dir, 'save_dir')
    batch_shape = (1, ) + image_size + (3, )
    
    model_path = os.path.join(save_path, "model.json")
    checkpoint_path = os.path.join(save_path, weight_file)
    save_dir = os.path.join(current_dir, 'save_dir')
    # model_path = os.path.join(current_dir, 'model_weights/fcn_atrous/model_change.hdf5')
    # model = FCN_Resnet50_32s((480,480,3))

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    model = globals()[model_name](batch_shape=batch_shape, input_shape=(512, 512, 3))
    model.load_weights(checkpoint_path, by_name=True)

    model.summary()

    results = []
    total = 0
    for img_num in image_list:
        img_num = img_num.strip('\n')
        total += 1
        print('#%d: %s' % (total,img_num))
        image = Image.open('%s/%s%s' % (data_dir, img_num, data_suffix))
        image = img_to_array(image)  # , data_format='default')

        label = Image.open('%s/%s%s' % (label_dir, img_num, label_suffix))
        label_size = label.size

        img_h, img_w = image.shape[0:2]

        # long_side = max(img_h, img_w, image_size[0], image_size[1])
        pad_w = max(image_size[1] - img_w, 0)
        pad_h = max(image_size[0] - img_h, 0)
        print(image.shape)
        image = np.lib.pad(image, ((math.ceil(pad_h/2), math.floor(pad_h - pad_h/2)), (math.ceil(pad_w/2), math.floor(pad_w - pad_w/2)), (0, 0)), 'constant', constant_values=0.)
        # image -= mean_value
        '''img = array_to_img(image, 'channels_last', scale=False)
        img.show()
        exit()'''
        # image = cv2.resize(image, image_size)
        print(image.shape)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        result = model.predict(image, batch_size=1)
        result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

        result_img = Image.fromarray(result, mode='P')
        result_img.palette = label.palette
        # result_img = result_img.resize(label_size, resample=Image.BILINEAR)
        result_img = result_img.crop((pad_w/2, pad_h/2, pad_w/2+img_w, pad_h/2+img_h))
        # result_img.show(title='result')
        if return_results:
            results.append(result_img)
        if save_dir:
            result_img.save(os.path.join(save_dir, img_num + '.png'))
    return results

if __name__ == '__main__':
    # model_name = 'AtrousFCN_Resnet50_16s'
    # model_name = 'Atrous_DenseNet'
    # model_name = 'DenseNet_FCN'
    model_name = 'myFCN_Vgg16_8s'
    weight_file = 'checkpoint_weights.hdf5'
    image_size = (512, 512)
    
    val_file_path   = os.path.expanduser('/4TB/ccho/VOC/benchmark_RELEASE/dataset/val.txt')
    data_dir        = os.path.expanduser('/4TB/ccho/VOC/benchmark_RELEASE/dataset/img')
    label_dir       = os.path.expanduser('/4TB/ccho/VOC2012/cls_png')
    

    #image_list = sys.argv[1:]#'2007_000491'
    fp = open(val_file_path)
    image_list = fp.readlines()
    fp.close()
    results = inference(model_name, weight_file, image_size, image_list, data_dir, label_dir)
    #for result in results:
        #result.show(title='result', command=None)
"""