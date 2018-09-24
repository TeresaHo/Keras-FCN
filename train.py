import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
#from utils.SegDataGenerator import *
from DataGenerator import*
import time
from parallel_model import ParallelModel
import dataProcess



def train(batch_size, epochs, lr_base, lr_power, weight_decay, classes,
          model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=None, batchnorm_momentum=0.9,
          resume_training=False, class_weight=None, dataset='VOC2012',
          loss_fn=softmax_sparse_crossentropy_ignoring_last_label,
          metrics=[sparse_accuracy_ignoring_last_label],
          loss_shape=None,
          label_suffix='.png',
          data_suffix='.jpg',
          ignore_label=255,
          label_cval=255):
    if target_size:
        input_shape = target_size + (3,)
    else:
        input_shape = (None, None, 3)
    batch_shape = (batch_size,) + input_shape

    ###########################################################
    #current_dir = os.path.dirname(os.path.realpath(__file__))
    path_to  = os.path.abspath(os.path.join(os.getcwd(), "../../..","4TB/ccho"))
    save_path = os.path.abspath(os.path.join(path_to, 'Models/' + model_name))
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)

    # ###############learning rate scheduler####################
    def lr_scheduler(epoch, mode='power_decay'):
        '''if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr'''

        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.9 * epochs:
                lr = 0.0001
            elif epoch > 0.75 * epochs:
                lr = 0.001
            elif epoch > 0.5 * epochs:
                lr = 0.01
            else:
                lr = 0.1

        
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)
    

    # ###################### make model ########################
    checkpoint_path = os.path.abspath(os.path.join(save_path, 'checkpoint_weights.hdf5'))
    
    model = globals()[model_name](weight_decay=weight_decay,
                                  input_shape=input_shape,
                                  batch_momentum=batchnorm_momentum,
                                  classes=classes)
    #model = multi_gpu_model(model, gpus=3)

    # ###################### optimizer ########################
    #optimizer = SGD(lr=lr_base, momentum=0.9)
    #optimizer = Nadam(lr=lr_base, beta_1 = 0.825, beta_2 = 0.99685)
    optimizer = Adam(lr=0.00001)
    
    #model.compile(optimizer=optimizer,loss=[lambda y_true,y_pred: y_pred ])
    
    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=metrics)
    
    if resume_training:
        model.load_weights(checkpoint_path, by_name=True)
    #model_path = os.path.join(save_path, "model.json")
    # save model structure
    """
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.close
    img_path = os.path.join(save_path, "model.png")
    # #vis_util.plot(model, to_file=img_path, show_shapes=True)
    model.summary()
    """

    callbacks = []

    # ####################### tfboard ###########################
    """
    if K.backend() == 'tensorflow':
        tensorboard = TensorBoard(log_dir=os.path.abspath(os.path.join(save_path, 'logs')), histogram_freq=10, write_graph=True)
        callbacks.append(tensorboard)
    """
    # ################### checkpoint saver#######################
    checkpoint = ModelCheckpoint(filepath=os.path.abspath(os.path.join(save_path, 'checkpoint_weights.hdf5')), save_weights_only=True)#.{epoch:d}
    callbacks.append(checkpoint)
    # set data generator and train
    train_datagen = SegDataGenerator(zoom_range=[0.5, 2.0],
                                     zoom_maintain_shape=True,
                                     crop_mode='none',
                                     crop_size=target_size,
                                     # pad_size=(505, 505),
                                     rotation_range=0.,
                                     shear_range=0,
                                     horizontal_flip=True,
                                     channel_shift_range=20.,
                                     fill_mode='constant',
                                     rescale=None,
                                     label_cval=label_cval)
    #val_datagen = SegDataGenerator()

    def get_file_len(file_path):
        imgList = os.listdir(file_path)
        nImgs = len(imgList)
        return nImgs
        #fp = open(file_path)
        #lines = fp.readlines()
        #fp.close()
        #return len(lines)
    #X,Y = loaddataset(train_file_path,data_dir,label_dir)
    # from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished
    # and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
    steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(batch_size)))
    #model.fit_generator(generator(x=X,y=Y, batch_size=batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs,callbacks=callbacks,class_weight=class_weight)
    
    history = model.fit_generator(
        generator=train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes,
            target_size=target_size, color_mode='rgb',
            batch_size=batch_size, shuffle=True,
            loss_shape=loss_shape,
            ignore_label=ignore_label,
            # save_to_dir='Images/'
        ),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        # workers=4,
        # validation_data=val_datagen.flow_from_directory(
        #     file_path=val_file_path, data_dir=data_dir, data_suffix='.jpg',
        #     label_dir=label_dir, label_suffix='.png',classes=classes,
        #     target_size=target_size, color_mode='rgb',
        #     batch_size=batch_size, shuffle=False
        # ),
        # nb_val_samples = 64
        class_weight=class_weight
       )
    
    eva_model = Model(inputs = model.input, outputs = model.get_layer('map').output)
    X_train = dataProcess.loaddataset(train_file_path)
    co_map = eva_model.predict(X_train)
    for i in range(len(co_map)):
        re = co_map[i,:,:,0]
        if i==0:
            print(re)
        rescaled = (255.0 / re.max() * (re - re.min())).astype(np.uint8)
        #rescaled = (re*255).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save('results/result%s.png'%i)
    

    #model.save_weights(save_path+'/model.hdf5')
    
if __name__ == '__main__':
    model_name = 'FCN_Vgg16_8s'
    #model_name = 'FCN_8s_unsupervised'
    
    batch_size = 5
    batchnorm_momentum = 0.95
    epochs = 1
    lr_base = 0.01 * (float(batch_size) / 16)
    lr_power = 0.9
    resume_training = False
    if model_name is 'FCN_Vgg16_8s':
        weight_decay = 0.0001/2
    else:
        weight_decay = 1e-4
    target_size = (224, 224)
    dataset = 'Car'
    if dataset == 'VOC2012_BERKELEY':
        # pascal voc + berkeley semantic contours annotations
        train_file_path = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        # train_file_path = os.path.expanduser('~/.keras/datasets/oneimage/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        val_file_path   = os.path.expanduser('~/.keras/datasets/VOC2012/combined_imageset_val.txt')
        data_dir        = os.path.expanduser('~/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages')
        label_dir       = os.path.expanduser('~/.keras/datasets/VOC2012/combined_annotations')
        data_suffix='.jpg'
        label_suffix='.png'
        classes = 21
    
    else:
        #train_file_path = os.path.expanduser('/4TB/ccho/VOC/benchmark_RELEASE/dataset/train.txt') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        train_file_path = os.path.expanduser('~/FCN/Car100') #Data/VOClarge/VOC2012/ImageSets/Segmentation
        #val_file_path   = os.path.expanduser('/4TB/ccho/VOC2012/ImageSets/Segmentation/val.txt')
        #data_dir        = os.path.expanduser('/4TB/ccho/VOC/benchmark_RELEASE/dataset/img')
        #label_dir       = os.path.expanduser('/4TB/ccho/VOC2012/cls_png')
        data_suffix='.jpg'
        label_suffix='.png'
        classes = 1


    # ###################### loss function & metric ########################
    
    #loss_fn = softmax_sparse_crossentropy_ignoring_last_label
    loss_fn = my_softmax_crossentropy
    metrics = [sparse_accuracy_ignoring_last_label]
    loss_shape = None
    ignore_label = 255
    label_cval = 255

    class_weight = None

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    train(batch_size, epochs, lr_base, lr_power, weight_decay, classes, model_name, train_file_path, val_file_path,
          data_dir, label_dir, target_size=target_size, batchnorm_momentum=batchnorm_momentum, resume_training=resume_training,
          class_weight=class_weight, loss_fn=loss_fn, metrics=metrics, loss_shape=loss_shape, data_suffix=data_suffix,
          label_suffix=label_suffix, ignore_label=ignore_label, label_cval=label_cval)
