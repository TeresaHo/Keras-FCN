from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from sklearn.cluster import KMeans
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs




extractor = ResNet50(weights = 'imagenet', include_top = False, pooling = 'avg',input_shape=(224,224,3))

vgg16_feature_list = []
#sub = os.path.expanduser('~/Keras-FCN/sub_dataset')
current_dir = os.path.dirname(os.path.realpath(__file__))
sub = os.path.join(current_dir,'sub_dataset')
subdir = os.listdir(sub)

for idx, dirname in enumerate(subdir):
    # get the directory names, i.e., 'dogs' or 'cats'
    # ...
    print(dirname)
    filenames = os.path.join(sub,dirname)
    joinpath = filenames
    filenames = os.listdir(filenames)
    
    for i, fname in enumerate(filenames):
        # process the files under the directory 'dogs' or 'cats'
        # ...
        img_path = os.path.join(joinpath,fname)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        vgg16_feature = extractor.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        
vgg16_feature_list_np = np.array(vgg16_feature_list)
kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(vgg16_feature_list_np)
print(vgg16_feature_list_np.shape)
print(kmeans)
class0=0
class1=0
class2=0
for i in kmeans:
    if i==0:
        class0 = class0+1
    elif i==1:
        class1 = class1+1
    elif i==2:
        class2 = class2+1
print("class 0 : "+str(class0))
print("class 1 : "+str(class1))
print("class 2 : "+str(class2))

#vgg16_feature_list_np.shape = np.reshape(vgg16_feature_list_np.shape,(40,1,1,2048))
plt.figure(figsize=(6, 6))
plt.scatter(vgg16_feature_list_np[:,900], vgg16_feature_list_np[:,990], c=kmeans)
plt.title("feature visualization")
plt.show()
