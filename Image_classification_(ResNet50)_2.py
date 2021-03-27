#!/usr/bin/env python
# coding: utf-8

# # Image categorisation

# *  JantaHack Computer Vision by analyticsvidhya
# *  Classify emergency and non-emergency vehicle from images

# In[3]:


import os 
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


# In[4]:


# import pandas as pd
# import numpy as np
# import random
# import math
# # from google.colab import drive
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Conv2D, MaxPool2D, BatchNormalization
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils.np_utils import to_categorical

# from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg16 import VGG16
# from keras.models import Model, Sequential
# from keras.layers import Input, Dense, GlobalAveragePooling2D
# from keras.optimizers import Adam
# from keras import backend as K

# import seaborn as sns
# from matplotlib import pyplot as plt


# In[2]:


# get_ipython().system('pwd')


# In[3]:


import pandas as pd
import numpy as np
import random
import math
# from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
# from tensorflow.keras.utils.np_utils import to_categorical

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import seaborn as sns
from matplotlib import pyplot as plt


# In[6]:


# numpy random number geneartor seed
# for reproducibility
np.random.seed(123)

# set plot rc parameters
# jtplot.style(grid=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#232323'
#plt.rcParams['axes.edgecolor'] = '#FFFFFF'
plt.rcParams['figure.figsize'] = 10, 7
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.framealpha'] = 0.2
plt.rcParams['text.color'] = '#666666'
plt.rcParams['axes.labelcolor'] = '#666666'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.color'] = '#666666'
plt.rcParams['ytick.labelsize'] = 14

# plt.rcParams['font.size'] = 16

sns.color_palette('dark')
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data

# In[7]:


# drive.mount('/content/drive')


# In[8]:


data_dir = 'data'
img_dir = 'data/images'


# In[9]:


train_img = pd.read_csv(data_dir+'/train.csv')
test_img = pd.read_csv(data_dir+'/test_vc2kHdQ.csv')
train_img.shape, test_img.shape


# In[10]:


train_img['emergency_or_not'] = train_img['emergency_or_not'].apply(lambda x: str(x))


# In[11]:


train_img.head()


# ## Load Image data

# ### Load image to array

# In[12]:


im1 = load_img(img_dir+'/23.jpg')
im1_array = img_to_array(im1)
im1_array.shape


# In[13]:


# x = np.zeros(shape = [1646, 224, 224, 3])


# In[14]:


# for i, img in enumerate(train_img['image_names'].values):
#     x[i] = img_to_array(load_img('data/images/'+img))


# ### Load image using generators

# In[15]:


datagen = ImageDataGenerator(rescale=1./255.,
                             validation_split=0.25)


# In[16]:


# train generator
train_generator=datagen.flow_from_dataframe(dataframe=train_img,
                                            directory=img_dir,
                                            x_col="image_names",
                                            y_col="emergency_or_not",
                                            subset="training",
                                            batch_size=8,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(224,224),
                                            color_mode='rgb')
# validation data generator
valid_generator=datagen.flow_from_dataframe(dataframe=train_img,
                                            directory=img_dir,
                                            x_col="image_names",
                                            y_col="emergency_or_not",
                                            subset="validation",
                                            batch_size=8,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(224,224),
                                            color_mode='rgb')


# In[17]:


# test data generator
test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_img,
                                                  directory=img_dir,
                                                  x_col="image_names",
                                                  y_col=None,
                                                  batch_size=8,
                                                  seed=42,
                                                  shuffle=False,
                                                  class_mode=None,
                                                  target_size=(224,224),
                                                  color_mode='rgb')


# ## CNN model

# In[18]:


# # initiate sequential model
# model = Sequential()
# # add convolutional layer
# # 16 sliding windows each of 3X3 size
# # default step is 1X1
# model.add(Conv2D(filters = 32,
#                  kernel_size = (5, 5),
#                  activation='relu',
#                  input_shape = (224, 224,3),
#                  padding='same'))
# # add batch normalization to normalize output of the layer
# model.add(BatchNormalization())
# # add another convolutional layer
# model.add(Conv2D(filters = 32,
#                  kernel_size = (5, 5),
#                  activation='relu',
#                  padding='same'))
# # batchnormalize
# model.add(BatchNormalization())
# # add maxpooling layer
# # this layer picks max value for every 2X2 window
# model.add(MaxPool2D(pool_size=(2,2)))
# # add dropout layer
# model.add(Dropout(0.3))
# # repeat above sequence once more
# model.add(Conv2D(filters = 64,
#                  kernel_size = (5, 5),
#                  activation='relu',
#                  padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters = 64,
#                  kernel_size = (5, 5),
#                  activation='relu',
#                  padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.3))
# # another pair of convolutional layers
# model.add(Conv2D(filters = 128,
#                  kernel_size = (5, 5),
#                  activation='relu',
#                  padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters = 128,
#                  kernel_size = (5, 5),
#                  activation='relu',
#                  padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters = 128,
#                  kernel_size = (5, 5),
#                  activation='relu',
#                  padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.3))
# # flatten cnn layers
# model.add(Flatten())
# # add dense layer
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.3))
# # finally add a softmax layer which will predict probability of each class
# model.add(Dense(1, activation='sigmoid'))
# # print model summary
# model.summary()

# # compile model
# model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])


# ## ResNet50

# In[1]:


# Setup transfer learning model

# # load model without classifier layers
# base_model = VGG16(include_top=False,
#                    input_tensor=Input(shape=(224, 224, 3)),
#                    weights='imagenet')
# # add new classifier layers
# flat1 = Flatten()(base_model.outputs)
# #flat1 = GlobalAveragePooling2D()(flat1)
# class1 = Dense(256, activation='relu')(flat1)
# output = Dense(1, activation='sigmoid')(class1)
# # define new model
# model = Model(inputs=base_model.inputs, outputs=output)
# # summarize
# model.summary()

# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional ResNet50 layers
# for layer in base_model.layers:
#     layer.trainable = False

model = Sequential()
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
model.add(ResNet50(include_top = False, input_shape = (224,224,3)))

for layer in model.layers:
    layer.trainable = False
    
# model.add(GlobalAveragePooling2D())    
model.add(Flatten())
# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

# Say not to train first layer (ResNet) model as it is already trained
# model.layers[0].trainable = False
model.summary()
# optimizer
opt = Adam(learning_rate=0.00001)
# compile model
model.compile(loss='binary_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])


# ## Train CNN model

# In[ ]:


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


# In[2]:


model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    epochs=10)


# In[47]:


model.evaluate_generator(generator=valid_generator,
                         steps=STEP_SIZE_TEST)


# ## Submissions

# In[48]:


# image genrator
datagen_final = ImageDataGenerator(rescale=1./255.)
# train generator
train_generator_final = datagen_final.flow_from_dataframe(dataframe=train_img,
                                            directory=img_dir,
                                            x_col="image_names",
                                            y_col="emergency_or_not",
                                            batch_size=8,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="binary",
                                            target_size=(224,224),
                                            color_mode='rgb')
# final model
final_model = Sequential()
# NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
final_model.add(ResNet50(include_top = False, input_shape = (224,224,3)))
# model.add(GlobalAveragePooling2D())
final_model.add(Flatten())
# 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
final_model.add(Dense(1024, activation = 'relu'))
final_model.add(Dropout(0.3))
final_model.add(Dense(512, activation = 'relu'))
final_model.add(Dropout(0.3))
final_model.add(Dense(1, activation = 'sigmoid'))

# Say not to train first layer (ResNet) model as it is already trained
# model.layers[0].trainable = False
final_model.summary()
# optimizer
opt = adam(learning_rate=0.00001)
# compile model
final_model.compile(loss='binary_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])


# In[49]:


final_model.fit_generator(generator=train_generator_final, epochs=5)


# In[50]:


test_generator.reset()
pred = final_model.predict_generator(test_generator,
                             verbose=1)


# In[51]:


pred.shape


# In[52]:


pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0
pred[:10]


# In[ ]:


pred = pred.ravel()


# In[ ]:


# labels = (train_generator.class_indices)
# labels = dict((v,k) for k,v in labels.items())
# predictions = [labels[k] for k in predicted_class_indices]


# In[ ]:


out_dir = '/content/drive/My Drive/JantaHack Computer Vision/'
filenames=test_generator.filenames
results=pd.DataFrame({"image_names":filenames,
                      "emergency_or_not":pred})
results.to_csv(out_dir+"Submissions_0.csv",index=False)


# In[ ]:




