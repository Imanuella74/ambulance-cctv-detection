# Ambulance CCTV Detection
*Created Using Python 3.7.10 and Tensorflow 2.4.1*
### Installing required library
```cmd
!pip install -r requirements.txt
```
### Import Library
```python
import os
import tensorflow as tf
import numpy as np

from PIL import Image
from skimage import transform
from tqdm import tqdm
from tensorflow.keras import models
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
## Dataset Preparation
We are using Open-Source Framework to download the selected categorized Dataset. 

There are 4 categories that we used *categories* - (train, test, validation):
1. Ambulance - (338, 51, 12)
2. Bus - (1000, 247, 73)
3. Car - (1000, 1000, 1000)
4. Truck - (1000, 820, 269)

### Dataset : [**Open Image Dataset v6**](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&r=false)

### Framework : [OIDv6](https://pypi.org/project/oidv6/).

#### Aquire the selected categorized Dataset and limiting maximum image categories to 1000 images. 
```
!oidv6 downloader --dataset OIDv6/ --type_data all --classes Ambulance Bus Car Truck Van --limit 1000 --yes 
```

The Dataset will be Saved in the Following Structure
```
OIDv6/test/ambulance/labels
 ├── 📂test
 │ ├── 📂ambulance
 │ │ ├── 📂labels
 │ │ │ ├── 📃image1_label.txt...image(n)_label.txt
 │ │ ├── 🖼️image1.jpg...image(n).jpg
 │ ├── 📂bus
 │ ├── 📂car
 │ ├── 📂truck
 ├── 📂train
 ├── 📂validation
```

## Dataset Preprocessing
To minimize Overfit and improving variance in the training data, we used `ImageDataGenerator` function to augment the train Dataset. The fuction respectively label each image based on their folder name.
```python
train_datagen = ImageDataGenerator(
    rescale=(1/255.),              # normalize the image vector, by dividing with 255.
    width_shift_range=0.2,         # randomize shifting width in the range of 0.2
    height_shift_range=0.2,        # randomize shifting height in the range of 0.2
    zoom_range=0.2,                # randomize zoom in the range of 0.2
    shear_range=0.2,               # randomize shear in the range of 0.2
    rotation_range=20,             # randomize rotation in the range of 20 degree
    brightness_range=[0.8,1.2],    # randomize brightness in between 0.8 - 1.2
    horizontal_flip=True,          # randomly flipping the image
    #fill_mode="nearest"           # use fill mode if dataset background are plain color
    )
    

test_datagen=ImageDataGenerator(
    rescale=(1/255.)               # normalize the image vector, by dividing with 255.
    )

validation_datagen=ImageDataGenerator(
    rescale=(1/255.)
    )

traindir = "OIDv6/train"           # defining dataset directory
testdir = "OIDv6/test"
valtdir = "OIDv6/validation"

train_generator=train_datagen.flow_from_directory(
    traindir,
    target_size =(224, 224),       # rescale the image into 224 x 224 to be matched as model input scale
    class_mode='categorical',      # type of label arrays that are returned
    batch_size=32                  # make image batch size to 32, train step = totalImg/ batch
    )

test_generator=test_datagen.flow_from_directory(
    testdir,
    target_size =(224, 224),
    class_mode='categorical',
    batch_size=32
    )

validation_generator=test_datagen.flow_from_directory(
    valtdir,
    target_size =(224, 224),
    class_mode='categorical',
    batch_size=32
    )

class_weights = class_weight.compute_class_weight(
    'balanced',
    np.unique(train_generator.classes),
    train_generator.classes
    )

print(class_weights)
```
```python
Found 3338 images belonging to 5 classes.
Found 2243 images belonging to 5 classes.
Found 1399 images belonging to 5 classes.

{'.ipynb_checkpoints': 0, 'ambulance': 1, 'bus': 2, 'car': 3, 'truck': 4}
```

### Plotting the Augmented image
```python
target_labels = next(os.walk(traindir))[1]
target_labels.sort()
batch = next(train_generator)
batch_images = np.array(batch[0])
batch_labels = np.array(batch[1])

target_labels = np.asarray(target_labels)

plt.figure(figsize=(15,10))
for n, i in enumerate(np.arange(10)):
    ax = plt.subplot(3,5,n+1)
    plt.imshow(batch_images[i])
    plt.title(target_labels[np.where(batch_labels[i]==1)[0][0]])
    plt.axis('off')
```
![image](https://user-images.githubusercontent.com/12151051/120280640-24c1a500-c2e2-11eb-8514-71b170d42e62.png)


## Building the Model
For the model we use MobileNet V2 by transfer learning and fine-tuning to our dataset.
### Defining Input Shape
```python
IMG_SIZE = (224,224)
IMG_SHAPE = IMG_SIZE + (3,)    # Result shape (3, 224, 224)
```

Instantiate a `MobileNet V2` model pre-loaded with `weights` trained on `ImageNet`. By specifying the `include_top = False` argument, it doesn't include the classification layers at the top, which is ideal for feature extraction.
```python
base_model = tf.keras.applications.MobileNetV2(
                 input_shape = IMG_SHAPE,
                 include_top = False,
                 weights = 'imagenet'
                 )
```
Looking inside the base model looks like
```python
base_model.summary()
```
<details> 
<summary>
Model: "mobilenetv2_1.00_224"
</summary>
<pre>
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_2 (InputLayer)            [(None, 224, 224, 3) 0                                            
__________________________________________________________________________________________________
Conv1 (Conv2D)                  (None, 112, 112, 32) 864         input_2[0][0]                    
__________________________________________________________________________________________________
bn_Conv1 (BatchNormalization)   (None, 112, 112, 32) 128         Conv1[0][0]                      
__________________________________________________________________________________________________
Conv1_relu (ReLU)               (None, 112, 112, 32) 0           bn_Conv1[0][0]                   
__________________________________________________________________________________________________
expanded_conv_depthwise (Depthw (None, 112, 112, 32) 288         Conv1_relu[0][0]                 
__________________________________________________________________________________________________
expanded_conv_depthwise_BN (Bat (None, 112, 112, 32) 128         expanded_conv_depthwise[0][0]    
__________________________________________________________________________________________________
expanded_conv_depthwise_relu (R (None, 112, 112, 32) 0           expanded_conv_depthwise_BN[0][0] 
__________________________________________________________________________________________________
expanded_conv_project (Conv2D)  (None, 112, 112, 16) 512         expanded_conv_depthwise_relu[0][0
__________________________________________________________________________________________________
expanded_conv_project_BN (Batch (None, 112, 112, 16) 64          expanded_conv_project[0][0]      
__________________________________________________________________________________________________
block_1_expand (Conv2D)         (None, 112, 112, 96) 1536        expanded_conv_project_BN[0][0]   
__________________________________________________________________________________________________
block_1_expand_BN (BatchNormali (None, 112, 112, 96) 384         block_1_expand[0][0]             
__________________________________________________________________________________________________
block_1_expand_relu (ReLU)      (None, 112, 112, 96) 0           block_1_expand_BN[0][0]          
__________________________________________________________________________________________________
block_1_pad (ZeroPadding2D)     (None, 113, 113, 96) 0           block_1_expand_relu[0][0]        
__________________________________________________________________________________________________
block_1_depthwise (DepthwiseCon (None, 56, 56, 96)   864         block_1_pad[0][0]                
__________________________________________________________________________________________________
block_1_depthwise_BN (BatchNorm (None, 56, 56, 96)   384         block_1_depthwise[0][0]          
__________________________________________________________________________________________________
block_1_depthwise_relu (ReLU)   (None, 56, 56, 96)   0           block_1_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_1_project (Conv2D)        (None, 56, 56, 24)   2304        block_1_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_1_project_BN (BatchNormal (None, 56, 56, 24)   96          block_1_project[0][0]            
__________________________________________________________________________________________________
block_2_expand (Conv2D)         (None, 56, 56, 144)  3456        block_1_project_BN[0][0]         
__________________________________________________________________________________________________
block_2_expand_BN (BatchNormali (None, 56, 56, 144)  576         block_2_expand[0][0]             
__________________________________________________________________________________________________
block_2_expand_relu (ReLU)      (None, 56, 56, 144)  0           block_2_expand_BN[0][0]          
__________________________________________________________________________________________________
block_2_depthwise (DepthwiseCon (None, 56, 56, 144)  1296        block_2_expand_relu[0][0]        
__________________________________________________________________________________________________
block_2_depthwise_BN (BatchNorm (None, 56, 56, 144)  576         block_2_depthwise[0][0]          
__________________________________________________________________________________________________
block_2_depthwise_relu (ReLU)   (None, 56, 56, 144)  0           block_2_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_2_project (Conv2D)        (None, 56, 56, 24)   3456        block_2_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_2_project_BN (BatchNormal (None, 56, 56, 24)   96          block_2_project[0][0]            
__________________________________________________________________________________________________
block_2_add (Add)               (None, 56, 56, 24)   0           block_1_project_BN[0][0]         
                                                                 block_2_project_BN[0][0]         
__________________________________________________________________________________________________
block_3_expand (Conv2D)         (None, 56, 56, 144)  3456        block_2_add[0][0]                
__________________________________________________________________________________________________
block_3_expand_BN (BatchNormali (None, 56, 56, 144)  576         block_3_expand[0][0]             
__________________________________________________________________________________________________
block_3_expand_relu (ReLU)      (None, 56, 56, 144)  0           block_3_expand_BN[0][0]          
__________________________________________________________________________________________________
block_3_pad (ZeroPadding2D)     (None, 57, 57, 144)  0           block_3_expand_relu[0][0]        
__________________________________________________________________________________________________
block_3_depthwise (DepthwiseCon (None, 28, 28, 144)  1296        block_3_pad[0][0]                
__________________________________________________________________________________________________
block_3_depthwise_BN (BatchNorm (None, 28, 28, 144)  576         block_3_depthwise[0][0]          
__________________________________________________________________________________________________
block_3_depthwise_relu (ReLU)   (None, 28, 28, 144)  0           block_3_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_3_project (Conv2D)        (None, 28, 28, 32)   4608        block_3_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_3_project_BN (BatchNormal (None, 28, 28, 32)   128         block_3_project[0][0]            
__________________________________________________________________________________________________
block_4_expand (Conv2D)         (None, 28, 28, 192)  6144        block_3_project_BN[0][0]         
__________________________________________________________________________________________________
block_4_expand_BN (BatchNormali (None, 28, 28, 192)  768         block_4_expand[0][0]             
__________________________________________________________________________________________________
block_4_expand_relu (ReLU)      (None, 28, 28, 192)  0           block_4_expand_BN[0][0]          
__________________________________________________________________________________________________
block_4_depthwise (DepthwiseCon (None, 28, 28, 192)  1728        block_4_expand_relu[0][0]        
__________________________________________________________________________________________________
block_4_depthwise_BN (BatchNorm (None, 28, 28, 192)  768         block_4_depthwise[0][0]          
__________________________________________________________________________________________________
block_4_depthwise_relu (ReLU)   (None, 28, 28, 192)  0           block_4_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_4_project (Conv2D)        (None, 28, 28, 32)   6144        block_4_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_4_project_BN (BatchNormal (None, 28, 28, 32)   128         block_4_project[0][0]            
__________________________________________________________________________________________________
block_4_add (Add)               (None, 28, 28, 32)   0           block_3_project_BN[0][0]         
                                                                 block_4_project_BN[0][0]         
__________________________________________________________________________________________________
block_5_expand (Conv2D)         (None, 28, 28, 192)  6144        block_4_add[0][0]                
__________________________________________________________________________________________________
block_5_expand_BN (BatchNormali (None, 28, 28, 192)  768         block_5_expand[0][0]             
__________________________________________________________________________________________________
block_5_expand_relu (ReLU)      (None, 28, 28, 192)  0           block_5_expand_BN[0][0]          
__________________________________________________________________________________________________
block_5_depthwise (DepthwiseCon (None, 28, 28, 192)  1728        block_5_expand_relu[0][0]        
__________________________________________________________________________________________________
block_5_depthwise_BN (BatchNorm (None, 28, 28, 192)  768         block_5_depthwise[0][0]          
__________________________________________________________________________________________________
block_5_depthwise_relu (ReLU)   (None, 28, 28, 192)  0           block_5_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_5_project (Conv2D)        (None, 28, 28, 32)   6144        block_5_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_5_project_BN (BatchNormal (None, 28, 28, 32)   128         block_5_project[0][0]            
__________________________________________________________________________________________________
block_5_add (Add)               (None, 28, 28, 32)   0           block_4_add[0][0]                
                                                                 block_5_project_BN[0][0]         
__________________________________________________________________________________________________
block_6_expand (Conv2D)         (None, 28, 28, 192)  6144        block_5_add[0][0]                
__________________________________________________________________________________________________
block_6_expand_BN (BatchNormali (None, 28, 28, 192)  768         block_6_expand[0][0]             
__________________________________________________________________________________________________
block_6_expand_relu (ReLU)      (None, 28, 28, 192)  0           block_6_expand_BN[0][0]          
__________________________________________________________________________________________________
block_6_pad (ZeroPadding2D)     (None, 29, 29, 192)  0           block_6_expand_relu[0][0]        
__________________________________________________________________________________________________
block_6_depthwise (DepthwiseCon (None, 14, 14, 192)  1728        block_6_pad[0][0]                
__________________________________________________________________________________________________
block_6_depthwise_BN (BatchNorm (None, 14, 14, 192)  768         block_6_depthwise[0][0]          
__________________________________________________________________________________________________
block_6_depthwise_relu (ReLU)   (None, 14, 14, 192)  0           block_6_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_6_project (Conv2D)        (None, 14, 14, 64)   12288       block_6_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_6_project_BN (BatchNormal (None, 14, 14, 64)   256         block_6_project[0][0]            
__________________________________________________________________________________________________
block_7_expand (Conv2D)         (None, 14, 14, 384)  24576       block_6_project_BN[0][0]         
__________________________________________________________________________________________________
block_7_expand_BN (BatchNormali (None, 14, 14, 384)  1536        block_7_expand[0][0]             
__________________________________________________________________________________________________
block_7_expand_relu (ReLU)      (None, 14, 14, 384)  0           block_7_expand_BN[0][0]          
__________________________________________________________________________________________________
block_7_depthwise (DepthwiseCon (None, 14, 14, 384)  3456        block_7_expand_relu[0][0]        
__________________________________________________________________________________________________
block_7_depthwise_BN (BatchNorm (None, 14, 14, 384)  1536        block_7_depthwise[0][0]          
__________________________________________________________________________________________________
block_7_depthwise_relu (ReLU)   (None, 14, 14, 384)  0           block_7_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_7_project (Conv2D)        (None, 14, 14, 64)   24576       block_7_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_7_project_BN (BatchNormal (None, 14, 14, 64)   256         block_7_project[0][0]            
__________________________________________________________________________________________________
block_7_add (Add)               (None, 14, 14, 64)   0           block_6_project_BN[0][0]         
                                                                 block_7_project_BN[0][0]         
__________________________________________________________________________________________________
block_8_expand (Conv2D)         (None, 14, 14, 384)  24576       block_7_add[0][0]                
__________________________________________________________________________________________________
block_8_expand_BN (BatchNormali (None, 14, 14, 384)  1536        block_8_expand[0][0]             
__________________________________________________________________________________________________
block_8_expand_relu (ReLU)      (None, 14, 14, 384)  0           block_8_expand_BN[0][0]          
__________________________________________________________________________________________________
block_8_depthwise (DepthwiseCon (None, 14, 14, 384)  3456        block_8_expand_relu[0][0]        
__________________________________________________________________________________________________
block_8_depthwise_BN (BatchNorm (None, 14, 14, 384)  1536        block_8_depthwise[0][0]          
__________________________________________________________________________________________________
block_8_depthwise_relu (ReLU)   (None, 14, 14, 384)  0           block_8_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_8_project (Conv2D)        (None, 14, 14, 64)   24576       block_8_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_8_project_BN (BatchNormal (None, 14, 14, 64)   256         block_8_project[0][0]            
__________________________________________________________________________________________________
block_8_add (Add)               (None, 14, 14, 64)   0           block_7_add[0][0]                
                                                                 block_8_project_BN[0][0]         
__________________________________________________________________________________________________
block_9_expand (Conv2D)         (None, 14, 14, 384)  24576       block_8_add[0][0]                
__________________________________________________________________________________________________
block_9_expand_BN (BatchNormali (None, 14, 14, 384)  1536        block_9_expand[0][0]             
__________________________________________________________________________________________________
block_9_expand_relu (ReLU)      (None, 14, 14, 384)  0           block_9_expand_BN[0][0]          
__________________________________________________________________________________________________
block_9_depthwise (DepthwiseCon (None, 14, 14, 384)  3456        block_9_expand_relu[0][0]        
__________________________________________________________________________________________________
block_9_depthwise_BN (BatchNorm (None, 14, 14, 384)  1536        block_9_depthwise[0][0]          
__________________________________________________________________________________________________
block_9_depthwise_relu (ReLU)   (None, 14, 14, 384)  0           block_9_depthwise_BN[0][0]       
__________________________________________________________________________________________________
block_9_project (Conv2D)        (None, 14, 14, 64)   24576       block_9_depthwise_relu[0][0]     
__________________________________________________________________________________________________
block_9_project_BN (BatchNormal (None, 14, 14, 64)   256         block_9_project[0][0]            
__________________________________________________________________________________________________
block_9_add (Add)               (None, 14, 14, 64)   0           block_8_add[0][0]                
                                                                 block_9_project_BN[0][0]         
__________________________________________________________________________________________________
block_10_expand (Conv2D)        (None, 14, 14, 384)  24576       block_9_add[0][0]                
__________________________________________________________________________________________________
block_10_expand_BN (BatchNormal (None, 14, 14, 384)  1536        block_10_expand[0][0]            
__________________________________________________________________________________________________
block_10_expand_relu (ReLU)     (None, 14, 14, 384)  0           block_10_expand_BN[0][0]         
__________________________________________________________________________________________________
block_10_depthwise (DepthwiseCo (None, 14, 14, 384)  3456        block_10_expand_relu[0][0]       
__________________________________________________________________________________________________
block_10_depthwise_BN (BatchNor (None, 14, 14, 384)  1536        block_10_depthwise[0][0]         
__________________________________________________________________________________________________
block_10_depthwise_relu (ReLU)  (None, 14, 14, 384)  0           block_10_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_10_project (Conv2D)       (None, 14, 14, 96)   36864       block_10_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_10_project_BN (BatchNorma (None, 14, 14, 96)   384         block_10_project[0][0]           
__________________________________________________________________________________________________
block_11_expand (Conv2D)        (None, 14, 14, 576)  55296       block_10_project_BN[0][0]        
__________________________________________________________________________________________________
block_11_expand_BN (BatchNormal (None, 14, 14, 576)  2304        block_11_expand[0][0]            
__________________________________________________________________________________________________
block_11_expand_relu (ReLU)     (None, 14, 14, 576)  0           block_11_expand_BN[0][0]         
__________________________________________________________________________________________________
block_11_depthwise (DepthwiseCo (None, 14, 14, 576)  5184        block_11_expand_relu[0][0]       
__________________________________________________________________________________________________
block_11_depthwise_BN (BatchNor (None, 14, 14, 576)  2304        block_11_depthwise[0][0]         
__________________________________________________________________________________________________
block_11_depthwise_relu (ReLU)  (None, 14, 14, 576)  0           block_11_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_11_project (Conv2D)       (None, 14, 14, 96)   55296       block_11_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_11_project_BN (BatchNorma (None, 14, 14, 96)   384         block_11_project[0][0]           
__________________________________________________________________________________________________
block_11_add (Add)              (None, 14, 14, 96)   0           block_10_project_BN[0][0]        
                                                                 block_11_project_BN[0][0]        
__________________________________________________________________________________________________
block_12_expand (Conv2D)        (None, 14, 14, 576)  55296       block_11_add[0][0]               
__________________________________________________________________________________________________
block_12_expand_BN (BatchNormal (None, 14, 14, 576)  2304        block_12_expand[0][0]            
__________________________________________________________________________________________________
block_12_expand_relu (ReLU)     (None, 14, 14, 576)  0           block_12_expand_BN[0][0]         
__________________________________________________________________________________________________
block_12_depthwise (DepthwiseCo (None, 14, 14, 576)  5184        block_12_expand_relu[0][0]       
__________________________________________________________________________________________________
block_12_depthwise_BN (BatchNor (None, 14, 14, 576)  2304        block_12_depthwise[0][0]         
__________________________________________________________________________________________________
block_12_depthwise_relu (ReLU)  (None, 14, 14, 576)  0           block_12_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_12_project (Conv2D)       (None, 14, 14, 96)   55296       block_12_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_12_project_BN (BatchNorma (None, 14, 14, 96)   384         block_12_project[0][0]           
__________________________________________________________________________________________________
block_12_add (Add)              (None, 14, 14, 96)   0           block_11_add[0][0]               
                                                                 block_12_project_BN[0][0]        
__________________________________________________________________________________________________
block_13_expand (Conv2D)        (None, 14, 14, 576)  55296       block_12_add[0][0]               
__________________________________________________________________________________________________
block_13_expand_BN (BatchNormal (None, 14, 14, 576)  2304        block_13_expand[0][0]            
__________________________________________________________________________________________________
block_13_expand_relu (ReLU)     (None, 14, 14, 576)  0           block_13_expand_BN[0][0]         
__________________________________________________________________________________________________
block_13_pad (ZeroPadding2D)    (None, 15, 15, 576)  0           block_13_expand_relu[0][0]       
__________________________________________________________________________________________________
block_13_depthwise (DepthwiseCo (None, 7, 7, 576)    5184        block_13_pad[0][0]               
__________________________________________________________________________________________________
block_13_depthwise_BN (BatchNor (None, 7, 7, 576)    2304        block_13_depthwise[0][0]         
__________________________________________________________________________________________________
block_13_depthwise_relu (ReLU)  (None, 7, 7, 576)    0           block_13_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_13_project (Conv2D)       (None, 7, 7, 160)    92160       block_13_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_13_project_BN (BatchNorma (None, 7, 7, 160)    640         block_13_project[0][0]           
__________________________________________________________________________________________________
block_14_expand (Conv2D)        (None, 7, 7, 960)    153600      block_13_project_BN[0][0]        
__________________________________________________________________________________________________
block_14_expand_BN (BatchNormal (None, 7, 7, 960)    3840        block_14_expand[0][0]            
__________________________________________________________________________________________________
block_14_expand_relu (ReLU)     (None, 7, 7, 960)    0           block_14_expand_BN[0][0]         
__________________________________________________________________________________________________
block_14_depthwise (DepthwiseCo (None, 7, 7, 960)    8640        block_14_expand_relu[0][0]       
__________________________________________________________________________________________________
block_14_depthwise_BN (BatchNor (None, 7, 7, 960)    3840        block_14_depthwise[0][0]         
__________________________________________________________________________________________________
block_14_depthwise_relu (ReLU)  (None, 7, 7, 960)    0           block_14_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_14_project (Conv2D)       (None, 7, 7, 160)    153600      block_14_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_14_project_BN (BatchNorma (None, 7, 7, 160)    640         block_14_project[0][0]           
__________________________________________________________________________________________________
block_14_add (Add)              (None, 7, 7, 160)    0           block_13_project_BN[0][0]        
                                                                 block_14_project_BN[0][0]        
__________________________________________________________________________________________________
block_15_expand (Conv2D)        (None, 7, 7, 960)    153600      block_14_add[0][0]               
__________________________________________________________________________________________________
block_15_expand_BN (BatchNormal (None, 7, 7, 960)    3840        block_15_expand[0][0]            
__________________________________________________________________________________________________
block_15_expand_relu (ReLU)     (None, 7, 7, 960)    0           block_15_expand_BN[0][0]         
__________________________________________________________________________________________________
block_15_depthwise (DepthwiseCo (None, 7, 7, 960)    8640        block_15_expand_relu[0][0]       
__________________________________________________________________________________________________
block_15_depthwise_BN (BatchNor (None, 7, 7, 960)    3840        block_15_depthwise[0][0]         
__________________________________________________________________________________________________
block_15_depthwise_relu (ReLU)  (None, 7, 7, 960)    0           block_15_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_15_project (Conv2D)       (None, 7, 7, 160)    153600      block_15_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_15_project_BN (BatchNorma (None, 7, 7, 160)    640         block_15_project[0][0]           
__________________________________________________________________________________________________
block_15_add (Add)              (None, 7, 7, 160)    0           block_14_add[0][0]               
                                                                 block_15_project_BN[0][0]        
__________________________________________________________________________________________________
block_16_expand (Conv2D)        (None, 7, 7, 960)    153600      block_15_add[0][0]               
__________________________________________________________________________________________________
block_16_expand_BN (BatchNormal (None, 7, 7, 960)    3840        block_16_expand[0][0]            
__________________________________________________________________________________________________
block_16_expand_relu (ReLU)     (None, 7, 7, 960)    0           block_16_expand_BN[0][0]         
__________________________________________________________________________________________________
block_16_depthwise (DepthwiseCo (None, 7, 7, 960)    8640        block_16_expand_relu[0][0]       
__________________________________________________________________________________________________
block_16_depthwise_BN (BatchNor (None, 7, 7, 960)    3840        block_16_depthwise[0][0]         
__________________________________________________________________________________________________
block_16_depthwise_relu (ReLU)  (None, 7, 7, 960)    0           block_16_depthwise_BN[0][0]      
__________________________________________________________________________________________________
block_16_project (Conv2D)       (None, 7, 7, 320)    307200      block_16_depthwise_relu[0][0]    
__________________________________________________________________________________________________
block_16_project_BN (BatchNorma (None, 7, 7, 320)    1280        block_16_project[0][0]           
__________________________________________________________________________________________________
Conv_1 (Conv2D)                 (None, 7, 7, 1280)   409600      block_16_project_BN[0][0]        
__________________________________________________________________________________________________
Conv_1_bn (BatchNormalization)  (None, 7, 7, 1280)   5120        Conv_1[0][0]                     
__________________________________________________________________________________________________
out_relu (ReLU)                 (None, 7, 7, 1280)   0           Conv_1_bn[0][0]                  
==================================================================================================
Total params: 2,257,984
Trainable params: 2,223,872
Non-trainable params: 34,112
__________________________________________________________________________________________________
</pre>
</details>

### Converts each image into a block of features.
```python
image_batch, label_batch = next(iter(train_generator))
feature_batch = base_model(image_batch)
```

### Freeze the convolutional layers
```python
base_model.trainable = False
```

### Adding Classification Head
```python
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(5, activation='softmax')
])
```
### Compiling the Model
```python
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
Compiled Model After Freezing Model and Add Classification Head
```python
model.summary()
```
<details> 
   <summary>
Model: "sequential_1"
   </summary>
   <pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Functi (None, 7, 7, 1280)        2257984   
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1280)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1280)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               327936    
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 1285      
=================================================================
Total params: 2,587,205
Trainable params: 329,221
Non-trainable params: 2,257,984
_________________________________________________________________
   </pre>
</details>

set 5 epoch to see the model initial accuracy
```python
initial_epochs = 5
```
### Train Model for 5 Epochs *Before Fine-Tuning*
```python
history = model.fit(
             train_generator,
             epochs = initial_epochs,
             alidation_data = validation_generator
                    )             
```
<details> 
   <summary>
Model: "sequential_1"
   </summary>
   <pre>
Epoch 1/5
105/105 [==============================] - 247s 2s/step - loss: 1.5021 - accuracy: 0.3531 - val_loss: 0.6067 - val_accuracy: 0.8070
Epoch 2/5
105/105 [==============================] - 242s 2s/step - loss: 0.8688 - accuracy: 0.6671 - val_loss: 0.5626 - val_accuracy: 0.8063
Epoch 3/5
105/105 [==============================] - 240s 2s/step - loss: 0.8078 - accuracy: 0.6889 - val_loss: 0.4774 - val_accuracy: 0.8320
Epoch 4/5
105/105 [==============================] - 241s 2s/step - loss: 0.7713 - accuracy: 0.6911 - val_loss: 0.4868 - val_accuracy: 0.8234
Epoch 5/5
105/105 [==============================] - 243s 2s/step - loss: 0.7265 - accuracy: 0.7187 - val_loss: 0.4787 - val_accuracy: 0.8292
   </pre>
</details>

### Fine-Tuning
We were using the last 20% layer to be un-freeze for the model to get some features from our dataset, and minimizing overfit.
```python
#Un-Freeze Top Layer
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 123            # Freeze first 80% from total 154 Layers

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
```

### Compiled Model After Fine-Tuning
```python
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              metrics = ['accuracy']
              )
              
model.summary()
```
<details> 
   <summary>
Model: "sequential_1"
   </summary>
   <pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Functi (None, 7, 7, 1280)        2257984   
_________________________________________________________________
global_average_pooling2d_1 ( (None, 1280)              0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 1280)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               327936    
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 1285      
=================================================================
Total params: 2,587,205
Trainable params: 1,947,781
Non-trainable params: 639,424
_________________________________________________________________
   </pre>
</details>

### Defining Scheduled Learning-rate Decay
to minimize overfit even further we used Scheduled Learning-rate Decay.
```python
# creating scheduled learning rate decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    base_learning_rate,     # Base is 0.0001
    decay_steps = 50,       # LR will decay every 50 step
    decay_rate = 0.9
    )
```

### Make Callback to Save Model Weights
```python
checkpoint_path = "new_checkpoint/cp_rev_1.ckpt"         # Checkpoint Save Path
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                  filepath = checkpoint_path,
                  save_weights_only = True,
                  verbose = 1
                  )
```

 ### Training
 We then train the fine-tuned model for another 25 epochs
 ```python
fine_tune_epochs = 25
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_generator,
                         epochs = total_epochs,
                         initial_epoch = history.epoch[-1],
                         validation_data = validation_generator,
                         callbacks = [cp_callback]
                         )
 ```
 <details> 
   <summary>
Training Result
   </summary>
   <pre>
Epoch 5/30
105/105 [==============================] - 277s 3s/step - loss: 0.8110 - accuracy: 0.6882 - val_loss: 0.7771 - val_accuracy: 0.7269

Epoch 00005: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 6/30
105/105 [==============================] - 267s 3s/step - loss: 0.6348 - accuracy: 0.7561 - val_loss: 0.6849 - val_accuracy: 0.7527

Epoch 00006: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 7/30
105/105 [==============================] - 268s 3s/step - loss: 0.5777 - accuracy: 0.7833 - val_loss: 0.6009 - val_accuracy: 0.7756

Epoch 00007: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 8/30
105/105 [==============================] - 266s 3s/step - loss: 0.5130 - accuracy: 0.8005 - val_loss: 0.5936 - val_accuracy: 0.7898

Epoch 00008: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 9/30
105/105 [==============================] - 268s 3s/step - loss: 0.4560 - accuracy: 0.8345 - val_loss: 0.5607 - val_accuracy: 0.8020

Epoch 00009: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 10/30
105/105 [==============================] - 266s 3s/step - loss: 0.4474 - accuracy: 0.8340 - val_loss: 0.5086 - val_accuracy: 0.8149

Epoch 00010: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 11/30
105/105 [==============================] - 269s 3s/step - loss: 0.4351 - accuracy: 0.8303 - val_loss: 0.5023 - val_accuracy: 0.8156

Epoch 00011: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 12/30
105/105 [==============================] - 267s 3s/step - loss: 0.4047 - accuracy: 0.8398 - val_loss: 0.5006 - val_accuracy: 0.8199

Epoch 00012: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 13/30
105/105 [==============================] - 266s 3s/step - loss: 0.4383 - accuracy: 0.8376 - val_loss: 0.4754 - val_accuracy: 0.8327

Epoch 00013: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 14/30
105/105 [==============================] - 268s 3s/step - loss: 0.4231 - accuracy: 0.8533 - val_loss: 0.4609 - val_accuracy: 0.8406

Epoch 00014: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 15/30
105/105 [==============================] - 265s 3s/step - loss: 0.3761 - accuracy: 0.8655 - val_loss: 0.4609 - val_accuracy: 0.8399

Epoch 00015: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 16/30
105/105 [==============================] - 268s 3s/step - loss: 0.4034 - accuracy: 0.8457 - val_loss: 0.4639 - val_accuracy: 0.8406

Epoch 00016: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 17/30
105/105 [==============================] - 267s 3s/step - loss: 0.3933 - accuracy: 0.8612 - val_loss: 0.4579 - val_accuracy: 0.8449

Epoch 00017: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 18/30
105/105 [==============================] - 268s 3s/step - loss: 0.3665 - accuracy: 0.8624 - val_loss: 0.4530 - val_accuracy: 0.8463

Epoch 00018: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 19/30
105/105 [==============================] - 265s 3s/step - loss: 0.3770 - accuracy: 0.8660 - val_loss: 0.4486 - val_accuracy: 0.8485

Epoch 00019: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 20/30
105/105 [==============================] - 269s 3s/step - loss: 0.3736 - accuracy: 0.8588 - val_loss: 0.4473 - val_accuracy: 0.8506

Epoch 00020: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 21/30
105/105 [==============================] - 265s 3s/step - loss: 0.3645 - accuracy: 0.8610 - val_loss: 0.4445 - val_accuracy: 0.8513

Epoch 00021: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 22/30
105/105 [==============================] - 267s 3s/step - loss: 0.3682 - accuracy: 0.8640 - val_loss: 0.4439 - val_accuracy: 0.8513

Epoch 00022: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 23/30
105/105 [==============================] - 267s 3s/step - loss: 0.3690 - accuracy: 0.8605 - val_loss: 0.4430 - val_accuracy: 0.8513

Epoch 00023: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 24/30
105/105 [==============================] - 267s 3s/step - loss: 0.3595 - accuracy: 0.8630 - val_loss: 0.4413 - val_accuracy: 0.8506

Epoch 00024: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 25/30
105/105 [==============================] - 266s 3s/step - loss: 0.3451 - accuracy: 0.8644 - val_loss: 0.4393 - val_accuracy: 0.8535

Epoch 00025: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 26/30
105/105 [==============================] - 267s 3s/step - loss: 0.3679 - accuracy: 0.8658 - val_loss: 0.4384 - val_accuracy: 0.8556

Epoch 00026: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 27/30
105/105 [==============================] - 268s 3s/step - loss: 0.3858 - accuracy: 0.8454 - val_loss: 0.4377 - val_accuracy: 0.8578

Epoch 00027: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 28/30
105/105 [==============================] - 267s 3s/step - loss: 0.3726 - accuracy: 0.8608 - val_loss: 0.4371 - val_accuracy: 0.8570

Epoch 00028: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 29/30
105/105 [==============================] - 267s 3s/step - loss: 0.3850 - accuracy: 0.8614 - val_loss: 0.4364 - val_accuracy: 0.8563

Epoch 00029: saving model to new_checkpoint/cp_rev_1.ckpt
Epoch 30/30
105/105 [==============================] - 267s 3s/step - loss: 0.3908 - accuracy: 0.8437 - val_loss: 0.4356 - val_accuracy: 0.8556

Epoch 00030: saving model to new_checkpoint/cp_rev_1.ckpt
   </pre>
</details>

### Accuracy
Plotting the training and validation accuracy and loss
```python
acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.5, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```
![image](https://user-images.githubusercontent.com/12151051/120288352-4f176080-c2ea-11eb-8bdf-eb00453b50fb.png)

As we can see from the graph, the model is not suffering to much from overfit, we can see it has found its convergence (no significant growth in the graph), It's also have a good Training and Validation Accuracy after just 25 epochs.

### Evaluate Test Accuracy
```python
loss, accuracy = model.evaluate(test_generator)
print('Test accuracy :', accuracy)
```
```
71/71 [==============================] - 129s 2s/step - loss: 0.5047 - accuracy: 0.8235
Test accuracy : 0.8234507441520691
```

### Saving the Model
Saving as TF format using
```python
save_path = 'Model/TheATeam_model_ver2'
model.save(save_path)
```

The exported model will be structered as follows
```
Model
 ├── 📂TheATeam_model_ver2
 │ ├── 📂Assets                    # Contains files used by the TensorFlow graph(not used now).
 │ ├── 📂Variables                 # Contains a standard training checkpoint
 │ ├── 📃saved_model.pb            # The saved model
```

We also save the model as `HDF5` format, using
```python
file_name = 'TheATeam_model_ver2.h5'
model.save( file_name,save_format='h5' )
```

## Testing
To test the model, we proposed a video file (.mp4) classification and not using CCTV input stream yet.

### Importing Required Library
```python
import tensorflow as tf
import numpy as np
import cv2
import pytube
import os

from PIL import Image
from skimage import transform
```

### Load the Model
```python

my_model = tf.keras.models.load_model('Model/TheATeam_model_ver2', compile = True)
```
*Make the compile argument* `True` *to compile the model after loading.*

*The example video that we are using [Link](https://youtu.be/bnX1JqglJ2E).*
### Infference Function
```python
#5 Load and pre-process image frames
 def load_frames(frame):
     frames = Image.open(frame)
     frames = np.array(frames).astype('float32')/255
     frames = transform.resize(frames, (224, 224, 3))
     frames = np.expand_dims(frames, axis=0)
     return frames

 #1 get video
 vidcap = cv2.VideoCapture('../ambulance.mp4')

 #3 converting video into frame image (jpg format)
 def getFrame(sec):
     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
     hasFrames,image = vidcap.read()

     if hasFrames:
         # Specify frame path file
         framePath = "../video-frames/"+str(count)+"_frame.jpg"
         # save frame as JPG file
         cv2.imwrite(framePath, image)

         #4 Load and Predict Frame directly
         image = load_frames(framePath)
         result = my_model.predict(image)

         #6 Print ambulance detected or not and probability value
          predict_result = (str(count)+") Ambulance Detected: {}".format("%.3f" % result[0][1]) if result[0][1]>0.03 
              else str(count)+") Ambulance not detected: {}".format("%.3f" % result[0][1]))
     return hasFrames

 sec = 0
 frameRate = 5               # Capture Image in Second
 count = 1                   # Video Frame Count
 success = getFrame(sec)     # Initial Function to Get the Frame and Predict the Frame

 #2 Looping the function to get the frame and predict frame directly
 while success:
     count = count + 1
     sec = sec + frameRate
     sec = round(sec, 2)
     success = getFrame(sec)
```
### Result
<details> 
   <summary>
    Test Result
   </summary>
   <pre>
1) Ambulance Detected: 0.059
2) Ambulance not detected: 0.001
3) Ambulance Detected: 0.666
4) Ambulance Detected: 0.441
5) Ambulance Detected: 0.543
6) Ambulance Detected: 0.700
7) Ambulance not detected: 0.002
8) Ambulance not detected: 0.006
9) Ambulance not detected: 0.022
10) Ambulance not detected: 0.020
11) Ambulance Detected: 0.088
12) Ambulance not detected: 0.015
13) Ambulance not detected: 0.007
14) Ambulance Detected: 0.597
15) Ambulance Detected: 0.769
16) Ambulance Detected: 0.039
17) Ambulance Detected: 0.092
18) Ambulance Detected: 0.218
19) Ambulance Detected: 0.292
20) Ambulance Detected: 0.170
21) Ambulance Detected: 0.232
22) Ambulance not detected: 0.013
23) Ambulance Detected: 0.163
24) Ambulance Detected: 0.189
25) Ambulance not detected: 0.009
26) Ambulance Detected: 0.385
27) Ambulance Detected: 0.046
28) Ambulance Detected: 0.368
29) Ambulance Detected: 0.861
30) Ambulance Detected: 0.315
31) Ambulance not detected: 0.021
32) Ambulance Detected: 0.059
   </pre>
</details>
*Each Numeber is counted the same as the frame number sequences*

### Example Resulted Frame

![image](https://user-images.githubusercontent.com/12151051/120451739-e26c9680-c3bb-11eb-9f49-7e2e78247496.png)

![image](https://user-images.githubusercontent.com/12151051/120452318-5313b300-c3bc-11eb-812e-a718b17eb12c.png)


## Deployment
We are using Flask (a python framework) to deploy it in the server and serves as REST API. When the API is send, it will return predicted value as a JSON format.

Transfering from previous test code in flask, and add socket so it can be requested at any time.
```python
def predict_process():
    # Code here
    #loading the model
    my_model = tf.keras.models.load_model('./TheATeam_model_ver2.h5', compile=True)

    #5 Load and pre-process image frames
    def load_frames(frame):
        frames = Image.open(frame)
        frames = np.array(frames).astype('float32')/255
        frames = transform.resize(frames, (224, 224, 3))
        frames = np.expand_dims(frames, axis=0)
        return frames

    #1 get video
    vidcap = cv2.VideoCapture('../ambulance.mp4')

    #3 converting video into frame image (jpg format)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()

        if hasFrames:
            # Specify frame path file
            framePath = "../video-frames/"+str(count)+"_frame.jpg"
            # save frame as JPG file
            cv2.imwrite(framePath, image)

            #4 Load and Predict Frame directly
            image = load_frames(framePath)
            result = my_model.predict(image)

            #6 Return ambulance detected or not, probability value, and in which frame
            predict_result = {
                "ambulance_detected": 1 if result[0][1] > 0.03 else 0,
                "frame_number": count,
                "precentage": "{}".format("%.3f" % result[0][1])
            }
            emit('predict_result', json.dumps(predict_result), broadcast=True)

        return hasFrames

    sec = 0
    frameRate = 5 # Capture image in second
    count=1
    success = getFrame(sec) # Initial function to get the frame and predict frame

    # Looping the function to get the frame and predict frame directly
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
        
# Defining Socket that can be called by emit predict and see the result at the listener
@socketio.on('predict')
def predict(data):
    emit('predict_result', 'Predict Start', broadcast=True)
    predict_process()        
    emit('predict_result', 'Predict End', broadcast=True)
```
*The Complete Flask Code can be Found [Here}(https://github.com/Imanuella74/ambulance-cctv-detection/blob/main/ambulance-notifier-service/app.py)*
