from tensorflow.keras import applications
from tensorflow.keras.layers import Input

vgg16=applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg16.save('/kw_resources/Master/Model/VGG16')
