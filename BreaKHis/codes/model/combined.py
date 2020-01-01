from tensorflow.keras.applications import InceptionV3,VGG16, Xception,VGG19
from tensorflow.keras.layers import Flatten,Dense,Dropout,GlobalAveragePooling2D,LeakyReLU,ELU ,ReLU, Concatenate, BatchNormalization, Input
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam


def build_combine_VGG16_VGG19(img_shape=(256,256,3),num_classes=1):
    
    input_common = Input(shape=img_shape, name="input_common")
    
    finetuning_layer = 17   #16
    finetuning_layer_vgg19 = 17  #19,17
    #vgg16
    vgg16 = VGG16(include_top=False, weights='imagenet')
    vgg16.trainable = False
    for layer in vgg16.layers[finetuning_layer:]:
        layer.trainable = True
    
    #vgg19
    vgg19 = VGG19(include_top=False, weights='imagenet')  # input_shape=img_shape
    vgg19.trainable = False 
    for layer in vgg19.layers[finetuning_layer_vgg19:]:
        layer.trainable = True  
    
    x1 = vgg16(input_common)
    x2 = vgg19(input_common)
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    x  = Concatenate()([x1,x2])
    x = BatchNormalization()(x)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = ReLU()(x)

    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    final_model = Model(inputs=[input_common], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return final_model
        
        
    