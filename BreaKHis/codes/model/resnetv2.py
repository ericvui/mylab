from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Flatten,Dense,Dropout,GlobalAveragePooling2D
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop,Adam


def build_InceptionResNetV2_pretrain(img_shape=(128,128,3),num_classes=1):
    
    resnet = InceptionResNetV2(include_top=False,weights='imagenet')
    
    resnet.trainable = True
    
    x = resnet.get_layer('block8_10').output
    
    x = GlobalAveragePooling2D()(x)
    
    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    final_model = Model(inputs=[resnet.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return final_model


def build_InceptionResNetV2_ft(img_shape=(128,128,3),num_classes=1):
    resnet = InceptionResNetV2(include_top=False,weights='imagenet')
    
    resnet.trainable = True
    
    finetuning_layer = 250
    
    x = resnet.get_layer('block8_10').output
    
    x = GlobalAveragePooling2D()(x)
    
    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    final_model = Model(inputs=[resnet.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return final_model