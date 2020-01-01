from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten,Dense,Dropout,GlobalAveragePooling2D
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop,Adam


def build_MobileNet_finetuning(img_shape=(256,256,3),num_classes=1):
    
    base_model = MobileNetV2(include_top=False, input_shape = img_shape,weights='imagenet')

    base_model.trainable = True
    
    fine_tune_at = 100
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False    
    
    x = model.output
    
    x = GlobalAveragePooling2D()(x)
    
    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    final_model = Model(inputs=[base_model.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return final_model

def build_MobileNet_pretrain(img_shape=(256,256,3),num_classes=1):
    
    base_model = MobileNetV2(include_top=False, input_shape = img_shape,weights='imagenet')

    base_model.trainable = False
    
    x = base_model.output
    
    x = GlobalAveragePooling2D()(x)
    
    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    final_model = Model(inputs=[base_model.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return final_model