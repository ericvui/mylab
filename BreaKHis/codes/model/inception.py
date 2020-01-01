from tensorflow.keras.applications import InceptionV3,VGG16, Xception
from tensorflow.keras.layers import Flatten,Dense,Dropout,GlobalAveragePooling2D,LeakyReLU,ELU ,ReLU, Concatenate
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam

def build_Inception_pretrain(img_shape=(128,128,3),num_classes=1, lr=0.001):
    
    base_model = InceptionV3(include_top=False, input_shape = img_shape,weights='imagenet')
    
    base_model.trainable = False
        
    x = Flatten()(base_model.output)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = ReLU()(x)
        
    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    final_model = Model(inputs=[base_model.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return final_model

def build_Xception_pretrain(img_shape=(128,128,3),num_classes=1, lr=0.001):
    
    base_model = Xception(include_top=False, input_shape = img_shape,weights='imagenet')
    
    base_model.trainable = False
    
    x = base_model.get_layer('avg_pool').output
        
    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    final_model = Model(inputs=[base_model.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    final_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return final_model

def build_Xception_FT(img_shape=(128,128),num_classes=1):
    
    base_model = Xception(include_top=False, input_shape = img_shape,weights='imagenet')
    
    base_model.trainable = True
    
    fine_tune_at = 130
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False   
        
    x = Flatten()(base_model.output)
    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = ReLU()(x)

    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    finetuning = Model(inputs=[base_model.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    finetuning.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return finetuning


def build_combine_VGG_Xception_FT(img_shape=(256,256,3),num_classes=1):
    
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=img_shape)

    for layer in vgg.layers[:17]:
        layer.trainable = False

    for layer in vgg.layers[17:]:
        layer.trainable = True
    
    inception = Xception(include_top=False, input_shape = img_shape)
    
    for layer in inception.layers[:128]:
        layer.trainable = False

    for layer in inception.layers[128:]:
        layer.trainable = True   
        
    x1 = Flatten()(vgg.output)
    x2 = Flatten()(inception.output)
    x  = Concatenate()([x1,x2])

    x = Dense(4096)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = ReLU()(x)

    outputs = Dense(num_classes
                    ,kernel_initializer = "he_normal"
                    , activation='sigmoid')(x)  
    finetuning = Model(inputs=[vgg.input,inception.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    finetuning.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    return finetuning
        
        
    