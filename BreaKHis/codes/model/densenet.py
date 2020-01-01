from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Flatten,Dense,Dropout,GlobalAveragePooling2D
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop,Adam


def build_DenseNet(img_shape=(256,256,3),num_classes=1, lr=0.001):
    
    resnet = DenseNet121(include_top=False, input_shape = img_shape)
    
    for layer in resnet.layers:
        layer.trainable = False
        
    x = resnet.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)            
    my_model = Model(inputs=[resnet.input], outputs=[outputs])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    my_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    #my_model.summary()
    
    return my_model
