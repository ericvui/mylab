# Import necessary components to build LeNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop,Adam


def alexnet_model(img_shape=(256, 256, 3), num_classes=1, l2_reg=0.):

    model = Sequential()
    
    model.add(Conv2D(filters=96,
                                  kernel_size=(11, 11),
                                  strides=4,
                                  input_shape=img_shape,
                                  activation="relu",
                                  padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2)))

    # layer 2 - "256 kernels of size 5 x 5 x 48"
    model.add(Conv2D(filters=256,
                                  kernel_size=(5, 5),
                                  activation="relu",
                                  padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2)))

    # layer 3 - "384 kernels of size 3 x 3 x 256"
    model.add(Conv2D(filters=384,
                                  kernel_size=(3, 3),
                                  activation="relu",
                                  padding="same"))
    # layer 4 - "384 kernels of size 3 x 3 x 192"
    model.add(Conv2D(filters=384,
                                  kernel_size=(3, 3),
                                  activation="relu",
                                  padding="same"))
    # layer 5 - "256 kernels of size 3 x 3 x 192"
    model.add(Conv2D(filters=256,
                                  kernel_size=(3, 3),
                                  activation="relu",
                                  padding="same"))
    model.add(MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2)))

    # flatten before feeding into FC layers
    model.add(Flatten())

    # fully connected layers
    # "The fully-connected layers have 4096 neurons each."
    # "We use dropout in the first two fully-connected layers..."
    model.add(Dense(units=4096))  # layer 6
    model.add(Dropout(0.5))
    model.add(Dense(units=4096))  # layer 7
    model.add(Dropout(0.5))
    model.add(Dense(units=num_classes))  # layer 8

    # output layer is softmax
    model.add(Activation('sigmoid'))
        
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    
    model.compile(loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

    return model