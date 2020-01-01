# -*- coding: utf-8 -*-

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Flatten,Dense,Dropout,LeakyReLU,ELU ,ReLU
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
import datetime
import numpy as np

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import Callback



class VGG:
    def __init__(self,rows=400, cols=400, channels=3,num_classes=1, pretrained=0,is_svm=0):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        if pretrained == 0:
            self.vgg = self.build_vgg16()            
        else:
            if is_svm == 0:
                self.vgg = self.build_vgg16_pretrained()
            else:
                self.vgg = self.build_vgg16_pretrained_svm()
            
            
        #self.checkpoint = ModelCheckpoint('C:\\VGG16_finetune.hdf5',
        #            monitor='val_acc',
        #            verbose=1,
        #            save_best_only=True,
        #            mode='max')
        
        #self.plot_training = plotTraining()
        
    #finetuning vgg16
    def build_vgg16(self):
        
        model = VGG16(include_top=False, weights='imagenet', input_shape=self.img_shape)
        
        for layer in model.layers[:17]:
            layer.trainable = False
        
        for layer in model.layers[17:]:
            layer.trainable = True            
        
        x = Flatten()(model.output)
        x = Dense(4096)(x)
        x = ReLU()(x)
        x = Dropout(0.2)(x)
        x = Dense(512)(x)
        x = ReLU()(x)
        '''
        outputs = Dense(self.num_classes
                        ,kernel_regularizer=regularizers.l2(0.01)   #0.01
                        ,kernel_initializer = "normal"
                        , activation='linear')(d3)  #(d3)  
        '''
        outputs = Dense(self.num_classes
                        ,kernel_initializer = "he_normal"
                        , activation='sigmoid')(x)  
        finetuning = Model(inputs=[model.input], outputs=[outputs])
        #adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay=1e-9)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        
        finetuning.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        return finetuning
    
    def build_vgg16_pretrained(self):
        '''
            img_shape = (224,224,3)
        '''
        model = VGG16(include_top=True, weights='imagenet', input_shape=self.img_shape) 
        
        x = model.get_layer('fc2').output
        
        outputs = Dense(self.num_classes
                        ,kernel_regularizer=regularizers.l2(0.01) 
                        ,kernel_initializer = "random_uniform"
                        , activation='softmax')(x)            
        pretrain = Model(inputs=[model.input], outputs=[outputs])
        
        pretrain.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return pretrain

    def build_vgg16_pretrained_svm(self,lr = 0.001):
        
        model = VGG16(include_top=True, weights='imagenet', input_shape=self.img_shape) 
        
        x = model.get_layer('fc2').output
        
        outputs = Dense(self.num_classes
                        ,kernel_regularizer=regularizers.l2(0.01) 
                        ,kernel_initializer = "normal"
                        ,activation = 'linear')(x)            
        pretrain = Model(inputs=[model.input], outputs=[outputs])
        
        rmsprop = RMSprop(lr=lr)
        
        pretrain.compile(optimizer=rmsprop, loss='hinge', metrics=['accuracy'])

        return pretrain    
    
    def train1(self,train_X,train_y, val_X, val_y, epochs, steps_per_epoch=5):
       
        self.vgg.fit(train_X, 
                      train_y, 
                      steps_per_epoch=steps_per_epoch,
                      epochs=epochs,
                      validation_data=(val_X, val_y),
                      callbacks=[self.plot_training, self.checkpoint],
                      verbose=1,
                      shuffle=True)
    
    def train(self,dataset_train, dataset_val, epochs, batch_size=5, print_interval=20 , total_batch=587,train_vol=1000,saved_file="c:\\file.h5"):
        
        start_time = datetime.datetime.now()

        t_gl_losses = []
        v_gl_losses = []
        
        for epoch in range(epochs):
            batch_i = 0     
            t_losses = []
            for x_train,y_train in dataset_train:
                t_loss = self.vgg.train_on_batch(x_train, y_train)
                t_losses.append(t_loss)
                elapsed_time = datetime.datetime.now() - start_time

                if batch_i % print_interval == 0:
                    print ("[Epoch %d/%d] [Batch %d/%d] [T loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                        batch_i, total_batch,
                                                        t_loss[0], 100 * t_loss[1],
                                                        elapsed_time))
                
                
                batch_i = batch_i + 1  
            #cal mean loss, acc per training epoch    
            t_gl_losses.append({'loss' : np.mean(t_losses[0]), 'acc' : np.mean(t_losses[1]) })
            
            #run validation per epoch and then calculate mean of loss, acc
            batch_i = 0
            v_losses = []
            for x_val, y_val in dataset_val:
                v_loss = self.vgg.test_on_batch(x_val, y_val)
                v_losses.append(v_loss)
                elapsed_time = datetime.datetime.now() - start_time
                if batch_i % print_interval == 0:
                    print ("[Epoch %d/%d] [Batch %d] [E loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                        batch_i, 
                                                        v_loss[0], 100 * v_loss[1],
                                                        elapsed_time))
                batch_i = batch_i + 1
            
            v_gl_losses.append({'loss' : np.mean(v_losses[0]), 'acc' : np.mean(v_losses[1]) })
            
            dataset_train = dataset_train.shuffle(train_vol)
            
            #save model per epoch
            self.vgg.save_weights(saved_file)
           
        self.vgg.save_weights(saved_file)
        print("Model saved")
            
        return t_gl_losses,v_gl_losses