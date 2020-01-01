
import sys
sys.path.append('../')

from codes.data.create_dataset_01 import load_gan_dataset, create_gan_images_list,create_gan_images_list_by_bach
from codes.data.train_model import generate_images


#from keras_contrib.layers.normalization import InstanceNormalization
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
import numpy as np
import os
import tensorflow as tf

from glob import glob

class Pix2Pix():
    def __init__(self,pix2pix=1):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.pix2pix=pix2pix
        
        # Configure data loader
        self.dataset_name = 'gen_breakhis'
        #self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                              img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        
        # vgg16 finetuning features
        #vgg16 = VGG16(include_top=False,input_shape=self.img_shape)
        if self.pix2pix == 0:
            vgg16 = VGG16(include_top=False,input_tensor=d0)
            for layer in vgg16.layers[:17]:
                layer.trainable = False

            for layer in vgg16.layers[17:]:
                layer.trainable = True  
            z1 = vgg16.output

            z1 = UpSampling2D(size=4)(z1)
            z1 = Conv2D(kernel_size=(3,3),filters=125,padding='same')(z1)
            z1 = LeakyReLU(alpha=0.2)(z1)
            z1 = BatchNormalization(momentum=0.8)(z1)

            z1 = UpSampling2D(size=4)(z1)
            z1 = Conv2D(kernel_size=(3,3),filters=64,padding='same',activation='relu')(z1)
            z1 = Dropout(0.2)(z1)
            z1 = BatchNormalization(momentum=0.8)(z1)

            z1 = UpSampling2D(size=2)(z1)    
            z1 = Conv2D(kernel_size=(1,1),filters=32,padding='same',activation='relu')(z1)
            z1 = BatchNormalization(momentum=0.8)(z1)


            z1 = Conv2D(kernel_size=(3,3),filters=16,padding='same',activation='relu')(z1)
            z1 = Dropout(0.2)(z1)
            z1 = BatchNormalization(momentum=0.8)(z1)

            z1 = Conv2D(kernel_size=(3,3),filters=8,padding='same',activation='relu')(z1)
            z1 = Dropout(0.2)(z1)
            z1 = BatchNormalization(momentum=0.8)(z1) 

            z1 = Conv2D(kernel_size=(3,3),filters=3,padding='same',activation='relu')(z1)
            z1 = Dropout(0.2)(z1)
            z1 = BatchNormalization(momentum=0.8)(z1)
        
        
        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)
        u7 = UpSampling2D(size=2)(u6)
        

        if self.pix2pix == 0:
            # combine vgg16 as context + u-net as normalize hist stain color
            u8 = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
            c  = Concatenate()([u8, z1])
        
            output_img =  Conv2D(kernel_size=(3,3),filters=3,padding='same', activation='tanh')(c)
        else:
            u8 = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
            output_img = u8
         
        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d # class 'tensorflow.python.framework.ops.Tensor'

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B]) #(?, 256, 256, 6)

        d1 = d_layer(combined_imgs, self.df, bn=False) # (?, 128, 128, 64)  
        d2 = d_layer(d1, self.df*2) # (?, 64, 64, 128)
        d3 = d_layer(d2, self.df*4) # (?, 32, 32, 256) 
        d4 = d_layer(d3, self.df*8) # (?, 16, 16, 512)
        
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4) # (?, 16, 16, 1)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=5,rate=0.9 ,generated_interval=1,save_model_checkpoint=20 , total_batch=200, cancer_type="B", maf_rate="40", directory="/tf/dataset/classes",transfer_type=0):
        
        start_time = datetime.datetime.now()
        total_images = (total_batch * batch_size) + (total_batch * batch_size) * (1 - rate) 
        #create dataset file
        if transfer_type == 1: #transfer image type is bach images
            create_gan_images_list_by_bach(1,cancer_type,maf_rate,directory,total_images)
        else: #transfer image type is breakhis images
            create_gan_images_list(epochs,cancer_type,maf_rate,directory,total_images)
        
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        

        for epoch in range(epochs):
            batch_i = 0
            csvfile = os.path.join(directory,"epoch","gan_epoch_{}.csv" . format(epoch))
            #csvfile = os.path.join(directory,"epoch","gan_epoch_0.csv" )
            dataset_train, dataset_test = load_gan_dataset(csvfile,total_images,batch_size,split = rate)
            dataset_train = dataset_train.batch(batch_size) 
            dataset_test = dataset_test.batch(3) 
            for imgs_A, imgs_B in dataset_train:
                

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)
                #print(fake_A.shape)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time

                batch_i = batch_i + 1
                
            
            if (epoch+1) % generated_interval == 0:
                self.sample_images(epoch,dataset_test,3)
                
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                    batch_i, total_batch,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))
                
                
            if (epoch+1) % save_model_checkpoint == 0:
                self.combined.save_weights("/tf/dataset/com_model_{}_{}_{}.h5" . format(maf_rate,cancer_type,self.pix2pix))
                self.generator.save_weights("/tf/dataset/gen_model_{}_{}_{}.h5". format(maf_rate,cancer_type,self.pix2pix))
                self.discriminator.save_weights("/tf/dataset/dis_model_{}_{}_{}.h5". format(maf_rate,cancer_type,self.pix2pix))                

        # serialize weights to HDF5
        self.combined.save_weights("/tf/dataset/com_model_{}_{}_{}.h5" . format(maf_rate,cancer_type,self.pix2pix))
        self.generator.save_weights("/tf/dataset/gen_model_{}_{}_{}.h5". format(maf_rate,cancer_type,self.pix2pix))
        self.discriminator.save_weights("/tf/dataset/dis_model_{}_{}_{}.h5". format(maf_rate,cancer_type,self.pix2pix))
        print("Model saved")
        
        # generating images
        print("Generating images with {} maginification" .format(maf_rate) )

        csvfile = os.path.join(directory,"epoch","gan_epoch_{}.csv" . format(0))
        dataset_train, dataset_test = load_gan_dataset(csvfile,total_images,batch_size,split = rate) 
       
        gen_model_file = "/tf/dataset/gen_model_{}_{}_{}.h5". format(maf_rate,cancer_type,self.pix2pix)
        generate_images(self.generator,gen_model_file,(256,256,3),dataset_train,maf_rate,cancer_type, is_check=False)

        print("Completed generating images with {} maginification" .format(maf_rate) )
        

    def sample_images(self, epoch,dataset_test,batch_size):
        
        

        stt = 0
        for imgs_A, imgs_B in dataset_test:
            
            if np.int(imgs_A.shape[0]) < batch_size:
                break
            
            fake_A = self.generator.predict(imgs_B)

            gen_imgs = [imgs_B[0], imgs_A[0], fake_A[0]]
            
            # Rescale images 0 - 1
            #gen_imgs = 0.5 * gen_imgs + 0.5
            gen_imgs = tf.cast(gen_imgs,tf.float32 ) * 0.5 + 0.5
            
            
            titles = ['Input Image', 'Ground Truth', 'Predicted Image']
            plt.figure(figsize=(15,15))
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.title(titles[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(gen_imgs[i])
                plt.axis('off')
            plt.show()
                
            
            '''
            os.makedirs('/tf/dataset/%s' % self.dataset_name, exist_ok=True)
            r, c = 3, 2
            fig, axs = plt.subplots(r, c)
            cnt = 0
            
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[i])
                    axs[i,j].axis('off')
                    cnt += 1

            fig.savefig("/tf/dataset/%s/%d_%d.png" % (self.dataset_name, epoch,  stt))
            plt.close()
            '''
            stt = stt + 1
            
            if stt > 0:
                break
            
           
