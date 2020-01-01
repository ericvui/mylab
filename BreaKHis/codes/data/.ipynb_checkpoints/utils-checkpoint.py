# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Conv2D
import datetime

import sys
sys.path.append('../') 
from codes.data.load_data import load_files



#------------------------Config class------------------------------------------#
class cls_config(object):
    
    def __init__(self):
        self.KAGGLE_SIZE = 277524
        self.BREAKHIS_SIZE = 7909
        self.BACH_SIZE = 400
        self.BACTCH_SIZE = 50
        self.BREAKHIS_NUM_CLASS = 8
        self.BACH_NUM_CLASS = 4    
        self.EPOCHS = 3
        self.KAGGLE_IMAGE_SIZE = (50,50,3)
        self.BREAKHIS_IMAGE_SIZE = [460,700,3]
        self.BACH_IMAGE_SIZE = [2048,1536,3]

#------------------------Common functions--------------------------------------#

def extract_feature(tfdata_record):        
    
    features = {'height':tf.FixedLenFeature([],tf.int64)
               ,'weight':tf.FixedLenFeature([],tf.int64) 
               ,'deep':tf.FixedLenFeature([],tf.int64)
               ,'binary_label':tf.FixedLenFeature([],tf.int64)
               ,'multiple_label':tf.FixedLenFeature([],tf.int64)
               ,'patient_id':tf.FixedLenFeature([],tf.string)
               ,'image_raw':tf.FixedLenFeature([],tf.string)
    }
    
    sample = tf.parse_single_example(tfdata_record,features)
    
    height = tf.cast(sample["height"], dtype = tf.int32)
    weight = tf.cast(sample["weight"], dtype = tf.int32)
    deep = tf.cast(sample["deep"], dtype = tf.int32)    
    img_shape = tf.stack([-1,height, weight, deep])
    
    image = tf.decode_raw( sample["image_raw"],tf.uint8)
    #reconstructed_img = tf.reshape(image,img_shape)
    #image = tf.image.decode_png( sample["image_raw"])
    reconstructed_img = tf.cast(image, tf.float32)
    
    binary_label = tf.cast(sample["binary_label"], dtype = tf.float32)
    multiple_label = tf.cast(sample["multiple_label"], dtype = tf.int32)
    patient_id = tf.cast(sample["patient_id"], dtype = tf.string)

    return reconstructed_img,img_shape,binary_label,multiple_label,patient_id

def resize_image(image_size):
    pass


def normalize_image(image,normalize_type=1):
    #normalize_type = 1: global , range = [-1,1]
    #normalize_type = 2: per channel, range = [0,1]
    #
    new_img = image
    if normalize_type == 1:
        new_img = new_img - new_img.mean()
        new_img = new_img / new_img.std()
    else:
        new_img = new_img - new_img.mean(axis=(1, 2), keepdims=True)
        new_img = new_img / new_img.std(axis=(1, 2), keepdims=True)
        
    return new_img

def extract_kaggle_image_label(file):        
    
    filename = tf.string_split([file],"\\").values[-1]
    
    filename_parts = tf.string_split([filename],"_")
    
    #get digit 0 or 1
    binary_label = tf.strings.substr(filename_parts.values[4],5,1)
    binary_label = tf.strings.to_number(binary_label)   
     
    multiple_label = tf.convert_to_tensor(0)
    patient_id = filename_parts.values[0]
    
    return binary_label, multiple_label, patient_id


def extract_breakhis_image_label(file):
    #filename = tf.string_split([file],"\\").values[-1]
    pass

def extract_bach_image_label(file):
    pass

def evaluate_model(model,val_data,epoch):
    print ("------------------------Evaluate model-----------------------------")   
    batch_val = 0
    val_logs = []
    checkpoint_print = 10
    for val_images,val_labels in val_data:
        val_acc = model.test_on_batch(val_images,val_labels)
        batch_val = batch_val + 1
        val_logs.append(val_acc)
        if batch_val % checkpoint_print == 0:
            print("epoch number: {} batch number:{} ---- VAL acc = {}".format(epoch,batch_val,val_acc))
        
    return  val_logs


def predict_model(model,test,filename='test_logs'):
    print ("------------------------Predict model-----------------------------")   
    test_logs = []
    i = 0
    for tst_image, tst_labels in test:
        i = i + 1
        y_pred_prob = model.predict_on_batch(tst_image)
        y_pred = [1 if prob>0.5 else 0 for prob in y_pred_prob]
        correct_prediction  = tf.equal(y_pred,tst_labels) 
        correct_prediction  = tf.cast(correct_prediction ,tf.float32)
        accuracy = tf.reduce_mean(correct_prediction )
        print("[Batch number:{}] [True label: {} ] [Pred label: {}-->{}] acc = {}".format(i,tst_labels,y_pred_prob,y_pred,accuracy))
        test_logs.append(accuracy)
    print("=> avg test acc = {}".format(np.mean(test_logs)))
    with open(filename + '.pkl' , 'wb') as lossfile:
        pickle.dump(test_logs, lossfile)    

def predict_model_svm(model,test,filename='test_logs.pkl'):
    print ("------------------------Predict model-----------------------------")   
    test = test.batch(1)
    total_accuracy = 0
    i = 0
    result = []
    for tst_image, tst_labels in test:
        i = i + 1
        y_pred_value = model.predict_on_batch(tst_image)
        pred_label = int(y_pred_value > 0)
        true_label = np.int(tst_labels)
        accuracy = int(pred_label == true_label)
        total_accuracy = total_accuracy + accuracy
        print("[True label: {} ] [Predicted label: {} ] [accuracy {}%]". format(true_label,pred_label,accuracy*100))
        result.append({'true_lbl':true_label,'pred_score':y_pred_value,'pred_lbl':pred_label,'acc':accuracy})
        
    print('Average testing accuracy = {}' .format(100*total_accuracy/i))
    with open(filename , 'wb') as lossfile:
        pickle.dump(result, lossfile) 
        
def predict(model,test,interval_print=50):
   
    test = test.batch(1)
    total_accuracy = 0
    i = 0
    result = []
    for tst_image, tst_labels in test:
        i = i + 1
        y_pred_value = model.predict_on_batch(tst_image)
        pred_label = int(y_pred_value > 0)
        true_label = np.int(tst_labels)
        accuracy = int(pred_label == true_label)
        total_accuracy = total_accuracy + accuracy
        if i % interval_print == 0:
            print("[Total testing images: {} ] [accuracy {}%]". format(i,(total_accuracy/i) * 100))
        #result.append({'true_lbl':true_label,'pred_score':y_pred_value,'pred_lbl':pred_label,'acc':accuracy})
        
    return (total_accuracy/i) 


def plotting(log_train, log_val):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    losses = [t['loss'] for t in log_train]
    val_losses = [t['loss'] for t in log_val]

    acc = [t['acc'] for t in log_train]
    val_acc = [t['acc'] for t in log_val]
    
    
    fig, (axs1,axs2) = plt.subplots(1, 2, sharex=True)
    
    clear_output(wait=True)

    axs1.set_yscale('log')
    axs1.plot(range(len(losses)), losses, label='loss')
    axs1.plot(range(len(val_losses)), val_losses, label='val-loss')
    axs1.legend()

    axs2.plot(range(len(acc)), acc, label='accuracy')
    axs2.plot(range(len(val_acc)), val_acc, label='val-accuracy')
    axs2.legend()

    plt.show();
    
def train_model(model,dataset_train, dataset_val, epochs, print_interval=20 , total_batch=587,train_vol=1000, filename="C:\\model.h5"):
        
    start_time = datetime.datetime.now()

    t_gl_losses = []
    v_gl_losses = []


    for epoch in range(epochs):
        batch_i = 0     
        t_losses = []
        for x_train,y_train in dataset_train:
            t_loss = model.train_on_batch(x_train, y_train)
            t_losses.append(t_loss)

            elapsed_time = datetime.datetime.now() - start_time

            if batch_i % print_interval == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] [T loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                    batch_i, total_batch,
                                                    t_loss[0], 100 * t_loss[1],
                                                    elapsed_time))
            batch_i = batch_i + 1  
            
        #cal mean loss, acc per training sepoch
        tmp_losses = [v[0] for i,v in enumerate(t_losses)]
        tmp_acc = [v[1] for i,v in enumerate(t_losses)]
        t_gl_losses.append({'loss' : np.mean(tmp_losses), 'acc' : np.mean(tmp_acc) })

        #run validation per epoch and then calculate mean of loss, acc
        batch_i = 0
        v_losses = []
        for x_val, y_val in dataset_val:
            v_loss = model.test_on_batch(x_val, y_val)
            v_losses.append(v_loss)
            elapsed_time = datetime.datetime.now() - start_time
            if batch_i % print_interval == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] [E loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                    batch_i, 
                                                    total_batch,                                      
                                                    v_loss[0], 100 * v_loss[1],
                                                    elapsed_time))
            batch_i = batch_i + 1

        tmp_losses = [v[0] for i,v in enumerate(v_losses)]
        tmp_acc = [v[1] for i,v in enumerate(v_losses)]
        v_gl_losses.append({'loss' : np.mean(tmp_losses), 'acc' : np.mean(tmp_acc) })

        dataset_train = dataset_train.shuffle(train_vol)

        #save model per epoch
        model.save_weights(filename)

    
    print("Model saved")

    return t_gl_losses,v_gl_losses    

def train_and_test_model(model, input_ds_filename,ratio=0.7, img_file_type="*.png",dataset_volume=7909, epochs=5,batch_size=16, print_interval=20 , output_filename="C:\\model.h5"):
        
    start_time = datetime.datetime.now()

    t_gl_losses = []
    v_gl_losses = []
    train_vol=dataset_volume * ratio
    total_batch = train_vol//batch_size
    test_accuracy = 0

    for epoch in range(epochs):
        batch_i = 0     
        t_losses = []
        datasets = load_files(input_ds_filename,ratio,"*.png",dataset_volume,0)
        dataset_train, dataset_test = datasets[0],datasets[1]
        dataset_train = dataset_train.batch(batch_size)
        for x_train,y_train in dataset_train:
            t_loss = model.train_on_batch(x_train, y_train)
            t_losses.append(t_loss)

            elapsed_time = datetime.datetime.now() - start_time

            if batch_i % print_interval == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] [T loss: %f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                    batch_i, total_batch,
                                                    t_loss[0], 100 * t_loss[1],
                                                    elapsed_time))
            batch_i = batch_i + 1  
            
        #cal mean loss, acc per training epoches
        tmp_losses = [v[0] for i,v in enumerate(t_losses)]
        tmp_acc = [v[1] for i,v in enumerate(t_losses)]
        t_gl_losses.append({'loss' : np.mean(tmp_losses), 'acc' : np.mean(tmp_acc) })

        #run test per epoch 
        test_tmp_acc = predict(model,dataset_test,interval_print=50)

        v_gl_losses.append({'loss' : 0, 'acc' : test_tmp_acc })

        #if test_accuracy < test_tmp_acc:
            #save model per epoch
        #    model.save_weights(output_filename)
        #    print("Model saved")
    model.save_weights(output_filename)
    print("Model saved")
    return t_gl_losses,v_gl_losses  


