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
from codes.data.create_dataset_01 import create_dataset,load_dataset,generate_train_test_data_split,generate_train_test_data_cv

from codes.data.create_dataset_01 import load_BACH_dataset,generate_train_test_data_split_for_BACH



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

def predict(model,test,epoch,interval_print=50,output_file="/tf/dataset/log_test.csv"): 
   
    test = test.batch(1)
    total_accuracy,B_accuracy,M_accuracy  = 0,0,0
    size40 = 0
    size40_acc,size40_pred = 0,0
    size100_acc,size100_pred = 0,0
    size100 = 0
    size200_acc,size200_pred = 0,0
    size200 = 0
    size400_acc,size400_pred = 0,0
    size400 = 0
    i = 0
    img_type = '' 
    result = []
    log_file = open(output_file,"a")
    for tst_image, tst_labels, tst_size,tst_fn in test:
        i = i + 1
        y_pred_prob = model.predict_on_batch(tst_image)
        
        if np.float(y_pred_prob[0]) <= 0.5:
            pred_label = 0
        else:
            pred_label = 1
        
        
        true_label = np.int(tst_labels)
        accuracy = int(pred_label == true_label)
        total_accuracy = total_accuracy + accuracy
        
        #print('[Predicted prob = %0.3f] [True label: %d] [Predicted label=%d]' % (y_pred_prob,true_label,pred_label))
        
        if true_label == 1:
            B_accuracy += accuracy
            img_type = 'B'
        else:
            M_accuracy += accuracy
            img_type = 'M'
            
        if np.int(tst_size) == 40:
            size40 = size40 + 1
            if accuracy == 1:
                size40_pred = size40_pred + 1
        elif np.int(tst_size) == 100:
            size100 = size100 + 1
            if accuracy == 1:
                size100_pred = size100_pred + 1
        elif np.int(tst_size) == 200:
            size200 = size200 + 1
            if accuracy == 1:
                size200_pred = size200_pred + 1
        elif np.int(tst_size) == 400:
            size400 = size400 + 1   
            if accuracy == 1:
                size400_pred = size400_pred + 1
            
        if i % interval_print == 0:
            print("[TESTING: %d] [accuracy: %0.4f] [40: %d/%d][100: %d/%d][200: %d/%d][400: %d/%d]"
                  % (i,total_accuracy/i , size40_pred, size40, size100_pred, size100, size200_pred, size200, size400_pred, size400))

        log_file.write("{},{},{},{},{}\n".format(epoch,tst_fn,true_label,y_pred_prob,accuracy))    
            
    if size40 > 0:
        size40_acc =size40_pred /size40
    else:
        size40_acc = -1

    if size100 > 0:
        size100_acc =size100_pred /size100
    else:
        size100_acc = -1

    if size200 > 0:
        size200_acc =size200_pred /size200
    else:
        size200_acc = -1

    if size400 > 0:
        size400_acc =size400_pred /size400
    else:
        size400_acc = -1

    #log_file.write("{},{}\n".format('ENDED TEST BATCH','ENDED TEST BATCH')) 
        
    return (total_accuracy/i) , size40_acc, size100_acc, size200_acc, size400_acc

def _predict(model,test,epoch,interval_print=50):
   
    test = test.batch(1)

    log_file = open("/tf/dataset/log_test.csv","a")
    for tst_image, tst_labels, tst_size,tst_fn in test:

        
        y_pred_value = model.predict_on_batch(tst_image)
        
        print(tst_fn,tst_labels,y_pred_value)
        
        log_file.write("{},{},{}\n".format(epoch,tst_fn,y_pred_value))
        
    return 0 , 0, 0, 0, 0


def train_and_test_model(model, directory,ratio=0.7, dataset_volume = 7909, epochs=5,batch_size=16, print_interval=10 , output_filename="C:\\model.h5", fake_volumn = 1456):
        
    start_time = datetime.datetime.now()

    t_gl_losses = []
    v_gl_losses = []
    train_vol=dataset_volume * ratio + fake_volumn
    total_batch = train_vol//batch_size
    test_accuracy = 0
    #log_file = open("/tf/dataset/log_train.csv","a")
    for epoch in range(epochs):
        batch_i = 0     
        t_losses = []
        datasets = create_dataset(directory,batch_size,ratio,fake_volumn>0)
        dataset_train, dataset_test = datasets[0],datasets[1]
        dataset_train = dataset_train.batch(batch_size)
        B_cnt,M_cnt = 0,0
        B_acc,M_acc = 0,0
        cancer_type = ""
        
        for x_train,y_train,_,filename in dataset_train:
            if y_train.shape[0] < batch_size:
                break
            t_loss = model.train_on_batch(x_train, y_train)
            t_losses.append(t_loss)
            '''
            if int(y_train) == 1:
                B_cnt += 1
                cancer_type = "B"
                B_acc += t_loss[1]
            else:
                M_cnt +=1
                cancer_type = "M"
                M_acc += t_loss[1]
            
            '''
            
            elapsed_time = datetime.datetime.now() - start_time
            
            if batch_i % print_interval == 0:
                
                print ("[Epoch %d/%d] [Batch %d/%d] [T loss: %0.4f, acc: %3d%%] time: %s" % (epoch, epochs,
                                                    batch_i, total_batch,
                                                    t_loss[0], 100 * t_loss[1],
                                                    elapsed_time))
                
                #print ("[Epoch %d/%d] [%d B images - accuracy=%0.3f ] [%d M images - accuracy = %0.3f ] [Current: %s; acc=%0.3f, loss=%0.3f]" % (epoch, epochs,B_cnt,B_acc/(B_cnt+1) , M_cnt,M_acc/(M_cnt+1) , cancer_type,100 * t_loss[1], t_loss[0]))
                #print(filename)
                
            #log_file.write("{},{},{}\n".format(epoch,filename,t_loss[1]))
            batch_i = batch_i + 1  
            
        #cal mean loss, acc per training epoches
        tmp_losses = [v[0] for i,v in enumerate(t_losses)]
        tmp_acc = [v[1] for i,v in enumerate(t_losses)]
        #t_gl_losses.append({'loss' : np.mean(tmp_losses), 'acc' : np.mean(tmp_acc) , 'b_acc' : B_acc/B_cnt , 'm_acc' :M_acc/M_cnt })
        t_gl_losses.append({'loss' : np.mean(tmp_losses), 'acc' : np.mean(tmp_acc)  })

        #run test per epoch 
        '''
        print("---------------------------------------------------------------")
        print("Train size at epoch [%d/%d]: B %d; M %d" % (epoch,epochs,B_cnt, M_cnt))
        print("Avg Loss: %0.4f - Avg Accuracy: %0.4f" % (np.mean(tmp_losses),np.mean(tmp_acc)) ) 
        print("B accuracy: %0.4f - M accuracy: %0.4f" % (B_acc/B_cnt,M_acc/M_cnt) ) 
        print("---------------------------------------------------------------")
        '''
        
        avg_test_tmp_acc,s1,s2,s3,s4 = predict(model,dataset_test,epoch,interval_print=50)

        v_gl_losses.append({'loss' : 0, 'acc' : avg_test_tmp_acc, 'acc_40':s1,'acc_100':s2,'acc_200':s3 ,'acc_400':s4})

        
    model.save_weights(output_filename)
    print("Model saved")
    
    tmp_acc =  [v['acc'] for i,v in enumerate(v_gl_losses)]
    tmp_s100 = [v['acc_100'] for i,v in enumerate(v_gl_losses)]
    tmp_s200 = [v['acc_200'] for i,v in enumerate(v_gl_losses)]
    tmp_s40 =  [v['acc_40'] for i,v in enumerate(v_gl_losses)]
    tmp_s400 = [v['acc_400'] for i,v in enumerate(v_gl_losses)]
    print("40x:{} +/- {}" .format( round(np.average(tmp_s40)*100,3),round(np.std(tmp_s40)*100),3))
    print("100x:{} +/- {}" .format(round(np.average(tmp_s100)*100,3),round(np.std(tmp_s100)*100),3))
    print("200x:{} +/- {}" .format(round(np.average(tmp_s200)*100,3),round(np.std(tmp_s200)*100),3))
    print("400x:{} +/- {}" .format(round(np.average(tmp_s400)*100,3),round(np.std(tmp_s400)*100),3))
    print("Average:{} +/- {}" .format(round(np.average(tmp_acc)*100,3),round(np.std(tmp_acc)*100),3))    
    
    return t_gl_losses,v_gl_losses  

def train_and_test_model_by_load_data(model, directory,ratio=0.7, dataset_volume = 7909, epochs=5,batch_size=16, print_interval=20 , output_filename="C:\\model.h5", fake_amount = 0, output_predict_file="/tf/dataset/log_test.csv", mode='split'):

    #delete output prediction file
    if os.path.exists(output_predict_file):
        print('Remove a %s file because of existed' % (output_predict_file) )
        os.remove(output_predict_file)
    
    start_time = datetime.datetime.now()
    print('Preparing dataset....')
    
    include_fake = fake_amount>0
    if mode == 'split':
        generate_train_test_data_split(directory,epochs,"_patch_4",include_fake = include_fake)
    elif mode == 'cv': 
        generate_train_test_data_cv(directory,epochs,"_patch_4",include_fake = include_fake)
    else:
        print('Default mode=split ' )
        generate_train_test_data_split(directory,epochs,"_patch_4",include_fake = include_fake)
        
    t_gl_losses = []
    v_gl_losses = []
    train_vol=dataset_volume + fake_amount
    total_batch = train_vol//batch_size
    test_accuracy = 0
    print('Training model....')
    for epoch in range(epochs):
        batch_i = 0     
        t_losses = []
        dataset_train, dataset_test = load_dataset(directory,epoch)
        dataset_train = dataset_train.batch(batch_size)

        for x_train,y_train,_,filename in dataset_train:
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
        avg_test_tmp_acc,s1,s2,s3,s4 = predict(model,dataset_test,epoch,100,output_predict_file)

        v_gl_losses.append({'loss' : 0, 'acc' : avg_test_tmp_acc, 's40':s1,'s100':s2,'s200':s3 ,'s400':s4})

        
    model.save_weights(output_filename)
    print("Model saved")
    
    tmp_acc =  [v['acc'] for i,v in enumerate(v_gl_losses)]
    tmp_s100 = [v['s100'] for i,v in enumerate(v_gl_losses)]
    tmp_s200 = [v['s200'] for i,v in enumerate(v_gl_losses)]
    tmp_s40 =  [v['s40'] for i,v in enumerate(v_gl_losses)]
    tmp_s400 = [v['s400'] for i,v in enumerate(v_gl_losses)]
    
    print("40x:{} +/- {}" .format( round(np.average(tmp_s40)*100,3),round(np.std(tmp_s40)*100),3))
    print("100x:{} +/- {}" .format(round(np.average(tmp_s100)*100,3),round(np.std(tmp_s100)*100),3))
    print("200x:{} +/- {}" .format(round(np.average(tmp_s200)*100,3),round(np.std(tmp_s200)*100),3))
    print("400x:{} +/- {}" .format(round(np.average(tmp_s400)*100,3),round(np.std(tmp_s400)*100),3))
    print("Average:{} +/- {}" .format(round(np.average(tmp_acc)*100,3),round(np.std(tmp_acc)*100),3)) 
    
    return t_gl_losses,v_gl_losses 

def generate_images(gen_model,weight_file,image_size,dataset, mag_rate,cancer_type,is_check):
    ## use the trained model to generate data
    
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    directory = os.path.join("/tf/dataset/gen_breakhis/","fake_{}".format(mag_rate))

    test_model = gen_model
    test_model.load_weights(weight_file)
    m,n,d = image_size[0],image_size[1],image_size[2]
    num = 1
    dataset = dataset.batch(1)
    for _,img_b in dataset:

        img_show = np.zeros((m,2*n,d))
        
        fake_A = test_model.predict(img_b)  * 127.5 + 127.5
        
        fake_A = tf.cast(np.array(fake_A).reshape(image_size), tf.uint8)
        
        fake_AA = fake_A.numpy()
        
        im_AA = Image.fromarray(fake_AA)
        
        im_AA.save(os.path.join(directory,"SOB_{}_F-0-F-{}-{}-2.png" . format(cancer_type,mag_rate, num)))
        
        print('{} num of generated images now!' . format(num))
        
        #img_show[:,:n,:] = img_bb
        
        #img_show[:,n:2*n,:] = fake_A
        
        #plt.imsave("C:\\images\\pathology\\SOB_B_F-F-40-%d.png" % num,fake_A)
        #plt.imsave("C:\\images\\pathology\\SOB_B_F-F-40-%d.png" % num,fake_A_reshape,format="png")
        #plt.imsave(os.path.join(directory,"SOB_{}_F-0-F-{}-{}-2.png" . format(cancer_type,mag_rate, num)),fake_A) 
        #plt.imsave(os.path.join(directory,"SOB_{}_F-0-F-{}-{}-1.png" . format(cancer_type,mag_rate, num)),img_bb) 
        #plt.imsave("C:\\images\\pathology\\test_%d.png" % num,img_bb)
        if is_check:
            img_bb = img_b * 127.5 + 127.5
            img_bb = tf.cast(np.array(img_bb).reshape(image_size), tf.uint8)    
            img_bb = img_bb.numpy()
            im_bb = Image.fromarray(img_bb)
            im_bb.save(os.path.join(directory,"SOB_{}_F-0-F-{}-{}-1.png" . format(cancer_type,mag_rate, num)) )                 
        num = num + 1

def train_and_test_model_by_load_data_for_BACH(model, directory,ratio=0.7, dataset_volume = 7909, epochs=5,batch_size=16, print_interval=20 , output_filename="C:\\model.h5", fake_amount = 0, output_predict_file="/tf/dataset/log_test.csv", mode='split'):

    #delete output prediction file
    if os.path.exists(output_predict_file):
        print('Remove a %s file because of existed' % (output_predict_file) )
        os.remove(output_predict_file)
    
    start_time = datetime.datetime.now()
    print('Preparing BACH dataset....')
    
    include_fake = fake_amount>0
    generate_train_test_data_split_for_BACH(directory,epochs,"_png",include_fake = include_fake)
        
    t_gl_losses = []
    v_gl_losses = []
    train_vol=dataset_volume + fake_amount
    total_batch = train_vol//batch_size
    test_accuracy = 0
    print('Training model....')
    for epoch in range(epochs):
        batch_i = 0     
        t_losses = []
        dataset_train, dataset_test = load_BACH_dataset(directory,epoch)
        dataset_train = dataset_train.batch(batch_size)

        for x_train,y_train,filename in dataset_train:
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
        avg_test_tmp_acc,B_acc,M_acc = predict_for_BACH(model,dataset_test,epoch,100,output_predict_file)

        v_gl_losses.append({'loss' : 0, 'acc' : avg_test_tmp_acc, 'M':M_acc,'B':B_acc})

        
    model.save_weights(output_filename)
    print("Model saved")
    
    tmp_acc =  [v['acc'] for i,v in enumerate(v_gl_losses)]
    tmp_M = [v['M'] for i,v in enumerate(v_gl_losses)]
    tmp_B = [v['B'] for i,v in enumerate(v_gl_losses)]

    print("B accuracy:{} +/- {}" .format(round(np.average(tmp_B)*100,3),round(np.std(tmp_B)*100),3))
    print("M accuracy:{} +/- {}" .format(round(np.average(tmp_M)*100,3),round(np.std(tmp_M)*100),3))
    print("Average:{} +/- {}" .format(round(np.average(tmp_acc)*100,3),round(np.std(tmp_acc)*100),3)) 
    
    return t_gl_losses,v_gl_losses 
def predict_for_BACH(model,test,epoch,interval_print=50,output_file="/tf/dataset/log_test.csv"): 
   
    test = test.batch(1)
    total_accuracy,B_accuracy,M_accuracy,B_total, M_total  = 0,0,0,0,0
    i = 0
    result = []
    log_file = open(output_file,"a")
    for tst_image, tst_labels, tst_fn in test:
        i = i + 1
        y_pred_prob = model.predict_on_batch(tst_image)
        
        if np.float(y_pred_prob[0]) <= 0.5:
            pred_label = 0
        else:
            pred_label = 1
        
        
        true_label = np.int(tst_labels)
        accuracy = int(pred_label == true_label)
        total_accuracy = total_accuracy + accuracy
                
        if true_label == 1:
            B_accuracy += accuracy
            B_total += 1
        else:
            M_accuracy += accuracy
            M_total += 1
            
            
        if i % interval_print == 0:
            print("[TESTING: %d] [accuracy: %0.4f] [B accuracy: %d] [M accuracy: %d]"
                  % (i,total_accuracy/i , B_accuracy,M_accuracy ))

        log_file.write("{},{},{},{},{}\n".format(epoch,tst_fn,true_label,y_pred_prob,accuracy))    
            
 
        
    return (total_accuracy/i),(B_accuracy/B_total),(M_accuracy/M_total)