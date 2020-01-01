# -*- coding: utf-8 -*-

import os
import sys
sys.path.append('../')
import pandas as pd
import numpy as np

#vote=2 ==> majority voting
#vote=1 ==> at least there is an accuracy vote ==> correct
#vote=3 ==> majority voting and if equal together, right accuracy will be voted
def calculate_accuracy(type_images,full_images,size_images,patch_acc,patch_w_acc,vote='method_A'):
    M_num, M_acc = 0,0
    B_num, B_acc = 0,0
    s40_num, s40_acc = 0,0
    s100_num, s100_acc = 0,0
    s200_num, s200_acc = 0,0
    s400_num, s400_acc = 0,0
    for i in range(len(full_images)):
        acc = patch_acc[i]
        w_acc = patch_w_acc[i]
        pred_acc, pred_w_acc = 0,0
        if vote=='method_A':
            if patch_acc[i] > patch_w_acc[i]:
                pred_acc = 1
            else:
                pred_acc = 0
        elif vote=='method_B':
            if patch_acc[i] >= patch_w_acc[i]:
                pred_acc = 1
            else:
                pred_acc = 0
        elif vote=='method_C':
            if patch_acc[i] > 0:
                pred_acc = 1
            else:
                pred_acc = 0    
        else:
            print('Select the voting method to calculate accuracy')
            
        
        if size_images[i] == '40' :
            s40_num += 1
            s40_acc += pred_acc
        elif size_images[i] == '100' :
            s100_num += 1
            s100_acc += pred_acc
        elif size_images[i] == '200' :
            s200_num += 1
            s200_acc += pred_acc
        elif size_images[i] == '400' :
            s400_num += 1
            s400_acc += pred_acc    
            
        if type_images[i] == 'M':
            M_num +=1
            M_acc +=pred_acc
        else:
            B_num +=1
            B_acc +=pred_acc
        

    return (M_acc/M_num),(B_acc/B_num),len(full_images),(M_acc+B_acc)/len(full_images),(s40_acc/s40_num),(s100_acc/s100_num),(s200_acc/s200_num), (s400_acc/s400_num)



# ignore maginification rate
def get_calculate_accuracy(csv_file,epoches=5,voting='method_A'):

    result_dataset = pd.read_csv(csv_file,names = ['epoch','image','true_label','prob_predict','accuracy'])
    total_size_40,total_size_100,total_size_200, total_size_400 = [],[],[],[]
    total_img = []

    for epo in range(epoches):
        epoch_ds = result_dataset.loc[result_dataset['epoch'] == epo]
        epoch_ds = epoch_ds.reset_index()
        patch_images = []
        full_images = []
        patch_acc,patch_w_acc = [],[]
        type_images = []
        size_images = []
        for i in range(len(epoch_ds)):
            filename = epoch_ds['image'][i].split('/')[-1].split('.')[0]
            acc = epoch_ds['accuracy'][i]
            typ = filename.split('_')[1]
            size = filename.split('_')[2].split('-')[3]
            idx =  filename.split('_')[2].split('-')[5]
            instance = filename.split('_')[2][:-2]
            patch_images.append({'type':typ, 'size':size, 'instance' : instance,'patch' : idx , 'acc' : acc })
            if instance not in full_images:
                full_images.append(instance)
                if acc == 1:
                    patch_acc.append(acc)
                    patch_w_acc.append(0)
                else:
                    patch_acc.append(0)
                    patch_w_acc.append(1)                
                type_images.append(typ)
                size_images.append(size)
            else:
                for i in range(len(full_images)):
                    if full_images[i] == instance:
                        if acc == 1:
                            patch_acc[i] += acc
                        else:
                            patch_w_acc[i] += 1
        M_acc_pct,B_acc_pct, total, all_accuracy, acc40, acc100, acc200, acc400  = calculate_accuracy(type_images,full_images,size_images,patch_acc,patch_w_acc,vote=voting)
        total_size_40.append(acc40)
        total_size_100.append(acc100)
        total_size_200.append(acc200)
        total_size_400.append(acc400)
        total_img.append(all_accuracy)
        print('Epoch:%d - Total %d images is %0.3f acuracy - Belgnin accuracy: %0.4f - Malgnin accuracy: %0.4f; 40 mag rate %0.3f; 100 mag rate: %0.3f; 200 mag rate:%0.3f; 400 mag rate: %0.3f' 
              % (epo,total,all_accuracy,B_acc_pct,M_acc_pct, acc40, acc100, acc200, acc400) )

    print('------------------VOTING: [%s] by file [%s]  -----------------' % (voting,csv_file) )
    print('TOTAL: %0.3f +/- %0.3f' % (np.mean(total_img) , np.std(total_img) )    )
    print('40 size : %0.3f +/- %0.3f' % (np.mean(total_size_40) , np.std(total_size_40) )   )
    print('100 size : %0.3f +/- %0.3f' % (np.mean(total_size_100) , np.std(total_size_100) )    )
    print('200 size : %0.3f +/- %0.3f' % (np.mean(total_size_200) , np.std(total_size_200) )  )
    print('400 size : %0.3f +/- %0.3f' % (np.mean(total_size_400) , np.std(total_size_400) )   )
    
    
#-------------------------------------------------------------------------------------------------------#

#vote=2 ==> majority voting
#vote=1 ==> at least there is an accuracy vote ==> correct
#vote=3 ==> majority voting and if equal together, right accuracy will be voted
def calculate_accuracy_BACH(type_images,full_images,patch_acc,patch_w_acc,vote='method_A'):
    M_num, M_acc = 0,0
    B_num, B_acc = 0,0
    for i in range(len(full_images)):
        acc = patch_acc[i]
        w_acc = patch_w_acc[i]
        pred_acc, pred_w_acc = 0,0
        if vote=='method_A':
            if patch_acc[i] > patch_w_acc[i]:
                pred_acc = 1
            else:
                pred_acc = 0
        elif vote=='method_B':
            if patch_acc[i] >= patch_w_acc[i]:
                pred_acc = 1
            else:
                pred_acc = 0
        elif vote=='method_C':
            if patch_acc[i] > 0:
                pred_acc = 1
            else:
                pred_acc = 0    
        else:
            print('Select the voting method to calculate accuracy')
            
               
        if type_images[i] == 'M':
            M_num +=1
            M_acc +=pred_acc
        else:
            B_num +=1
            B_acc +=pred_acc
        

    return (M_acc/M_num),(B_acc/B_num),len(full_images),(M_acc+B_acc)/len(full_images)



# ignore maginification rate
def get_calculate_accuracy_BACH(csv_file,epoches=5,voting='method_A'):

    result_dataset = pd.read_csv(csv_file,names = ['epoch','image','true_label','prob_predict','accuracy'])
    total_M,total_B = [],[]
    total_img = []

    for epo in range(epoches):
        epoch_ds = result_dataset.loc[result_dataset['epoch'] == epo]
        epoch_ds = epoch_ds.reset_index()
        patch_images = []
        full_images = []
        patch_acc,patch_w_acc = [],[]
        type_images = []

        for i in range(len(epoch_ds)):
            filename = epoch_ds['image'][i].split('/')[-1].split('.')[0]
            acc = epoch_ds['accuracy'][i]
            if filename[0] == 'b' or filename[0] == 'n':
                typ = 'B'
            else:
                typ = 'M'
            idx =  filename.split('-')[1]
            instance = filename.split('-')[0]
            patch_images.append({'type':typ, 'instance' : instance,'patch' : idx , 'acc' : acc })
            if instance not in full_images:
                full_images.append(instance)
                if acc == 1:
                    patch_acc.append(acc)
                    patch_w_acc.append(0)
                else:
                    patch_acc.append(0)
                    patch_w_acc.append(1)                
                type_images.append(typ)

            else:
                for i in range(len(full_images)):
                    if full_images[i] == instance:
                        if acc == 1:
                            patch_acc[i] += acc
                        else:
                            patch_w_acc[i] += 1
        M_acc_pct,B_acc_pct, total, all_accuracy = calculate_accuracy_BACH(type_images,full_images,patch_acc,patch_w_acc,vote=voting)

        total_M.append(M_acc_pct)
        total_B.append(B_acc_pct)
        total_img.append(all_accuracy)
        
        print('Epoch:%d - Total %d images is %0.3f acuracy - Belgnin accuracy: %0.3f - Malgnin accuracy: %0.3f' 
              % (epo,total,all_accuracy,B_acc_pct,M_acc_pct) )

    print('------------------VOTING: [%s] by file [%s]  -----------------' % (voting,csv_file) )
    print('TOTAL: %0.3f +/- %0.3f' % (np.mean(total_img) , np.std(total_img) )    )
    print('M accuracy : %0.3f +/- %0.3f' % (np.mean(total_M) , np.std(total_M) )   )
    print('B accuracy : %0.3f +/- %0.3f' % (np.mean(total_B) , np.std(total_B) )    )



