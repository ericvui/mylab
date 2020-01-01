# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
import os


RESIZE_IMG = [256,256]
DATASET_TYPE = "breakhis" # breakhis, bach, kaggle
base_file = "F:\\GraduateClass\\Thesis\\Dataset\\BreaKHis\\histology_slides\\breast_40"
cond4x_file = "F:\\GraduateClass\\Thesis\\Dataset\\BreaKHis\\histology_slides\\breast_100"

def split_train_val_test(dataset,fullsize,ratio):
    train_size = int(fullsize * ratio)
    test_size = int(fullsize * (1-ratio - 0.1))
    val_size =  int(fullsize * 0.1)


    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, val_dataset, test_dataset   

def split_train_test(dataset,fullsize,ratio):

    test_size = int(fullsize * (1-ratio))
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)

    return train_dataset,  test_dataset      


def select_true_label(image,label):
    return tf.equal(label,1.0)


#format file: xxx_xxx_xxx_xxx.<ext>
def extract_kaggle_image_label(file):        
    
    filename = tf.string_split([file],"/").values[-1]
    
    filename_parts = tf.string_split([filename],"_")
    
    #get digit 0 or 1
    binary_label = tf.strings.substr(filename_parts.values[4],5,1)
    binary_label = tf.strings.to_number(binary_label)   
     
    multiple_label = tf.convert_to_tensor(0)
    patient_id = filename_parts.values[0]
    
    return binary_label, multiple_label, patient_id

def extract_breakhis_image_label(file):        
    
    filename = tf.string_split([file],"/").values[-1]
    
    filename_parts = tf.string_split([filename],"_")
    
    #get digit 0 or 1
    binary_label = tf.strings.regex_full_match(filename_parts.values[1],"B")
    binary_label = tf.cast(binary_label,dtype=tf.float32)   
    
    #multi label and patient
    last_tring = filename_parts.values[3]
    multiple_label = tf.strings.substr(last_tring,0,1)
    
    att_arr =  tf.string_split([last_tring],"-")
    size = tf.strings.to_number(att_arr.values[3])
        
    return binary_label, size

def extract_bach_image_label(file):        
    
    filename = tf.string_split([file],"/").values[-1]
        
    #get digit 0 or 1
    multiple_label = tf.strings.substr(filename,0,2)
    binary_label = tf.strings.regex_full_match(multiple_label,"i") 
    binary_label = tf.cast(binary_label,dtype=tf.uint8)
  
    patient_id = tf.constant(0)
    
    return binary_label, multiple_label, patient_id

def _normalize(image,normalize_type=1):
    #normalize_type = 1: global , range = [-1,1]
    #normalize_type = 2: per channel, range = [0,1]
    #
    new_img = image
    if normalize_type == 1:
        new_img = new_img - tf.reduce_mean(new_img)
        new_img = new_img / K.std(new_img)
    else:
        new_img = new_img - tf.reduce_mean(new_img,axis=(1,2),keepdims=True) #new_img.mean(axis=(1, 2), keepdims=True)
        new_img = new_img / K.std(new_img,axis=(1,2),keepdims=True) #new_img.std(axis=(1, 2), keepdims=True)
        
    return new_img

def normalize_image(file_image):
    
    image = tf.io.read_file(file_image)

    image = tf.image.decode_png(image)  
    
    image = tf.cast(image, tf.float32)
     
    image = _normalize(image,2)

    image = tf.image.resize(image, RESIZE_IMG)
    
    label,size = extract_breakhis_image_label(file_image)

    return image,label ,size

def get_image(file_image):
    
    image = tf.io.read_file(file_image)

    if DATASET_TYPE == "kaggle" or DATASET_TYPE == "breakhis":
        image = tf.image.decode_png(image)

    else: #DATASET_TYPE == "bach":
        image = tf.image.decode_image(image)    
    

    image = tf.image.resize(image, RESIZE_IMG)

    image = tf.cast(image, tf.float32)
    
    image = (image - 127.5) / 127.5
    
    return image

def get_label(file_image):
    
    if DATASET_TYPE == "kaggle":
        label,multiple_label,_ = extract_kaggle_image_label(file_image)
    elif DATASET_TYPE == "breakhis":
        label,multiple_label,_ = extract_breakhis_image_label(file_image)
    elif DATASET_TYPE == "bach":
        label,multiple_label,_ = extract_bach_image_label(file_image)

    return label

def load_files_for_fit(src_folder,split=0.7, filenamepattern="*.png",dataset_volume=0,flag=0):
    
    list_dataset = []
    
    pattern_file = os.path.join(src_folder,filenamepattern) 

    dataset = tf.data.Dataset.list_files(file_pattern=pattern_file,shuffle=True)

    if flag == 0:
        train,test = split_train_test(dataset,dataset_volume,split)
        train_x = train.map(normalize_image)
        train_y = train.map(get_label)
        test_x = train.map(normalize_image)
        test_y = train.map(get_label)
        list_dataset.append({'X':train_x, 'Y': train_y})
        list_dataset.append({'X':test_x, 'Y': test_y})
    else:
        train, val, test = split_train_val_test(dataset,dataset_volume,split)
        train_x = train.map(normalize_image)
        train_y = train.map(get_label)
        test_x = train.map(normalize_image)
        test_y = train.map(get_label)
        val_x = train.map(normalize_image)
        val_y = train.map(get_label)   
        list_dataset.append({'X':train_x, 'Y': train_y})
        list_dataset.append({'X':test_x, 'Y': test_y})   
        list_dataset.append({'X':val_x, 'Y': val_y})   
        
    return list_dataset



def load_files(src_folder,split=0.7, filenamepattern="*.png"):
    
    file40 = os.path.join(src_folder , "40",filenamepattern)
    file100 = os.path.join(src_folder ,"100",filenamepattern)
    file200 = os.path.join(src_folder ,"200",filenamepattern)
    file400 = os.path.join(src_folder ,"400",filenamepattern)
    
    list_dataset = []
    
    dataset40 = tf.data.Dataset.list_files(file_pattern=file40,shuffle=True)
    dataset100 = tf.data.Dataset.list_files(file_pattern=file100,shuffle=True)
    dataset200 = tf.data.Dataset.list_files(file_pattern=file200,shuffle=True)
    dataset400 = tf.data.Dataset.list_files(file_pattern=file400,shuffle=True) 
    
    #if flag == 0:
    train1,test1 = split_train_test(dataset40,1995,split)
    train2,test2 = split_train_test(dataset100,2081,split)
    train3,test3 = split_train_test(dataset200,2013,split)
    train4,test4 = split_train_test(dataset400,1820,split)
    train_tmp1 = train1.concatenate(train2) 
    train_tmp2 = train_tmp1.concatenate(train3)
    train = train_tmp2.concatenate(train4)
    train = train.shuffle(7909)
    
    test_tmp1 = test1.concatenate(test2) 
    test_tmp2 = test_tmp1.concatenate(test3)
    test = test_tmp2.concatenate(test4)
    
    train = train.map(normalize_image)
    test = test.map(normalize_image)
    list_dataset.append(train)
    list_dataset.append(test)
    
    return list_dataset

def normalize_image_without_label(f1,f2):
    
    #original image
    image_1 = tf.io.read_file(f1)    
    image_1 = tf.image.decode_png(image_1)
    image_1 = tf.cast(image_1, tf.float32)
    image_A = _normalize(image_1,2)
    image_A = tf.image.resize(image_A, RESIZE_IMG)
    
    #conditional image
    image_2 = tf.io.read_file(f2)
    image_2 = tf.image.decode_png(image_2)
    image_2 = tf.cast(image_2, tf.float32)
    image_B = _normalize(image_2,2)
    image_B = tf.image.resize(image_B, RESIZE_IMG)
        
    return image_A,image_B

def load_files_for_gan(src_folder,dataset_volume,split=0.8):
    
    dataset = tf.data.experimental.CsvDataset(src_folder,[tf.string, tf.string])

    train,  test = split_train_test(dataset,dataset_volume,split)
    
    train = train.map(normalize_image_without_label)

    test = test.map(normalize_image_without_label)


    return train,test







