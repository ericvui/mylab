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
    
    filename = tf.string_split([file],"\\").values[-1]
    
    filename_parts = tf.string_split([filename],"_")
    
    #get digit 0 or 1
    binary_label = tf.strings.substr(filename_parts.values[4],5,1)
    binary_label = tf.strings.to_number(binary_label)   
     
    multiple_label = tf.convert_to_tensor(0)
    patient_id = filename_parts.values[0]
    
    return binary_label, multiple_label, patient_id

def extract_breakhis_image_label(file):        
    
    filename = tf.string_split([file],"\\").values[-1]
    
    filename_parts = tf.string_split([filename],"_")
    
    #get digit 0 or 1
    binary_label = tf.strings.regex_full_match(filename_parts.values[1],"B")
    binary_label = tf.cast(binary_label,dtype=tf.float32)   
    
    
    #multi label and patient
    last_tring = filename_parts.values[2]
    multiple_label = tf.strings.substr(last_tring,0,1)
    
    patient_id = last_tring
    
    return binary_label, multiple_label, patient_id

def extract_bach_image_label(file):        
    
    filename = tf.string_split([file],"\\").values[-1]
        
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

    if DATASET_TYPE == "kaggle" or DATASET_TYPE == "breakhis":
        image = tf.image.decode_png(image)

    else: #DATASET_TYPE == "bach":
        image = tf.image.decode_image(image)    
    
    image = tf.cast(image, tf.float32)
    
    #image = (image - 127.5) / 127.5
   
    image = _normalize(image,2)

    image = tf.image.resize(image, RESIZE_IMG)
    #image = tf.image.resize_image_with_crop_or_pad(image,target_height=RESIZE_IMG[0],target_width=RESIZE_IMG[0])

    
    if DATASET_TYPE == "kaggle":
        label,multiple_label,_ = extract_kaggle_image_label(file_image)
    elif DATASET_TYPE == "breakhis":
        label,multiple_label,_ = extract_breakhis_image_label(file_image)
    elif DATASET_TYPE == "bach":
        label,multiple_label,_ = extract_bach_image_label(file_image)


    return image,label

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



def load_files(src_folder,split=0.7, filenamepattern="*.png",dataset_volume=7909,flag=0):
    
    list_dataset = []
    
    pattern_file = os.path.join(src_folder,filenamepattern) 

    dataset = tf.data.Dataset.list_files(file_pattern=pattern_file,shuffle=True)

    if flag == 0:
        train,test = split_train_test(dataset,dataset_volume,split)
        train = train.map(normalize_image)
        test = test.map(normalize_image)
        list_dataset.append(train)
        list_dataset.append(test)
    else:
        train, val, test = split_train_val_test(dataset,dataset_volume,split)
        train = train.map(normalize_image)
        test = test.map(normalize_image)
        val = val.map(normalize_image)    
        list_dataset.append(train)
        list_dataset.append(test)    
        list_dataset.append(val)    
        
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







