# -*- coding: utf-8 -*-
'''
    Generate BreaKHis dataset label
'''
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import numpy as np

RESIZE_IMG = [256,256]
BUFFER_SIZE = 100000

def split_train_val_test(dataset,fullsize,ratio):
    train_size = int(fullsize * ratio)
    test_size = int(fullsize * (1-ratio - 0.1))
    val_size =  int(fullsize * 0.1)


    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, val_dataset, test_dataset   

def split_train_test(dataset,fullsize,batch_size,ratio):
    
    train_size = int(fullsize * ratio)
    if train_size % batch_size > 0:
        train_size = train_size - int(train_size%batch_size) 
    test_size = fullsize - train_size
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    return train_dataset,  test_dataset      


def _normalize(image,normalize_type=1):
    #normalize_type = 1: global , range = [-1,1]
    #normalize_type = 2: per channel, range = [0,1]
    #else: global, not depend on specified images
    new_img = image
    if normalize_type == 1:
        new_img = new_img - tf.reduce_mean(new_img)
        new_img = new_img / K.std(new_img)
    elif normalize_type == 2:
        new_img = new_img - tf.reduce_mean(new_img,axis=(1,2),keepdims=True) #new_img.mean(axis=(1, 2), keepdims=True)
        new_img = new_img / K.std(new_img,axis=(1,2),keepdims=True) #new_img.std(axis=(1, 2), keepdims=True)
    elif normalize_type == 3:
        new_img = (new_img - 127.5)/127.5
        # new_img = new_img / K.std(new_img,axis=(1,2),keepdims=True) #new_img.std(axis=(1, 2), keepdims=True)
    else:
        new_img = new_img / 255.0
    return new_img

def normalize_image(file_image, label, size):
    
    image = tf.io.read_file(file_image)

    image = tf.image.decode_png(image)  
    
    image = tf.cast(image, tf.float32)
     
    image = _normalize(image,2)

    image = tf.image.resize(image, RESIZE_IMG)
    
    return image,label,size,file_image

def create_dataset(directory,bs_size,split=0.7,include_fake=0):
        
    list_dataset = []
    
    dataset40 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_40.csv"),[tf.string, tf.int32,tf.int32])
    dataset100 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_100.csv"),[tf.string, tf.int32,tf.int32])
    dataset200 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_200.csv"),[tf.string, tf.int32,tf.int32])
    dataset400 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_400.csv"),[tf.string, tf.int32,tf.int32])
    
    if include_fake == 1:
    # get fake images per maginification
        fake_ds_40 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_Fake_B40.csv"),[tf.string, tf.int32,tf.int32])
        fake_ds_100 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_Fake_B100.csv"),[tf.string, tf.int32,tf.int32])
        fake_ds_200 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_Fake_B200.csv"),[tf.string, tf.int32,tf.int32])
        fake_ds_400 = tf.data.experimental.CsvDataset(os.path.join(directory,"data_file_Fake_B400.csv"),[tf.string, tf.int32,tf.int32])
        
        # add fake images per maginification to training dataset but unbalance again
        f_train_1 = fake_ds_40.concatenate(fake_ds_100)
        f_train_2 = f_train_1.concatenate(fake_ds_200)
        f_train_3 = f_train_1.concatenate(fake_ds_400)    
        f_train_3 = f_train_3.shuffle(5000)
    
    train1,test1 = split_train_test(dataset40,1995,bs_size,split)
    train2,test2 = split_train_test(dataset100,2081,bs_size,split)
    train3,test3 = split_train_test(dataset200,2013,bs_size,split)
    train4,test4 = split_train_test(dataset400,1820,bs_size,split)
    
    
    train_tmp1 = train1.concatenate(train2) 
    train_tmp2 = train_tmp1.concatenate(train3)
    train_tmp3 = train_tmp2.concatenate(train4)
    

    
    if include_fake == 1:
        train = f_train_3.concatenate(train_tmp3)
    else:
        train = train_tmp3
        
    train = train.shuffle(BUFFER_SIZE)    #7909 + fake volumns
    
    test_tmp1 = test1.concatenate(test2) 
    test_tmp2 = test_tmp1.concatenate(test3)
    test = test_tmp2.concatenate(test4)
    
    train = train.map(normalize_image)
    test = test.map(normalize_image)
    list_dataset.append(train)
    list_dataset.append(test)
    
    return list_dataset

def load_dataset(directory, epoch):
    fn_train_ep = "train_data_file_ep_{}.csv" . format(epoch)
    fn_test_ep = "test_data_file_ep_{}.csv" . format(epoch)
    
    train = tf.data.experimental.CsvDataset(os.path.join(directory,'epoch',fn_train_ep),[tf.string, tf.int32,tf.int32])

    test = tf.data.experimental.CsvDataset(os.path.join(directory,'epoch',fn_test_ep),[tf.string, tf.int32,tf.int32])

    train = train.shuffle(BUFFER_SIZE)    
    
    train = train.map(normalize_image)
    
    test = test.map(normalize_image)    

    return train, test

def extract_label(filename):
    parts = filename.split("_")
    if parts[1] == "B":
        label = 1
    else:
        label = 0
    sub_parts = parts[2].split("-")
    size = np.int(sub_parts[3])
    return label, size

def remove_all_training_files(directory):
    for path,_,files in os.walk(directory):
        for filename in files:
            os.remove(os.path.join(path,filename))

# Cross - validation
def generate_train_test_data_cv(directory, epoch,prefix,include_fake=False):
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold 

    import random
    
    remove_all_training_files(os.path.join(directory,'epoch'))
    

    #load 4 mag rates into list
    full_img_data = ['40','100','200','400']
    
    for img_size in full_img_data:
        #read full images
        list_img = []
        with open(os.path.join(directory,'data_file_{}.csv' . format(img_size)), "r") as f:
            lines = f.readlines()
        for ln in lines:
            line = ln.strip().split('/')[-1]
            list_img.append(line)
    
        #create 5 cross-validation
        cv = KFold(n_splits=epoch, random_state=2019)   
        i_ep = 0
        for train_index, test_index in cv.split(list_img):
            train_filename = "train_data_file_ep_{}.csv" . format(i_ep)
            test_filename = "test_data_file_ep_{}.csv" . format(i_ep)
            f_train = open(os.path.join(directory,'epoch',train_filename),'a')
            f_test = open(os.path.join(directory,'epoch',test_filename),'a')

            #convert images into 4 patch images
            #d_train, d_test = list_img[train_index],list_img[test_index] 
            for idx in train_index:
                x = list_img[idx]
                label, size = extract_label(x)
                fn_without_ext = x.split(".")[0]
                for i_patch in range(4):
                    fn_with_ext = "{}/{}-{}.png,{},{}" .format(os.path.join(directory,img_size+prefix) ,fn_without_ext,i_patch,label,size)
                    f_train.write(fn_with_ext + "\n")

            for idx in test_index:
                y = list_img[idx]
                fn_without_ext = y.split(".")[0]
                label, size = extract_label(y)
                for i_patch in range(4):
                    fn_with_ext = "{}/{}-{}.png,{},{}" .format(os.path.join(directory,img_size+prefix) ,fn_without_ext,i_patch,label,size)
                    f_test.write(fn_with_ext + "\n")

            #increase epoch
            i_ep += 1

    #add fake images into train dataset
    if include_fake == True: 
        for i_ep in range(epoch):
            train_filename = "train_data_file_ep_{}.csv" . format(i_ep)
            f_train = open(os.path.join(directory,'epoch',train_filename),'a')
            with open(os.path.join(directory,'data_file_fake.csv'), "r") as fake_file:
                lines = fake_file.readlines()
            for line in lines:
                f_train.write(line.strip() + "\n")
            

#split train vs test as 70/30
def generate_train_test_data_split(directory, epoch,prefix,include_fake=False):
    from sklearn.model_selection import train_test_split
    import random
    
    remove_all_training_files(os.path.join(directory,'epoch'))
    
    
    full_img_data = ['40','100','200','400']
   
    for i_ep in range(epoch):
        random_seed=random.randint(1,1001) + i_ep
        train_filename = "train_data_file_ep_{}.csv" . format(i_ep)
        test_filename = "test_data_file_ep_{}.csv" . format(i_ep)
        f_train = open(os.path.join(directory,'epoch',train_filename),'a')
        f_test = open(os.path.join(directory,'epoch',test_filename),'a')
        for img_size in full_img_data:
            
            #read full images
            list_img = []
            with open(os.path.join(directory,'data_file_{}.csv' . format(img_size)), "r") as f:
                lines = f.readlines()
            for ln in lines:
                line = ln.strip().split('/')[-1]
                list_img.append(line)
            
            #convert images into 4 patch images
            d_train, d_test = train_test_split(list_img,test_size=0.3, random_state=random_seed)
            for x in d_train:
                label, size = extract_label(x)
                fn_without_ext = x.split(".")[0]
                for i_patch in range(4):
                    fn_with_ext = "{}/{}-{}.png,{},{}" .format(os.path.join(directory,img_size+prefix) ,fn_without_ext,i_patch,label,size)
                    f_train.write(fn_with_ext + "\n")

            for y in d_test:
                fn_without_ext = y.split(".")[0]
                label, size = extract_label(y)
                for i_patch in range(4):
                    fn_with_ext = "{}/{}-{}.png,{},{}" .format(os.path.join(directory,img_size+prefix) ,fn_without_ext,i_patch,label,size)
                    f_test.write(fn_with_ext + "\n")
            
    
        #add fake images into train dataset
        if include_fake == True: 
            fake_filename = 'data_file_fake_pix2pix.csv'  #  stylegan: 'data_file_fake.csv'
            with open(os.path.join(directory,fake_filename), "r") as fake_file:
                lines = fake_file.readlines()
            for line in lines:
                f_train.write(line.strip() + "\n")
            

def create_gan_images_list(epoches,cancer_type,generated_size,dirs,total_images):
    # create training data file 
    import random 

    remove_all_training_files(os.path.join(dirs,'epoch'))

    sizes = ['40','100','200','400']
    #dirs = "/tf/dataset/classes"
    for epoch in range(epoches):
        cond_images = []
        orgi_images = []
        for size in sizes:
            data_file = "{}_{}_files.csv" . format(cancer_type,size)
            if size != generated_size:
                with open(os.path.join(dirs,data_file), "r") as f:
                    lines = f.readlines()
                for ln in lines:
                    cond_images.append(ln.strip())
            else:
                with open(os.path.join(dirs,data_file), "r") as f:
                    lines = f.readlines()
                for ln in lines:
                    orgi_images.append(ln.strip())
        result_images = []
        random.shuffle(cond_images)
        i = 0
        while i < total_images:
            for fn_org in orgi_images:

                idx = random.randint(1,len(cond_images)-1)

                gen_line =  cond_images[idx] +"," + fn_org

                result_images.append(gen_line)
                
                i = i + 1
                
                if i > total_images:
                    break

        output_file = os.path.join(dirs,"epoch","gan_epoch_{}.csv".format(epoch))
        with open(output_file, 'w') as writer:
            for line in result_images:
                writer.write(line + "\n")
                
def create_gan_images_list_by_bach(epoches,cancer_type,generated_size,dirs,total_images):
    # create training data file 
    import random 

    #dirs = "/tf/dataset/classes"
    for epoch in range(epoches):
        cond_images = []
        orgi_images = []

        data_file = "{}_{}_files.csv" . format(cancer_type,generated_size)
        bach_data_file = "BACH_{}_data_files.csv" . format(cancer_type)

        with open(os.path.join(dirs,bach_data_file), "r") as f:
            lines = f.readlines()
        for ln in lines:
            cond_images.append(ln.strip())

        with open(os.path.join(dirs,data_file), "r") as f:
            lines = f.readlines()
        for ln in lines:
            orgi_images.append(ln.strip())
                
        result_images = []
        random.shuffle(cond_images)
        i = 0
        while i < total_images:
            for fn_org in orgi_images:

                idx = random.randint(1,len(cond_images)-1)

                #gen_line =  cond_images[idx] +"," + fn_org
                gen_line =  fn_org +"," + cond_images[idx]

                result_images.append(gen_line)
                
                i = i + 1
                
                if i > total_images:
                    break

        output_file = os.path.join(dirs,"epoch","gan_epoch_{}.csv".format(epoch))
        with open(output_file, 'w') as writer:
            for line in result_images:
                writer.write(line + "\n")
                
                
def load_gan_dataset(csvfile,dataset_volume,batch_size,split=0.9):
    
    dataset = tf.data.experimental.CsvDataset(csvfile,[tf.string, tf.string])
    
    train, test = split_train_test(dataset,dataset_volume,batch_size,split)
    
    train = train.map(normalize_gan_image)
    
    train = train.shuffle(BUFFER_SIZE)

    test = test.map(normalize_gan_image)


    return train,test

def normalize_gan_image(f1,f2):
    
    #original image
    image_1 = tf.io.read_file(f1)    
    image_1 = tf.image.decode_png(image_1)
    image_1 = tf.cast(image_1, tf.float32)
    image_A = _normalize(image_1,3)
    #image_A = image_1
    image_A = tf.image.resize(image_A, RESIZE_IMG)
    
    #conditional image
    image_2 = tf.io.read_file(f2)
    image_2 = tf.image.decode_png(image_2)
    image_2 = tf.cast(image_2, tf.float32)
    image_B = _normalize(image_2,3)
    #image_B = image_2
    image_B = tf.image.resize(image_B, RESIZE_IMG)
        
    return image_A,image_B

#---------------------------------------------------BACH dataset-------------------------------------------#

def normalize_BACH_image(file_image, label):
    
    image = tf.io.read_file(file_image)

    image = tf.image.decode_png(image)  
    
    image = tf.cast(image, tf.float32)
     
    image = _normalize(image,2)

    image = tf.image.resize(image, RESIZE_IMG)
    
    return image,label,file_image

def load_BACH_dataset(directory, epoch):
    fn_train_ep = "train_data_file_ep_{}.csv" . format(epoch)
    fn_test_ep = "test_data_file_ep_{}.csv" . format(epoch)
    
    train = tf.data.experimental.CsvDataset(os.path.join(directory,'epoch',fn_train_ep),[tf.string, tf.int32])

    test = tf.data.experimental.CsvDataset(os.path.join(directory,'epoch',fn_test_ep),[tf.string, tf.int32])

    train = train.shuffle(BUFFER_SIZE)    
    
    train = train.map(normalize_BACH_image)
    
    test = test.map(normalize_BACH_image)    

    return train, test

def extract_BACH_label(filename):
    parts = filename
    if parts[0] == "b" or parts[0] == "n":
        label = 1
    else:
        label = 0
    return label

#split train vs test as 70/30
def generate_train_test_data_split_for_BACH(directory, epoch,prefix,include_fake=False):
    from sklearn.model_selection import train_test_split
    import random
    
    remove_all_training_files(os.path.join(directory,'epoch'))
    
    
    full_img_data = ["Normal","Invasive","InSitu","Benign"]
   
    for i_ep in range(epoch):
        random_seed=random.randint(1,1001) + i_ep
        train_filename = "train_data_file_ep_{}.csv" . format(i_ep)
        test_filename = "test_data_file_ep_{}.csv" . format(i_ep)
        f_train = open(os.path.join(directory,'epoch',train_filename),'a')
        f_test = open(os.path.join(directory,'epoch',test_filename),'a')
        for img_size in full_img_data:
            
            #read full images
            list_img = []
            with open(os.path.join(directory,'data_file_{}.csv' . format(img_size)), "r") as f:
                lines = f.readlines()
            for ln in lines:
                line = ln.strip().split('/')[-1]
                list_img.append(line)
            
            #convert images into 4 patch images
            d_train, d_test = train_test_split(list_img,test_size=0.3, random_state=random_seed)
            for x in d_train:
                label = extract_BACH_label(x)
                fn_without_ext = x.split(".")[0]
                for i_patch in range(4):
                    fn_with_ext = "{}/{}-{}.png,{}" .format(os.path.join(directory,img_size+prefix) ,fn_without_ext,i_patch,label)
                    f_train.write(fn_with_ext + "\n")

            for y in d_test:
                fn_without_ext = y.split(".")[0]
                label = extract_BACH_label(y)
                for i_patch in range(4):
                    fn_with_ext = "{}/{}-{}.png,{}" .format(os.path.join(directory,img_size+prefix) ,fn_without_ext,i_patch,label)
                    f_test.write(fn_with_ext + "\n")
            