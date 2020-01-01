
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

from codes.data.load_data import load_files

import os
import tensorflow as tf

files = "D:\\small_dataset\\breakhis"
vol = 80
ratio = 0.8
list_dataset = []

pattern_file = os.path.join(files,"*.png") 

dataset = tf.data.Dataset.list_files(file_pattern=pattern_file,shuffle=True)

from codes.model.unet_network import unet

unet_model = unet(input_size=(512,512,3))

---------------------------------------


# Batch before shuffle.
dataset = tf.data.Dataset.from_tensor_slices([0, 0, 0, 1, 1, 1, 2, 2, 2])
dataset = dataset.batch(3)
dataset = dataset.shuffle(9)

for elem in dataset:
  print(elem)

# Prints:
# tf.Tensor([1 1 1], shape=(3,), dtype=int32)
# tf.Tensor([2 2 2], shape=(3,), dtype=int32)
# tf.Tensor([0 0 0], shape=(3,), dtype=int32)

# Shuffle before batch.
dataset = tf.data.Dataset.from_tensor_slices([0, 0, 0, 1, 1, 1, 2, 2, 2])
dataset = dataset.shuffle(9)
dataset = dataset.batch(3)

for elem in dataset:
  print(elem)