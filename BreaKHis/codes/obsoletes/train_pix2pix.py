# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

from codes.model.pix2pix_network import Pix2Pix
from codes.data.load_data import load_files_for_gan


dic_text_file = "C:\\train_data.csv"
batch_size = 5

train,test = load_files_for_gan(dic_text_file,587)

train = train.batch(batch_size)
for x,y in train:
    print(x.shape)
    print(y.shape)


model = Pix2Pix()
#model.generator.summary()
model.train(train, test, epochs=5, batch_size=batch_size)