# -*- coding: utf-8 -*-


#base image: 40X
#conditional image: 100X, 200X, 400X

import sys
sys.path.append('../')

base_file = "F:\\GraduateClass\\Thesis\\Dataset\\BreaKHis\\histology_slides\\breast - 40"
cond4x_file = "F:\\GraduateClass\\Thesis\\Dataset\\BreaKHis\\histology_slides\\breast - 100"
output_file = 'D:\\gan_file_one_many.csv'
import os
import csv
lines = []
#base files
for path, _ , files in os.walk(base_file):
    for f in files:
        base_full_filename = os.path.join(path, f)
        if f.endswith(".png"):
            cond_path = path.replace("breast - 40","breast - 100")
            cond_path = cond_path.replace("40X","100X")
            cond_full_filename = os.path.join(cond_path, cond_filename)
            for subpath,_,files_cond in os.walk(cond_path):
                for f_cond in files_cond:
                    line = base_full_filename + "," + os.path.join(subpath, f_cond)
                    lines.append(line)
            
with open(output_file, 'w') as writer:
    for line in lines:
        writer.write(line + "\n")





