# -*- coding: utf-8 -*-

import os
base_file = "C:\\dataset\\BreaKHis_v1\\histology_slides\\breast_40"
cond4x_file = "C:\\dataset\\BreaKHis_v1\\histology_slides\\breast_100"
data_file = "C:\\train_data.csv"

lines = []

for path,_, files in os.walk(base_file):
    for f in files:
        if f.endswith(".png"):
            base_full_fn = os.path.join(path,f)
            cond_f = f.replace("-40-","-100-")
            cond_path = path.replace("breast_40","breast_100")
            cond_path = cond_path.replace("40X","100X")
            cond_full_fn = os.path.join(cond_path,cond_f)
            print(cond_full_fn)
            if os.path.isfile(cond_full_fn):
                
                line = base_full_fn + "," + cond_full_fn
                
                lines.append(line)
                

with open(data_file, 'w') as f:
    for s in lines:
        f.write(str(s) + '\n') 