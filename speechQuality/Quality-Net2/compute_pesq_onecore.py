###用于单进程计算pesq

from pesq import pesq
import os
import numpy as np
from scipy.io import wavfile
# from torch import nn
import multiprocessing
os.environ["CUDA_VISIBLE_DEVICES"]='0'
def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def PESQ_compute(noisy_path,clean_path):
    length=len(noisy_path)
    i=0
    list=[]
    for i in range(length):
        rate, ref = wavfile.read(clean_path[i])
        rate, deg = wavfile.read(noisy_path[i])
        score = pesq(rate, ref, deg, 'wb')
        list.append([noisy_path[i],score])
        print(list[i])
    return list

clean_list_name="/home/superwu/PycharmProjects/TIMIT/train_audio.list"
noisy_list_name="/home/superwu/PycharmProjects/TIMIT/addnoisy_train.list"
noisy_path=ListRead(noisy_list_name)
clean_path=ListRead(clean_list_name)

clean_audio=clean_path[450:650]
result=[]
for i in range(490):
    output_list=[]
    temp=noisy_path[i*200:(i+1)*200]
    output_list=PESQ_compute(temp,clean_audio)
    result=result+output_list
    print(len(result))
# output_list=PESQ_compute(clean_path,clean_path)
# #
file_name="/home/superwu/PycharmProjects/TIMIT/addnoisy_train_pesq.list"

with open(file_name,'w') as file:
    for item in result:
        file.write(str(item)+'\n')
print('ok')
