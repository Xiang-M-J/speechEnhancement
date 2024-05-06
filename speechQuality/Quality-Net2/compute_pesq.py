###用于cpu多进程计算pesq

from pesq import pesq
import os
import numpy as np
from scipy.io import wavfile
# from torch import nn
import multiprocessing
#
print(multiprocessing.cpu_count())#24核
os.environ["CUDA_VISIBLE_DEVICES"]='0'
def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path


def PESQ_cumpute_uselist(noisy_path,clean_path):
    length=len(noisy_path)
    resultlist=[]
    for i in range(length):
        rate,ref=wavfile.read(clean_path[i])
        rate,deg=wavfile.read(noisy_path[i])
        score=pesq(rate,ref, deg,'wb')
        resultlist.append([clean_path[i],score])
        print(resultlist[i])
    return resultlist

def PESQ_compute_usename(noisy_path,clean_path):
    rate,ref=wavfile.read(clean_path)
    rate,deg=wavfile.read(noisy_path)
    score=pesq(rate,ref, deg,'wb')
    resultlist=[clean_path,score]
    print(resultlist)
    return resultlist
# ######



if __name__ =='__main__':
    clean_list_name="/home/superwu/PycharmProjects/TIMIT/test.list"
    noisy_list_name="/home/superwu/PycharmProjects/TIMIT/test.list"
    noisy_path=ListRead(clean_list_name)
    clean_path=ListRead(clean_list_name)

# output_list=np.zeros(len(noisy_path))
# output_list=PESQ_compute(noisy_path,clean_path)

    file_name="/home/superwu/PycharmProjects/TIMIT/test_pesq.list"
    print(multiprocessing.cpu_count)
    pool=multiprocessing.Pool(processes=multiprocessing.cpu_count())


    results=[]
    data=[]
    for i in range(len(clean_path)):
        temp=(noisy_path[i],clean_path[i])
        data.append(temp)
    data=list(data)
    result=pool.starmap(PESQ_compute_usename,data)

    pool.close()
    pool.join()
    for item in result:
        results.append(item)


#     result.start()
# print(Process, '1')
# for process in Process:
#     process.join()
    with open(file_name,'w') as file:
        for item in results:
            file.write(str(item)+'\n')

    print('ok')