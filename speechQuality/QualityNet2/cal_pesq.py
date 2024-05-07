# 用于cpu多进程计算pesq

# from torch import nn
import multiprocessing
import os
import time
from glob import glob

import numpy as np
from pesq import pesq
from scipy.io import wavfile
import ctypes

# pDLL = ctypes.CDLL("pesq.dll")




def ListRead(path):
    with open(path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list


def PESQ_cumpute_uselist(noisy_path, clean_path):
    length = len(noisy_path)
    resultlist = []
    for i in range(length):
        rate, ref = wavfile.read(clean_path[i])
        rate, deg = wavfile.read(noisy_path[i])
        score = pesq(rate, ref, deg, 'wb')
        resultlist.append([clean_path[i], score])
        print(resultlist[i])
    return resultlist


def PESQ_compute_usename(clean_path, noisy_path):
    rate, ref = wavfile.read(clean_path)
    rate, deg = wavfile.read(noisy_path)
    score = pesq(rate, ref, deg, 'wb')
    resultlist = [noisy_path, score]
    return resultlist

def get_clean_list(noise_list):
    clean_list = []
    for noise in noise_list:
        type = noise.split('\\')[-1].split('.')[0].split('_')[:-1]
        clean_list.append(os.path.join(r"D:\work\speechEnhancement\datasets\TIMIT\data\TRAIN", *type)+".WAV.wav")
    return clean_list

def cal_fun(clean, noise):
    results = []
    for i in range(len(clean)):
        results.append(PESQ_compute_usename(clean[i], noise[i]))
    return results


def start_fun(clean, noise):
    for i in range(len(clean)):
        pesq_results.append(PESQ_compute_usename(clean[i], noise[i]))




if __name__ == '__main__':

    clean_list1 = ListRead("list/clean.list")
    noise_list1 = ListRead("list/clean.list")
    clean_list2 = ListRead("list/noise.list")
    noise_list2 = glob("noise_wavs/*.wav")
    clean_list3 = ListRead("list/enhance.list")
    noise_list3 = glob("enhance_wavs/*.wav")

    clean_list = np.array(clean_list1 + get_clean_list(noise_list2) + get_clean_list(noise_list3))
    noise_list = np.array(noise_list1 + noise_list2 + noise_list3)
    if len(clean_list) != len(noise_list):
        assert "The length of clean and noise lists do not match"
    # cpu_num = multiprocessing.cpu_count()
    # chunks = len(clean_list) // cpu_num + 1

    pesq_results = cal_fun(clean_list, noise_list)
    #
    # process = []
    # for i in range(cpu_num):
    #     clean = clean_list[i * chunks:(i + 1) * chunks]
    #     noise = noise_list[i * chunks:(i + 1) * chunks]
    #     p = multiprocessing.Process(target=start_fun, args=(clean, noise))
    #     p.start()
    #     process.append(p)
    # for p in process:
    #     p.join()
    #
    # time.sleep(10)
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # results = []
    # data = []
    # for i in range(len(clean_list)):
    #     temp = (clean_list[i], noise_list[i])
    #     data.append(temp)
    # data = list(data)
    # result = pool.starmap(PESQ_compute_usename, data)
    #
    # pool.close()
    # pool.join()
    # for item in result:
    #     results.append(item)

    with open(r"list/train.list", 'w') as f:
        for res in pesq_results:
            f.write(str(res[0])  + "," + str(res[1]) + '\n')
