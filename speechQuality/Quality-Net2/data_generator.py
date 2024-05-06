###生成不同噪声下不同信噪比的语音


import os
import numpy as np
import soundfile as sf
import random
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"]='0'
def ge_NoiseData(clean_data, noise_data, snr):
    """
    利用纯净语音和噪声产生混合语音
    clean_data:纯净语音
    noise_data:噪音数据
    snr:信噪比 dB
    """
    # 噪声的归一化处理
    noise_data=noise_data/max(abs(noise_data))
    # 语音合成要保证两段信号的长度相等
    if len(noise_data) < len(clean_data):
        tmp = (len(clean_data) // len(noise_data)) + 1
        noise_data = np.tile(noise_data, tmp)
        noise_data = noise_data[:len(clean_data)]
    else:
        noise_data = noise_data[:len(clean_data)]

    clean_pwr = sum(abs(clean_data)**2)
    noise_pwr = sum(abs(noise_data)**2)  # 计算音频信号能量
    if noise_pwr==0:
        noise_pwr=1

    coefficient = np.sqrt((clean_pwr/(10**(snr / 10)))/noise_pwr)  # 噪声系数计算
    new_data = clean_data + coefficient * noise_data
    if max(abs(new_data))>32768:
        new_data = (new_data/max(abs(new_data)))*32768
    return new_data

# 生成音频采样率和幅度数组
def generate_audio_data(path):
    sig,sample_rate=sf.read(path)
    sample_sig_list=[sample_rate,sig]
    return sample_sig_list

def write_audio_data(new_path,data_list):
    sf.write(file=new_path,data=data_list,samplerate=16000)

def ListRead(filelist):
    f = open(filelist, 'r')
    Path = []
    for line in f:
        Path = Path + [line[0:-1]]
    return Path

def start_fun(clean_path):
    for i in range(len(clean_path)):
        file_name = os.path.basename(clean_path[i])
        for j in range(len(noisy_audio)):
            snr_len=len(snr_list)
            for k in range(snr_len):
                new_data = ge_NoiseData(generate_audio_data(clean_path[i])[1], generate_audio_data(noisy_audio[j])[1],snr_list[k])
                print(str(new_path_list[j * snr_len + k]) + str(file_name))

                # 文件路径+文件名
                write_audio_data(str(new_path_list[j * snr_len + k]) + str(file_name), new_data)



clean_audio_all=ListRead('/home/superwu/PycharmProjects/TIMIT/train_audio.list')
noisy_audio_all=ListRead('/home/superwu/PycharmProjects/TIMIT/noisy_recording.list')
new_path_list=ListRead("/home/superwu/PycharmProjects/TIMIT/directory_noisy.list") #更改新路径
# random.shuffle(clean_audio_all)
random.shuffle(noisy_audio_all)
# 0:250为纯净语音，250：450生成enhance组，450：650生成noisy组，
# 测试集由test-audio的前一百个语音生成，四种噪声，六种信噪比snr_list=[-6,0,6,12,18,24]
clean_audio=clean_audio_all[450:650]
noisy_audio=noisy_audio_all # 目前使用70个语音
snr_list=[0,10,18,21,24,27,30] # 手动设置需要的SNR


if __name__=="__main__":
    cpu_num=multiprocessing.cpu_count()
    chunks=len(clean_audio)//cpu_num+1
    process=[]
    for i in range(cpu_num):
        temp=clean_audio[i*chunks:(i+1)*chunks]
        p=multiprocessing.Process(target=start_fun,args=(temp,))
        p.start()
        process.append(p)
    for p in process:
        p.join()




    # for i in range(len(clean_audio)):
    #     file_name = os.path.basename(clean_audio[i])
    #     for j in range(len(noisy_audio)):
    #         for k in range(len(snr_list)):
    #             new_data=ge_NoiseData(generate_audio_data(clean_audio[i])[1],generate_audio_data(noisy_audio[j])[1],snr_list[k])
    #
    #             # 文件路径+文件名
    #             write_audio_data(str(new_path_list[j*8+k])+str(file_name),new_data)



