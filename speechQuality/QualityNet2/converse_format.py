###用于将WAV（sphfile）转为wav格式


import params as hp
from sphfile import SPHFile
import glob
import os

if __name__ == "__main__":
    path = '/home/superwu/PycharmProjects/TIMIT/NOISY_TRAIN/*.WAV'
    sph_files = glob.glob(path)
    print(len(sph_files), "train utterences")
    for i in sph_files:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace(".WAV", "_.wav"))
        os.remove(i)
    path = '/home/superwu/PycharmProjects/TIMIT/TEST/*.WAV'
    sph_files_test = glob.glob(path)
    print(len(sph_files_test), "test utterences")
    for i in sph_files_test:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace(".WAV", "_.wav"))
        os.remove(i)
    print("Completed")