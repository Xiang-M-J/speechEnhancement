## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from scipy.io import loadmat, savemat
from soundfile import SoundFile, SEEK_END
import numpy as np
import glob, os, pickle, platform
import soundfile as sf
import tensorflow as tf
import json

# from config import  max_wav_len
max_wav_len = 8 * 16000


def save_wav(path, wav, f_s):
    """
	Save .wav file.

	Argument/s:
		path - absolute path to save .wav file.
		wav - waveform to be saved.
		f_s - sampling frequency.
	"""
    wav = np.squeeze(wav)
    if isinstance(wav[0], np.float32): wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
    sf.write(path, wav, f_s)


def read_wav(path):
    """
	Read .wav file.

	Argument/s:
		path - absolute path to save .wav file.

	Returns:
		wav - waveform.
		f_s - sampling frequency.
	"""
    try:
        wav, f_s = sf.read(path, dtype='int16')
    # try: wav, f_s = sf.read(path)
    except TypeError:
        f_s, wav = sf.read(path)
    return wav, f_s


def save_mat(path, data, name):
    """
	Save .mat file.

	Argument/s:
		path - absolute path to save .mat file.
		data - data to be saved.
		name - dictionary key name.
	"""
    if not path.endswith('.mat'): path = path + '.mat'
    savemat(path, {name: data})


def read_mat(path):
    """
	Read .mat file.

	Argument/s:
		path - absolute path to save .mat file.

	Returns:
		Dictionary.
	"""
    if not path.endswith('.mat'): path = path + '.mat'
    return loadmat(path)


def gpu_config(gpu_selection, log_device_placement=False):
    """
	Selects GPU.

	Argument/s:
		gpu_selection - GPU to use.
		log_device_placement - log the device that each node is placed on.
	"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_selection)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)


def batch_list(json_file, file_dir, list_name, data_path='data', name_flag=False, make_new=False):
    """
	Places the file paths and wav lengths of an audio file into a dictionary, which
	is then appended to a list. 'glob' is used to support Unix style pathname
	pattern expansions. Checks if the training list has already been saved, and loads
	it.

	Argument/s:
		file_dir - directory containing the audio files.
		list_name - name for the list.
		data_path - path to store pickle files.
		make_new - re-create list.

	Returns:
		batch_list - list of file paths and wav length.
	"""
    # extension = ['*.wav', '*.flac', '*.mp3']
    extension = ['*.wav']
    if not make_new:
        if os.path.exists(data_path + '/' + list_name + '_list_' + platform.node() + '.p'):
            print('Loading ' + list_name + ' list...')
            with open(data_path + '/' + list_name + '_list_' + platform.node() + '.p', 'rb') as f:
                batch_list = pickle.load(f)
            if batch_list[0]['file_path'].find(file_dir) != -1:
                print(list_name + ' list has a totaltry: of %i entries.' % (len(batch_list)))
                return batch_list

    print('Creating ' + list_name + ' list...')
    batch_list = []
    with open(json_file, 'r') as f:
        json_list = json.load(f)
    # json_list = json_list[: 200]
    # json_list = json_list[: 50000]
    # print(json_list[:20])
    for i in extension:
        for j_file in json_list:
            if name_flag:
                j_file = process_txt(j_file, 0, '_')
            j = os.path.join(file_dir, j_file + '.wav')
            # for j in glob.glob(os.path.join(file_dir, i)):
            try:
                f = SoundFile(j)
                wav_len = f.seek(0, SEEK_END)
                if wav_len == -1:
                    wav, _ = read_wav(j)
                    wav = wav[:max_wav_len]
                    wav_len = len(wav)
            except NameError:
                wav, _ = read_wav(j)
                wav = wav[:max_wav_len]
                wav_len = len(wav)
            batch_list.append({'file_path': j, 'wav_len': wav_len})  # append dictionary.
    if not os.path.exists(data_path): os.makedirs(data_path)  # make directory.
    with open(data_path + '/' + list_name + '_list_' + platform.node() + '.p', 'wb') as f:
        pickle.dump(batch_list, f)
    print('The ' + list_name + ' list has a total of %i entries.' % (len(batch_list)))
    return batch_list


def process_txt(txt_line, part=1, symbol='-'):
    txt_line = os.path.splitext(txt_line)
    txt_line = txt_line[0].split(symbol)
    return txt_line[part]


def val_wav_batch(json_file, val_s_dir, val_d_dir):
    """
	Produces the validation batchs. Identical filenames for the clean speech and
	noise must be placed in their respective directories, with the SNR at the
	end of the filename. Their lengths must also be identical.

	As an example: './val_clean_speech/ANY_NAME_-5dB.wav'

	contains the clean speech, and

	'./val_noise/ANY_NAME_-5dB.wav'

	contains the noise at the same length. They will be mixed together at the SNR
	level specified in the filename.

	Argument/s:
		val_s_dir - path to clean-speech validation files.
		val_d_dir - path to noise validation files.

	Outputs:
		val_s - batch of clean-speech padded waveforms.
		val_d - batch of noise padded waveforms.
		val_s_len - lengths of clean-speech waveforms.
		val_d_len - lengths of noise waveforms.
		val_snr - batch of SNR levels.
	"""
    print("Loading validation waveforms...")
    val_s_list = []
    val_d_list = []
    val_s_len_list = []
    val_d_len_list = []
    val_snr_list = []
    # extension = ['*.wav', '*.flac', '*.mp3']
    with open(json_file, 'r') as f:
        json_list = json.load(f)
    # json_list = json_list[: 1000]
    # json_list = json_list[: 8]

    # s_paths = sorted(glob.glob(os.path.join(val_s_dir, i)))
    # d_paths = sorted(glob.glob(os.path.join(val_d_dir, i)))

    # for (j,k) in zip(s_paths, d_paths):
    # 	s_basename = os.path.basename(os.path.splitext(j)[0])
    # 	d_basename = os.path.basename(os.path.splitext(k)[0])
    # 	if s_basename != d_basename:
    # 		raise ValueError("The clean speech and noise validation files do not match: {} and {}.".format(s_basename, d_basename))
    # 	if s_basename[-2:] != "dB":
    # 		raise ValueError("The basename of the following file must end in dB: {}.".format(s_basename))
    for file_name in json_list:
        j = os.path.join(val_s_dir, process_txt(file_name, 0, '_') + '.wav')
        k = os.path.join(val_d_dir, file_name + '.wav')
        #		(s_wav, _) = read_wav(j) # read waveform from given file path. #WSJ
        (s_wav, _) = read_wav(k)  # read waveform from given file path. #VB
        (d_wav, _) = read_wav(k)  # read waveform from given file path.
        s_wav = s_wav[: 8 * 16000]
        d_wav = d_wav[: 8 * 16000]
        if len(s_wav) != len(d_wav):
            raise ValueError(
                "The clean speech and noise validation waveforms have different lengths: {} and {} for {}.".format(
                    len(s_wav), len(d_wav), file_name))
        if np.isnan(s_wav).any() or np.isinf(s_wav).any():
            raise ValueError("The clean speech waveform has either NaN or Inf values: {}.".format(j))
        if np.isnan(d_wav).any() or np.isinf(d_wav).any():
            raise ValueError("The noise waveform has either NaN or Inf values: {}.".format(k))
        val_s_list.append(s_wav)
        val_d_list.append(d_wav)
        val_s_len_list.append(len(s_wav))
        val_d_len_list.append(len(d_wav))
        # val_snr_list.append(float(s_basename.split("_")[-1][:-2]))
        val_snr_list.append(float(process_txt(file_name, 1, '_')))

    if len(val_s_len_list) != len(val_d_len_list):
        raise ValueError("The number of clean speech and noise validation files do not match.")
    max_wav_len = max(val_s_len_list)  # maximum length of waveforms.
    val_s = np.zeros([len(val_s_len_list), max_wav_len], np.int16)  # numpy array for padded waveform matrix.
    val_d = np.zeros([len(val_d_len_list), max_wav_len], np.int16)  # numpy array for padded waveform matrix.
    for (i, s_wav) in enumerate(val_s_list): val_s[i, :len(s_wav)] = s_wav  # add clean-speech waveform to numpy array.
    for (i, d_wav) in enumerate(val_d_list): val_d[i, :len(d_wav)] = d_wav  # add noise waveform to numpy array.
    val_s_len = np.array(val_s_len_list, np.int32)
    val_d_len = np.array(val_d_len_list, np.int32)
    val_snr = np.array(val_snr_list, np.int32)
    return val_s, val_d, val_s_len, val_d_len, val_snr
