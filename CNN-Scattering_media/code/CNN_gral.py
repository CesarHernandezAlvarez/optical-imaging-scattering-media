#!/usr/bin/env python
# coding: utf-8

## LIBRARIES
import os,datetime,time,timeit,scipy.io,random,cv2,math
import keras
import traceback
import contextlib
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as tfb
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim_
from PIL import Image
from pathlib import Path
from scipy.fft import fftn
from natsort import os_sorted
from IPython.display import clear_output
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import losses,metrics
from keras.regularizers import l2
from keras.models import Sequential,Model,load_model, save_model
from keras.callbacks import TensorBoard,EarlyStopping, CSVLogger,ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Average,Input,Dropout,BatchNormalization,Flatten,Activation,Dense
from emnist import list_datasets,extract_training_samples,extract_test_samples

## INITIALIZATION OF GPUS
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

@contextlib.contextmanager
def options(options):
  old_opts = tf.config.optimizer.get_experimental_options()
  tf.config.optimizer.set_experimental_options(options)
  try:
    yield
  finally:
    tf.config.optimizer.set_experimental_options(old_opts)

def intorstr(n):
    if n.isdigit():
        nint = int(n)
    else:
        nint = str(n)
    return nint
        
task_id = int(os.environ['SGE_TASK_ID'])
#filetoopen = 'parameters_HPC.txt'
filetoopen = 'parameters_HPC_best.txt'
with open(filetoopen) as arg_file:
    args = arg_file.readlines()[task_id].split()

data_select, input_, output_, optimizer, init_, input_size, version, grit, totalDataset, epochsN, minibatch, dout, lr = map(intorstr, args[1:])
print("Selection of data: {}\nInput: {}\nOutput: {}\nInput size: {}\nVersion: {}\nGrit: {}\nDataset size: {}\nEpochs: {}\nBatch size: {}\nDropout: {}\nOptimizer: {}\nLearning rate: {}\nInitializer: {}".format(data_select,input_,output_,input_size,version,grit,totalDataset,epochsN,minibatch,dout,optimizer,lr,init_))

lr_str = '1e-'+ str(lr)
lr_ = float(lr_str)
if optimizer == 'Adam':
    optim = keras.optimizers.Adam(learning_rate=lr_)
elif optimizer == 'Adadelta':
    optim = keras.optimizers.Adadelta(learning_rate=lr_)

if init_ == 'glunif':
    k_initializer = 'glorot_uniform'
elif init_ == 'glnorm':
    k_initializer = 'glorot_normal'

dropoutR = dout/100

## CREATION OF PATHS/FOLDERS
ty = datetime.date.today()
basepath = '/data/home/exx484/'
savepath_ = '/data/scratch/exx484/models/'
savepath = savepath_ + '{}/'.format(ty) + input_ + '_' + output_ + '/v{:02}_sz{:03}_g{:03}_ds{:05}_ep{:04}_mb{:03}_do{:02}_'.format(version,input_size,grit,totalDataset,epochsN,minibatch,dout) + optimizer + str(lr) + '_' + init_ + '/'
if os.path.exists(savepath)==False:
    os.makedirs(savepath)
savelog_path = savepath_ + '{}/'.format(ty) + input_ + '_' + output_ + '/metricsToday/'
if os.path.exists(savelog_path)==False:
    os.makedirs(savelog_path)
best_model_path = savepath + 'best.hdf5'
model_path = savepath + 'finished.hdf5'
log_path = savepath + 'metrics.csv'
dataIDX_path = savepath + 'data_idx.npy'
dataEMNIST_path = savepath + 'data_emnist.npy'

total_log_path = savelog_path + 'v{:02}_sz{:03}_g{:03}_ds{:05}_ep{:04}_mb{:03}_do{:02}_'.format(version,input_size,grit,totalDataset,epochsN,minibatch,dout) + optimizer + str(lr) + '_' + init_ + '.csv'

fig_dig = 'digits'
seed_dig = extract_training_samples(fig_dig)
idx_d = np.argsort(np.array(seed_dig[1]))
sort_targets_d = np.array(seed_dig[0])[idx_d]
sort_labels_d = np.array(seed_dig[1])[idx_d]
sort_labels_d = np.asarray(sort_labels_d)
del seed_dig

fig_let = 'letters'
seed_let = extract_training_samples(fig_let)
idx_l = np.argsort(np.array(seed_let[1]))
sort_targets_l = np.array(seed_let[0])[idx_l]
sort_labels_l = np.array(seed_let[1])[idx_l]
sort_labels_l = np.asarray(sort_labels_l)
del seed_let

rnd_dl = np.load('/data/home/exx484/newData/EMNIST_gral/list_sortedemnist.npy')
if data_select == 'Random':
    for lod in range(0,10):
        range1 = lod*2600
        range2 = (lod+1)*2600
        class_d = rnd_dl[range1:range2]
        temp_d_idx = random.sample(range(range1,range2),int(totalDataset/20))
        n_ = 0
        temp_d = np.zeros((int(totalDataset/20),))
        for idx_ in temp_d_idx:
            temp_d[n_] = class_d[idx_-2600*lod]
            n_ += 1
        if lod == 0:
            final_d = list(temp_d)
            final_d_idx = temp_d_idx
            clear_output(True)
            #print("Digits class " + str(lod) + " added")
        else:
            final_d.extend(list(temp_d))
            final_d_idx.extend(temp_d_idx)
            clear_output(True)
            #print("Digits class " + str(lod) + " added")
        del temp_d, temp_d_idx
    clear_output(True)
    #print("All classes added\nDigits list finished")
    print("Lenght of digits list: " + str(len(final_d)))
    time.sleep(1.5)

    for lol in range(0,26):
        range1 = int(len(rnd_dl)/2 + lol*1000)
        range2 = int(len(rnd_dl)/2 + (lol+1)*1000)
        class_l = rnd_dl[range1:range2]
        extras = int(totalDataset/2-int(totalDataset/52)*26)
        if lol < extras:
            temp_l_idx = random.sample(range(range1,range2),int(totalDataset/52+1))
            temp_l = np.zeros((int(totalDataset/52+1),))
        else:
            temp_l_idx = random.sample(range(range1,range2),int(totalDataset/52))
            temp_l = np.zeros((int(totalDataset/52),))
        n_ = 0
        for idx_ in temp_l_idx:
            temp_l[n_] = class_l[idx_- int(len(rnd_dl)/2) - 1000*lol]
            n_ += 1
        if lol == 0:
            final_l = list(temp_l)
            final_l_idx = temp_l_idx
            clear_output(True)
            #print("Letters class " + str(lol + 1) + " added")
        else:
            final_l.extend(temp_l)
            final_l_idx.extend(temp_l_idx)
            clear_output(True)
            #print("Letters class " + str(lol + 1) + " added")
        del temp_l, temp_l_idx
    clear_output(True)    
    #print("All classes added\nLetters list finished")
    print("Lenght of letters list: " + str(len(final_l)))
    time.sleep(1.5)

    final_dl = final_d[:]
    final_dl.extend(final_l)
    final_dl_idx = final_d_idx[:]
    final_dl_idx.extend(final_l_idx)
    clear_output(True)
    print("Lenght of final list: " + str(len(final_dl)))
    print("Lenght of final index list: " + str(len(final_dl_idx)))
elif data_select == 'Loadnpy':
    ty_load = ''
    #loading_path1k = savepath_ + '2025-01-04/{}_{}/v03_sz128_g600_ds01000_ep0020_mb128_do25_Adam3_glnorm/'.format(input_,output_)
    loading_path500 = savepath_ + '500_'
    loading_path1k = savepath_ + '1k_'
    loading_path2k = savepath_ + '2k_'
    loading_path5k = savepath_ + '5k_'
    loading_path10k = savepath_ + '10k_'
    loading_path20k = savepath_ + '20k_'
    loading_path30k = savepath_ + '30k_'
    loading_path30k = savepath_ + '40k_'
    loading_path108 = savepath_ + '2023-06-19/FM_Original/v01_sz128_g600_ds05000_ep0050_mb032_do00_Adam1_glnorm/'.format(totalDataset)

    if totalDataset == 500:
      load_dataIDX_path = loading_path500 + 'data_idx.npy'
      load_dataEMNIST_path = loading_path500 + 'data_emnist.npy'
    elif totalDataset == 1000:
      load_dataIDX_path = loading_path1k + 'data_idx.npy'
      load_dataEMNIST_path = loading_path1k + 'data_emnist.npy'
    elif totalDataset == 2000:
      load_dataIDX_path = loading_path2k + 'data_idx.npy'
      load_dataEMNIST_path = loading_path2k + 'data_emnist.npy'
    elif totalDataset == 5000:
      load_dataIDX_path = loading_path5k + 'data_idx.npy'
      load_dataEMNIST_path = loading_path5k + 'data_emnist.npy'
    elif totalDataset == 10000:
      load_dataIDX_path = loading_path10k + 'data_idx.npy'
      load_dataEMNIST_path = loading_path10k + 'data_emnist.npy'
    elif totalDataset == 20000:
      load_dataIDX_path = loading_path20k + 'data_idx.npy'
      load_dataEMNIST_path = loading_path20k + 'data_emnist.npy'
    elif totalDataset == 30000:
      load_dataIDX_path = loading_path30k + 'data_idx.npy'
      load_dataEMNIST_path = loading_path30k + 'data_emnist.npy'
    elif totalDataset == 40000:
      load_dataIDX_path = loading_path40k + 'data_idx.npy'
      load_dataEMNIST_path = loading_path40k + 'data_emnist.npy'
    else:
      load_dataIDX_path = loading_path108 + 'data_idx.npy'
      load_dataEMNIST_path = loading_path108 + 'data_emnist.npy'
          
    idx_scipy = scipy.io.loadmat(load_dataIDX_path)
    emnist_scipy = scipy.io.loadmat(load_dataEMNIST_path)
    final_dl_idx = idx_scipy['final_dl_idx']
    final_dl_idx = final_dl_idx[0]
    final_dl = emnist_scipy['final_dl']
    final_dl = final_dl[0]
    del idx_scipy,emnist_scipy
    
scipy.io.savemat(dataIDX_path,{"final_dl_idx" : final_dl_idx})
scipy.io.savemat(dataEMNIST_path,{"final_dl" : final_dl})

spec_path = '/data/home/exx484/newData/EMNIST_gral/G' + str(grit) + '_D1/'
gt_path = '/data/home/exx484/newData/EMNIST_gral/GT_0/'
targ_path = ['0_4/','5_9/','a_m/','n_z/']

def get_fft2(image):   
    fft2_ = np.fft.fftshift(np.fft.fft2(image))
    return fft2_

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r

def getRandomInputs(type_in,list_idx,im_sz):
    nan_count = 0
    q_ = 0
    for target_ in targ_path:
        path_ = spec_path + target_
        img_dir = os.listdir(path_)
        sorted_files = os_sorted(img_dir)
        list_idx_q = list_idx[int(q_*totalDataset/4):int((q_+1)*totalDataset/4)]
        for noFile in list_idx_q:
            if noFile >= 13000:
                noFile = noFile%13000
            file = sorted_files[noFile]
            image = Image.open(path_ + file)
            img = np.asarray(image)
            img = np.rot90(np.fliplr(img))
            if type_in == "ACav":
                x_temp = np.zeros((im_sz,im_sz))
                auto_av = np.zeros((im_sz,im_sz))
                for i16 in range(0,16):
                    x_temp = img[(math.floor(i16/4)*im_sz):((math.floor(i16/4)+1)*im_sz),((i16%4)*im_sz):((i16%4+1)*im_sz)]
                    x_temp = np.log(abs(get_fft2(x_temp)))
                    x_temp = np.nan_to_num(x_temp)
                    auto_av = auto_av[:,:] + x_temp[:,:]
                auto_av = auto_av/16
                auto_av = preprocessing.normalize(auto_av)
                yield auto_av
            elif type_in == 'MegaFM':
                speckle = np.zeros((im_sz*2, im_sz*2))
                min_ = int(img.shape[1]/2 - im_sz)
                max_ = int(img.shape[1]/2 + im_sz)
                speckle = img[min_:max_,min_:max_]
                autoc = get_fft2(speckle)
                fs_ = get_fft2(abs(autoc))
                fmag = abs(fs_)
                fmag = preprocessing.normalize(fmag)
                min_min = int(fmag.shape[1]/2 - im_sz/2)
                max_max = int(fmag.shape[1]/2 + im_sz/2)
                megafmag = np.zeros((im_sz,im_sz))
                megafmag = fmag[min_min:max_max,min_min:max_max]
                yield megafmag
            else:
                speckle = np.zeros((im_sz, im_sz))
                min_ = int(img.shape[1]/2 - im_sz/2)
                max_ = int(img.shape[1]/2 + im_sz/2)
                speckle = img[min_:max_,min_:max_]
                if type_in == 'Speckle':
                    yield speckle
                else:
                    autoc = get_fft2(speckle)
                    if type_in == 'AC':
                        yield autoc
                    else:
                        fs_ = get_fft2(abs(autoc))
                        if type_in == 'FA':
                            famp = np.real(fs_)
                            yield famp
                        elif type_in == 'FM':
                            fmag = abs(fs_)
                            fmag = preprocessing.normalize(fmag)
                            yield fmag
                        elif type_in == 'ResizeFM':
                            fmag = abs(fs_)
                            fmag = fmag[32:96,32:96]
                            rsz_fmag = cv2.resize(fmag,(im_sz,im_sz))
                            yield rsz_fmag
                        elif type_in == 'FMlog':
                            fmag_log = np.log(abs(fs_))
                            yield fmag_log
        q_ += 1
                
def getRandomGT(list_idx,im_sz):
    q_ = 0
    for target_ in targ_path:
        path_ = gt_path + target_
        img_dir = os.listdir(path_)
        sorted_files = os_sorted(img_dir)
        list_idx_q = list_idx[int(q_*totalDataset/4):int((q_+1)*totalDataset/4)]
        for noFile in list_idx_q:
            if noFile >= 13000:
                noFile = noFile%13000
            file = sorted_files[noFile]
            image = Image.open(path_ + file)
            img = np.asarray(image)
            img = np.rot90(np.fliplr(img))
            img = cv2.resize(img,(im_sz,im_sz))
            yield img
        q_ += 1
                
def getRandomOutputs(list_final,im_sz):  
    for m_ in range(0,2):
        if m_ == 0:
            list_final_h = list_final[0:int(totalDataset/2)]
            for noFile in list_final_h:
                gt = sort_targets_d[int(noFile),:,:]
                gt = cv2.resize(gt,(im_sz,im_sz))
                gt = preprocessing.binarize(gt,threshold=0.6*255)
                yield gt
        else:
            list_final_h = list_final[int(totalDataset/2):totalDataset]
            for noFile in list_final_h:
                gt = sort_targets_l[int(noFile),:,:]
                gt = cv2.resize(gt,(im_sz,im_sz))
                gt = preprocessing.binarize(gt,threshold=0.6*255)
                yield gt

gri = getRandomInputs(input_,final_dl_idx,input_size)
if output_ == 'Original':
    gro = getRandomOutputs(final_dl,input_size)
elif output_ == 'GT':
    gro = getRandomGT(final_dl_idx,input_size)

## ONLY ONE DIFFUSER
final_inputs = np.zeros((totalDataset,input_size,input_size))
for i in range(0,totalDataset):
    x0 = next(gri)
    final_inputs[i,:,:] = x0[:,:]
    #clear_output(True)
    #print("Image {} added".format(i+1))
#clear_output(True)
print("All input images added")

final_outputs = np.zeros((totalDataset,input_size,input_size))
for i in range(0,totalDataset):
    x0 = next(gro)
    final_outputs[i,:,:] = x0[:,:]
    #clear_output(True)
    #print("Image {} added".format(i+1))
clear_output(True)
print("All output images added")

clear_output(True)
print("Input images - shape: {}".format(final_inputs.shape))
print("Output images - shape: {}".format(final_outputs.shape))

## PREPARATION OF DATASETS FOR TRAINING, RESHAPING AND REARRANGING
input_shape = (final_inputs.shape[1],final_inputs.shape[2],1)
output_shape = (final_outputs.shape[1],final_outputs.shape[2],1)

## MODIFICATION OF LOSS FUNCTION
numone = 0
numzero = 0
pos_weight_sum = 0
for x_ in range(final_outputs.shape[0]):
    for y_ in range(final_outputs.shape[1]):
        for z_ in range(final_outputs.shape[2]):
            if final_outputs[x_,y_,z_] <= 0.5:
                numzero = numzero + 1
            else:
                numone = numone + 1
    pos_weight_sum = pos_weight_sum + numzero/numone    
pos_weight_ = pos_weight_sum/final_outputs.shape[0]
print("Number of zeros: {} ".format(numzero))
print("Number of ones: {} ".format(numone))
print("Value of pos_weight: {} ".format(pos_weight_))

POS_WEIGHT = pos_weight_  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

## JACCARD INDEX METRICS
def JI(y_true, y_pred):
    y_true = tfb.flatten(y_true)
    y_pred = tfb.flatten(y_pred)

    threshold_value = 0.3

    y_pred = tfb.cast(tfb.greater(y_pred, threshold_value), tfb.floatx())
    fenzi = tfb.sum(y_true * y_pred, keepdims=True)
    # true_positives_sum = tfb.sum(true_positives, keepdims=True)
    fenmu = tfb.sum(tfb.cast((tfb.greater(y_true + y_pred, 0.8)), tfb.floatx()), keepdims=True)

    return tfb.mean(fenzi / fenmu, axis=-1)
    
### ACCURACY SCORE METRICS
#def acc_score(y_true, y_pred):
#    x = y_true
#    y = y_pred
#    acc_score_ = accuracy_score(x,y)
#    return acc_score_
    
### STRUCTURAL SIMILARITY METRICS
#def ssim(y_true, y_pred):
#    x = y_true
#    y = y_pred
#    x = tf.expand_dims(x, axis=0)
#    y = tf.expand_dims(y, axis=0)
#    x = tf.image.convert_image_dtype(x, tf.float32)
#    y = tf.image.convert_image_dtype(y, tf.float32)
#    ssim_return = tf.image.ssim(x, y, max_val=1.0, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)
#    return ssim_return
    
## PEARSON CORRELATION COEFFICIENT METRICS
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tfb.mean(x, axis=0)
    my = tfb.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = tfb.sum(xm * ym)
    x_square_sum = tfb.sum(xm * xm)
    y_square_sum = tfb.sum(ym * ym)
    r_den = tfb.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return tfb.mean(r)

## PREPARATION OF DATASETS FOR TRAINING, SPLIT BETWEEN TRAIN AND VALIDATION
#final_inputs_shuffle, final_outputs_shuffle = shuffle(final_inputs,final_outputs)
x_train, x_test, y_train, y_test = train_test_split(final_inputs, final_outputs,
    test_size=0.1, random_state = 0)
x_train = np.expand_dims(x_train,3)
x_test = np.expand_dims(x_test,3)
outsize = y_train.shape[1]
if version == 3 or version == 4 or version == 6:
  y_train = np.reshape(y_train,(y_train.shape[0],outsize*outsize))
  y_test = np.reshape(y_test,(y_test.shape[0],outsize*outsize))
print("x_train shape: {}".format(x_train.shape))
print("x_test shape: {}".format(x_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))

## MODEL/ARCHITECTURE OF NEURAL NETWORK
def createCNN(input_shape,version_):
# if needed:
#  x = BatchNormalization(axis=-1)(x)
#  x = Dropout(dropoutR)(x)
    model_input = Input(shape=input_shape)
    
    if version_ == 1:
      #encoder
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(model_input) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x1 = Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 128
      x = BatchNormalization(axis=-1)(x1) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = MaxPooling2D((2,2))(x) #size 64
      x2 = Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 64
      x = BatchNormalization(axis=-1)(x2) #size 64
      x = Dropout(dropoutR)(x) #size 64
      x = MaxPooling2D((2,2))(x) #size 32
      x3 = Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 32
      x = BatchNormalization(axis=-1)(x3) #size 32
      x = Dropout(dropoutR)(x) #size 32
      x = MaxPooling2D((2,2))(x) #size 16
      #decoder
      x = Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 16
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 16
      x = UpSampling2D(2)(x) #size 32
      x = tf.keras.layers.Average()([x, x3]) #size 32
      x = Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 32
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 32
      x = UpSampling2D(2)(x) #size 64
      x = tf.keras.layers.Average()([x, x2]) #size 64
      x = Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 64
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 64
      x = UpSampling2D(2)(x) #size 128
      x = tf.keras.layers.Average()([x, x1]) #size 128
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 128
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 128
    
      x = Conv2D(1,kernel_size=(3, 3), padding= 'same', activation='sigmoid')(x) #size 128
      
    elif version_ == 2:
      #encoder
      x = Conv2D(8, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(model_input) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x1 = Conv2D(16, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 128
      x = BatchNormalization(axis=-1)(x1) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = MaxPooling2D((2,2))(x) #size 64
      x2 = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 64
      x = BatchNormalization(axis=-1)(x2) #size 64
      x = Dropout(dropoutR)(x) #size 64
      x = MaxPooling2D((2,2))(x) #size 32
      #decoder   
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 32
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 32
      x = UpSampling2D(2)(x) #size 64
      x = tf.keras.layers.Average()([x, x2]) #size 64
      x = Conv2D(16, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 64
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 64
      x = UpSampling2D(2)(x) #size 128
      x = tf.keras.layers.Average()([x, x1]) #size 128
      x = Conv2D(8, kernel_size=(3, 3),padding='same', activation='relu',kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x) #size 128
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 128
    
      x = Conv2D(1,kernel_size=(3, 3), padding= 'same', activation='sigmoid')(x) #size 128
    
    elif version_ == 3:
      
      x = Conv2D(8, kernel_size=(3, 3),padding='same', activation='relu')(model_input) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = Conv2D(16, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = MaxPooling2D((2,2))(x) #size 64
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 64
      x = BatchNormalization(axis=-1)(x) #size 64
      x = Dropout(dropoutR)(x) #size 64
      x = MaxPooling2D((2,2))(x) #size 32
    
      x = Flatten()(x)
      x = Activation('relu')(x)
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x)
      x = Dense(outsize*outsize, activation='sigmoid')(x)  
    
    elif version_ == 4:
      
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu')(model_input) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = MaxPooling2D((2,2))(x) #size 64
      x = Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 64
      x = BatchNormalization(axis=-1)(x) #size 64
      x = Dropout(dropoutR)(x) #size 64
      x = MaxPooling2D((2,2))(x) #size 32
    
      x = Flatten()(x)
      x = Activation('relu')(x)
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x)
      x = Dense(outsize*outsize, activation='sigmoid')(x)  
      
    elif version_ == 5:
      #encoder
      x = Conv2D(16, kernel_size=(3, 3),padding='same', activation='relu')(model_input) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x1 = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 128
      x = BatchNormalization(axis=-1)(x1) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = MaxPooling2D((2,2))(x) #size 64
      x2 = Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 64
      x = BatchNormalization(axis=-1)(x2) #size 64
      x = Dropout(dropoutR)(x) #size 64
      x = MaxPooling2D((2,2))(x) #size 32
      #decoder   
      x = Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 32
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 32
      x = UpSampling2D(2)(x) #size 64
      x = tf.keras.layers.Concatenate(axis=3)([x, x2]) #size 64
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 64
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 64
      x = UpSampling2D(2)(x) #size 128
      x = tf.keras.layers.Concatenate(axis=3)([x, x1]) #size 128
      x = Conv2D(16, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 128
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x) #size 128
#      x = Conv2D(8, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 128
#      x = BatchNormalization(axis=-1)(x)  
#      x = Dropout(dropoutR)(x) #size 128
      
      x = Conv2D(1,kernel_size=(3, 3), padding= 'same', activation='sigmoid')(x) #size 128
      
      
    elif version_ == 6:
      
      x = Conv2D(8, kernel_size=(3, 3),padding='same', activation='relu')(model_input) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = Conv2D(16, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 128
      x = BatchNormalization(axis=-1)(x) #size 128
      x = Dropout(dropoutR)(x) #size 128
      x = MaxPooling2D((2,2))(x) #size 64
      x = Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu')(x) #size 64
      x = BatchNormalization(axis=-1)(x) #size 64
      x = Dropout(dropoutR)(x) #size 64
      x = MaxPooling2D((2,2))(x) #size 32
    
      x = Flatten()(x)
      x = Activation('relu')(x)
      x = BatchNormalization(axis=-1)(x)
      x = Dropout(dropoutR)(x)
      x = Dense(outsize*outsize, activation='sigmoid')(x) 
    
    cnn = Model(inputs=model_input, outputs=x)

    return cnn

## COMPILATION AFTERWARDS
model = createCNN(input_shape,version)
model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer= optim, metrics = ['accuracy',pearson_r,JI])
#model.compile(loss = weighted_binary_crossentropy, optimizer= optim, metrics = ['accuracy',pearson_r,JI])
model.summary()

## MONITORING NEURAL NETWORK: CHECKPOINTS, EARLY STOPPING, LEARNING RATE REDUCTION AND CSV LOGGER
def get_callbacks():
    checkpoint = ModelCheckpoint(best_model_path, monitor='loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
    early_stopping = EarlyStopping(monitor='loss', min_delta=1e-6, patience=20, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,
                                  verbose=1, mode='auto', cooldown=0, min_lr=1e-7)
    csv_logger = CSVLogger(log_path, append=True)
    csv_logger2 = CSVLogger(total_log_path,append=True)
    return [checkpoint, csv_logger,csv_logger2,
            reduce_lr, early_stopping
            ]

## TRAINING PROCESS
starttime = timeit.default_timer()
with options({'layout_optimizer':False}):
    model.fit(x_train,y_train,batch_size=minibatch, epochs=epochsN,callbacks=get_callbacks(), validation_data=(x_test, y_test))
endtime = timeit.default_timer()
timetime = endtime - starttime
hours_ = int(timetime/3600)
minutes_ = int(timetime%3600/60)
seconds_ = format(float(timetime%3600)%60,".2f")
save_model(model, model_path)
print("Training time - {:02}h:{:02}m:{}s".format(hours_,minutes_,seconds_))
