#!/usr/bin/python
# Badri Adhikari, 6-7-2019
# https://github.com/badriadhikari/

################################################################################

from keras.layers import *
import keras
from keras.models import Model
from keras.models import load_model
import os, sys, datetime
import tensorflow as tf
import numpy as np
K.set_image_data_format('channels_last')
import argparse
import keras.backend as K
epsilon = K.epsilon()
from io import BytesIO, StringIO
from tensorflow.python.lib.io import file_io

################################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = "3" #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

################################################################################
n_channels = 13
pathname = os.path.dirname(sys.argv[0])
#model_weights_file_name = pathname + '/weights-rdd-seq.hdf5'
model_weights_file_name='model-09_06_2019_17_48_46_118149.h5'

################################################################################
# Feature file that has 1D, and 2D features
def getX_asis(feature_file):
    features_to_accept = []
    features_to_accept.append('# Psipred')
    features_to_accept.append('# Psisolv')
    features_to_accept.append('# Shannon entropy sum')
    features_to_accept.append('# ccmpred')
    features_to_accept.append('# freecontact')
    features_to_accept.append('# pstat_pots')
    lines = []
    f = open(feature_file, mode = 'r')
    lines = f.read()
    f.close()
    lines = lines.splitlines()
    S = ''
    for i, line in enumerate(lines):
        if line.strip() == '# Sequence':
            S = lines[i+1].strip()
    L = 0
    Data = []
    for i, line in enumerate(lines):
        if line.strip() == '# Psipred':
            L = lines[i+1].strip().split()
            L = int(len(L))
    if L != len(S):
        print (('Expected lengths L ' + str(L) + ' and seq len ' + S + ' do not match!'))
        sys.exit(1)
    flag_accept_this_feature = False
    for line in lines:
        if line.startswith('#'):
            if line.strip() in features_to_accept:
                flag_accept_this_feature = True
            else:
                flag_accept_this_feature = False
            continue
        if not flag_accept_this_feature:
            continue
        this_line = line.strip().split()
        if len(this_line) ==  0:
            continue
        if len(this_line) ==  1:
            # 0D feature
            feature2D = np.zeros((L, L))
            feature2D[:, :] = float(this_line[0])
            Data.append(feature2D)
        elif len(this_line) ==  L:
            # 1D feature
            feature2D1 = np.zeros((L, L))
            feature2D2 = np.zeros((L, L))
            for i in range(0, L):
                feature2D1[i, :] = float(this_line[i])
                feature2D2[:, i] = float(this_line[i])
            Data.append(feature2D1)
            Data.append(feature2D2)
        elif len(this_line) ==  L * L:
            # 2D feature
            feature2D = np.asarray(this_line).reshape(L, L)
            Data.append(feature2D)
        else:
            print (line)
            print(('Error!! Unknown length of feature in !!' + feature_file))
            print(('Expected length 0, ' + str(L) + ', or ' +
                   str(L*L) + ' - Found ' + str(len(this_line))))
            sys.exit()
    F = len(Data)
    X = np.zeros((L, L, F))
    for i in range(0, F):
        X[0:L, 0:L, i] = Data[i]
    return X, L, F, S

################################################################################
def sanity_check_input(X):
    print('ChannelID         Avg        Max        Sum')
    for i in range(0, len(X[0, 0, :])):
        (m, s, a) = (X[:, :, i].flatten().max(), X[:, :, i].flatten().sum(), X[:, :, i].flatten().mean())
        print(' Channel%2s %10.4f %10.4f %10.1f' % (i, a, m, s))

################################################################################
def build_model(xydim, n_channels, arch_name = 'rdd', depth = 32):
    print('')
    print('Build a model..')
    F=n_channels
    input = Input(shape = (256, 256, F))
    tower = BatchNormalization()(input)
    tower = Activation('relu')(tower)
    tower1 = Convolution2D(256, 3, padding = 'same')(tower)
    tower1 = LeakyReLU(alpha=0.05)(tower1)
    tower1=Dropout(0.15)(tower1)
    #tower1 = Activation('relu')(tower1)
    # tower1 = BatchNormalization()(tower1)
    tower2 = Convolution2D(128, 3, padding = 'same')(tower1)
    tower2 = LeakyReLU(alpha=0.05)(tower2)
    tower2=Dropout(0.15)(tower2)
#    tower2 = Activation('relu')(tower2)
    #tower2 = BatchNormalization()(tower2)
    d_rate=1
    tower3 = Convolution2D(64, 3, dilation_rate=(d_rate,d_rate),padding = 'same')(tower2)
#    tower3 = Activation('relu')(tower3)
    tower3 = LeakyReLU(alpha=0.05)(tower3)
    tower3=Dropout(0.15)(tower3)
    d_rate=2
    #tower3 = BatchNormalization()(tower3)
    tower4 = Convolution2D(64, 3,dilation_rate=(d_rate,d_rate),padding = 'same')(tower3)
    tower4 = LeakyReLU(alpha=0.05)(tower4)
    tower4=Dropout(0.25)(tower4)
#    tower4 = Activation('relu')(tower4)
    #tower4 = BatchNormalization()(tower4)
    d_rate=4
    tower5 = Convolution2D(64, 3,dilation_rate=(d_rate,d_rate), padding = 'same')(tower4)
    tower5 = LeakyReLU(alpha=0.05)(tower5)
    tower5=Dropout(0.25)(tower5)
#    tower5 = Activation('relu')(tower5)
    #tower5 = BatchNormalization()(tower5)
    tower6 = Convolution2D(32, 3, padding = 'same')(tower5)
    tower6 = LeakyReLU(alpha=0.05)(tower6)
    tower6=Dropout(0.25)(tower6)
#    tower6 = Activation('relu')(tower6)
    #tower4 = BatchNormalization()(tower4)
    tower7 = Convolution2D(32, 3, padding = 'same')(tower6)
    tower7 = LeakyReLU(alpha=0.05)(tower7)
    tower7=Dropout(0.25)(tower7)
#    tower7 = Activation('relu')(tower7)
    #tower5 = BatchNormalization()(tower5)


    tower7_1 = Deconvolution2D(32, 3, padding = 'same')(tower7)
    tower7_1 = LeakyReLU(alpha=0.05)(tower7_1)
    tower7_1 = concatenate([tower7_1,tower6],axis=3)
    #tower5_1 = BatchNormalization()(tower5_1)
    tower6_1 = Deconvolution2D(32, 3, padding = 'same')(tower7_1)
    tower6_1 = LeakyReLU(alpha=0.05)(tower6_1)
    tower6_1 = concatenate([tower6_1,tower5],axis=3)
    #tower4_1 = BatchNormalization()(tower4_1)
    tower5_1 = Deconvolution2D(64, 3, padding = 'same')(tower6_1)
    tower5_1 = LeakyReLU(alpha=0.05)(tower5_1)
    tower5_1 = concatenate([tower5_1,tower4],axis=3)
    #tower5_1 = BatchNormalization()(tower5_1)
    tower4_1 = Deconvolution2D(64, 3, padding = 'same')(tower5_1)
    tower4_1 = LeakyReLU(alpha=0.05)(tower4_1)
    tower4_1 = concatenate([tower4_1,tower3],axis=3)
    #tower4_1 = BatchNormalization()(tower4_1)
    tower3_1 = Deconvolution2D(64, 3, padding = 'same')(tower4_1)
    tower3_1 = LeakyReLU(alpha=0.05)(tower3_1)
    tower3_1 = concatenate([tower3_1,tower2],axis=3)
    #tower3_1 = BatchNormalization()(tower3_1)
    tower2_1 = Deconvolution2D(128, 3, padding = 'same')(tower3_1)
    tower2_1 = LeakyReLU(alpha=0.05)(tower2_1)
    tower2_1 = concatenate([tower2_1,tower1],axis=3)
    #tower2_1 = BatchNormalization()(tower2_1)
    tower = Deconvolution2D(256, 3, padding = 'same')(tower2_1)
    tower = LeakyReLU(alpha=0.05)(tower)
    tower=Dense(32,activation='relu')(tower)
    tower=Dense(1,activation='relu')(tower)
    model = Model(input, tower)
    return model
   # print(xydim, n_channels, arch_name, depth)
    #if arch_name == "rdd": # residual with dilation and dropout
     #   dropout_value = 0.3
      #  print('Building model ' + arch_name + ' with depth = ' + str(depth) + ' and dropout = ' + str(dropout_value))
       # my_input = Input(shape = (xydim, xydim, n_channels))
        #tower = BatchNormalization()(my_input)
       # tower = Activation('relu')(tower)
       # tower = Convolution2D(64, 1, padding = 'same')(tower)
       # n_channels = 64
       # d_rate = 1
       # for i in range(depth):
       #     block = BatchNormalization()(tower)
       #     block = Activation('relu')(block)
       #     block = Convolution2D(64, 3, padding = 'same')(block)
       #     block = Dropout(dropout_value)(block)
       #     block = Activation('relu')(block)
       #     block = Convolution2D(n_channels, 3, dilation_rate=(d_rate, d_rate), padding = 'same')(block)
        #    tower = keras.layers.add([block, tower])
         #   if d_rate == 1:
          #      d_rate = 2
          #  elif d_rate == 2:
          #      d_rate = 4
          #  else:
          #      d_rate = 1
#        tower = BatchNormalization()(tower)
 #       tower = Activation('relu')(tower)
  #      tower = Convolution2D(1, 3, padding = 'same')(tower)
   #     tower = Activation('sigmoid')(tower)
    #    model = Model(my_input, tower)
     #   return model
  #  else:
   #     print("Error!! Unxpected model type!!")
    #    sys.exit(1)

################################################################################
def main(feat, file_rr):
    print("Start " + str(sys.argv[0]) + " - " + str(datetime.datetime.now()))
    global n_channels
    global model_weights_file_name
    if not os.path.isfile(model_weights_file_name):
        print('Model weights file ' + model_weights_file_name + ' is absent!\n')
        print('Please download from https://github.com/badriadhikari/DEEPCON/')
        sys.exit(1)
    print ('')
    X, L, F, sequence = getX_asis(feat)
    L = len(sequence)
    if L < 20:
        print ("ERROR!! Too short sequence!!")
    if X.shape != (L, L, n_channels):
        print('Unexpected shape from cov21stats!')
        print(X.shape)
        sys.exit(1)
    X = X.reshape(1, L, L, n_channels)
    XX=np.full((1, 256, 256, n_channels), 0.0)
    if L >256:
        L=256
    XX[0, 0:L, 0:L, :] = X[:L, :L, :]
    L=256
    print ('Build a model of the size of the input (and not bigger)..')
    sys.stdout.flush()
    model = build_model(L, n_channels)
    print ('')
    print ('Load weights from ' + model_weights_file_name + '..')
    model.load_weights(model_weights_file_name)
    print ('')
    print ('Predict..')
    sys.stdout.flush()
    P1 = model.predict(XX)
    P2 = P1[0, 0:L, 0:L]
    P3 = np.zeros((L, L))
    for p in range(0, L):
        for q in range(0, L):
            P3[q, p] = (P2[q, p] + P2[p, q]) / 2.0
    print ('')
    print (('Write RR file ' + file_rr + '.. '))
    rr = open(file_rr, 'w')
    rr.write(sequence + "\n")
    for i in range(0, L):
        for j in range(i, L):
            if abs(i - j) < 5:
                continue
            rr.write("%i %i 0 8 %.5f\n" %(i+1, j+1, P3[i][j]))
    rr.close()
    print("Done " + str(sys.argv[0]) + " - " + str(datetime.datetime.now()))

################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat'
        , help = 'Input Feature file (obtained from step 1)'
        , required = True
    )
    parser.add_argument('--rr'
        , help = 'Output RR file (CASP format)'
        , required = True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    main(arguments['feat'], arguments['rr'])
