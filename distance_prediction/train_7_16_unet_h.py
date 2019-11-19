#dri Adhikari
# https://badriadhikari.github.io/
################################################################################

import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.models import load_model
import datetime
import keras.backend as K
epsilon = K.epsilon()
from io import BytesIO, StringIO
from tensorflow.python.lib.io import file_io

################################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)


flag_show_plots = False # True for Notebooks, False otherwise
if flag_show_plots:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

################################################################################
dirlocal = '../protein-distance/'
dirgcp = 'gs://protein-distance/'
dirpredictions = './predictions/' # only if building 3D models

dataset = 'full' # 'sample' or 'full'

stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
modelfile = 'model-' + str(stamp) + '.h5'

################################################################################
def determine_number_of_channels(input_features, pdb_list, length_dict):
    F = 0
    x = input_features[pdb_list[0]]
    l = length_dict[pdb_list[0]]
    for feature in x:
        if len(feature) == l:
            F += 2
        elif len(feature) == l * l:
            F += 1
        else:
            print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
            sys.exit(1)
    return F

################################################################################
def print_max_avg_sum_of_each_channel(x):
    print(' Channel        Avg        Max        Sum')
    for i in range(len(x[0, 0, :])):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i, a, m, s))

################################################################################
# Roll out 1D features to two 2D features, all to 256 x 256 (because many are smaller)
def prepare_input_features_2D(pdbs, input_features, distance_maps_cb, length_dict, F):
    X = np.full((len(pdbs), 256, 256, F), 0.0)
    Y = np.full((len(pdbs), 256, 256, 1), 100.0)
    for i, pdb in enumerate(pdbs):
        x = input_features[pdb]
        y = distance_maps_cb[pdb]
        l = length_dict[pdb]
        newi = 0
        xmini = np.zeros((l, l, F))
        for feature in x:
            feature = np.array(feature)
            feature = feature.astype(np.float)
            if len(feature) == l:
                for k in range(0, l):
                    xmini[k, :, newi] = feature
                    xmini[:, k, newi + 1] = feature
                newi += 2
            elif len(feature) == l * l:
                xmini[:, :, newi] = feature.reshape(l, l)
                newi += 1
            else:
                print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
                sys.exit(1)
        if l > 256:
            l = 256
        X[i, 0:l, 0:l, :] = xmini[:l, :l, :]
        Y[i, 0:l, 0:l, 0] = y[:l, :l]
    return X, Y

################################################################################
def plot_input_output_of_this_protein(X, Y):
    figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', frameon=True, edgecolor='k')
    for i in range(13):
        plt.subplot(7, 7, i + 1)
        plt.grid(None)
        plt.imshow(X[:, :, i], cmap='RdYlBu', interpolation='nearest')
    # Last plot is the true distance map
    plt.subplot(7, 7, 14)
    plt.grid(None)
    plt.imshow(Y[:, :], cmap='Spectral', interpolation='nearest')
    plt.show()

################################################################################
def calculate_mae(PRED, YTRUE, pdb_list, length_dict):
    plot_count = 0
    if flag_show_plots:
        plot_count = 4
    avg_mae = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        P = np.zeros((L, L))
        # Average the predictions from both triangles (optional)
        # This can improve MAE by upto 6% reduction
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 24:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_mae = 0.0
        for pair in top_pairs:
            abs_dist = abs(y_dict[pair] - p_dict[pair])
            sum_mae += abs_dist
        sum_mae /= L
        avg_mae += sum_mae
        print('MAE for ' + str(i) + ' - ' + str(pdb_list[i]) + ' = %.2f' % sum_mae)
        if plot_count > 0:
            plot_count -= 1
            for j in range(0, L):
                for k in range(0, L):
                    if not (j, k) in top_pairs:
                        P[j, k] = np.inf
                        Y[j, k] = np.inf
            for j in range(0, L):
                for k in range(j, L):
                    P[k, j] = Y[j, k]
            plt.grid(None)
            plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
            plt.show()
    print('Average MAE = %.2f' % (avg_mae / len(PRED[:, 0, 0, 0])))

################################################################################
#def calculate_longrange_contact_precision(PRED, YTRUE, pdb_list, length_dict):
#    if flag_show_plots:
 #       plot_count = 4
 #   avg_precision = 0.0
 #   for i in range(0, len(PRED[:, 0, 0, 0])):
 #       L = length_dict[pdb_list[i]]
 #       P = np.zeros((L, L))
 #       # Average the predictions from both triangles
  #      for j in range(0, L):
   #         for k in range(0, L):
    #            P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
     #   Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
      #  for j in range(0, L):
       #     for k in range(0, L):
        #        if k - j < 24:
         #           P[j, k] = np.inf
          #          Y[j, k] = np.inf
       # p_dict = {}
       # y_dict = {}
       # for j in range(0, L):
       #     for k in range(0, L):
       #         p_dict[(j,k)] = P[j, k]
       #         y_dict[(j,k)] = Y[j, k]
       # top_pairs = []
       # x = L
       # for pair in sorted(p_dict.items(), key=lambda x: x[1]):
       #     (k, v) = pair
       #     top_pairs.append(k)
       #     x -= 1
       #     if x == 0:
       #         break
       # sum_num=0
       # for pair in top_pairs:
       #     if y_dict[pair] >0 and y_dict[pair] < 8:
       #         sum_num+=1
       # precision = sum_num/L
       # avg_precision += precision
       # print('Precision for ' + str(i) + ' - ' + str(pdb_list[i]) +  ' ' + str(L) + ' [' + str(sum_num) + '/' + str(L) + '] = %.2f ' % precision)
       # plot_count = 0
        # Contact maps visualization of prediction against truth
        # Legend: lower triangle = true, upper triangle = prediction
       # if plot_count > 0:
        #    plot_count -= 1
         #   for j in range(0, L):
          #      for k in range(j, L):
           #         P[k, j] = Y[j, k]
           # plt.grid(None)
          #  plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
          #  plt.show()
   # print('Average Precision = %.2f' % (avg_precision / len(PRED[:, 0, 0, 0])))
def calculate_longrange_contact_precision(PRED, YTRUE, pdb_list, length_dict):
    if flag_show_plots:
        plot_count = 4
    avg_precision_L = 0.0
    avg_precision_M = 0.0
    avg_precision_S = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        P = np.zeros((L, L))
        # Average the predictions from both triangles
#precision long-range
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 24:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_num=0
        for pair in top_pairs:
            if y_dict[pair] >0 and y_dict[pair] < 8:
                sum_num+=1
        precision_L = sum_num/L
        avg_precision_L += precision_L
#precision medium-range
        P = np.zeros((L, L))
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j > 23 or k - j < 13:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_num=0
        for pair in top_pairs:
            if y_dict[pair] >0 and y_dict[pair] < 8:
                sum_num+=1
        precision_M = sum_num/L
        avg_precision_M += precision_M
#precision short-range
        P = np.zeros((L, L))
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 7 or k - j >12:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_num=0
        for pair in top_pairs:
            if y_dict[pair] >0 and y_dict[pair] < 8:
                sum_num+=1
        precision_S = sum_num/L
        avg_precision_S += precision_S
#        print('Precision for ' + str(i) + ' - ' + str(pdb_list[i]) +  ' ' + str(L) + ' [' + str(sum_num) + '/' + str(L) + '] = %.2f ' % precision_L)
        print('Precision for ' + str(i) + ' - ' + str(pdb_list[i]) +  ' ' + str(L) +' '+'Long-range'+' '+str(precision_L)+' '+'Medium-range'+' '+str(precision_M)+' '+'Short-range'+' '+str(precision_S))
        plot_count = 0
        plot_count = 0
        # Contact maps visualization of prediction against truth
        # Legend: lower triangle = true, upper triangle = prediction
        if plot_count > 0:
            plot_count -= 1
            for j in range(0, L):
                for k in range(j, L):
                    P[k, j] = Y[j, k]
            plt.grid(None)
            plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
            plt.show()
    print('Average Long-range Precision = %.2f' % (avg_precision_L / len(PRED[:, 0, 0, 0])))
    print('Average Medium-range Precision = %.2f' % (avg_precision_M / len(PRED[:, 0, 0, 0])))
    print('Average Short-range Precision = %.2f' % (avg_precision_S / len(PRED[:, 0, 0, 0])))



def write_predictions_to_file(PRED,YTRUE, pdb_list, length_dict, dirpredictions):
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        P = np.zeros((L, L))
        # Average the predictions from both triangles (optional)
        # This can improve MAE by upto 6% reduction
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        f = open(dirpredictions + pdb_list[i] + '.dmap', 'w')
        for j in range(0, L):
            for k in range(0, L):
                f.write(str(j)+' '+str(k)+' '+str(P[j, k]) +' '+str(YTRUE[i,j,k,0])+'\n')
#                f.write(P[j, k] + ' ')
 #           f.write('\n')    
        f.close()

def main():
    ################################################################################
    print('')
    print('Load input features..')
    x = dirlocal + dataset + '-input-features.npy'
    if not os.path.isfile(x):
        x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-input-features.npy', binary_mode=True))
    (pdb_list, length_dict, input_features) = np.load(x, encoding='latin1')

    ################################################################################
    print('')
    print('Load distance maps..')
    x = dirlocal + dataset + '-distance-maps-cb.npy'
    if not os.path.isfile(x):
        x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-distance-maps-cb.npy', binary_mode=True))
    (pdb_list_y, distance_maps_cb) = np.load(x, encoding='latin1')

    ################################################################################
    print('')
    print ('Some cross checks on data loading..')
    for pdb in pdb_list:
        if not pdb in pdb_list_y:
            print ('I/O mismatch ', pdb)
            sys.exit(1)

    ################################################################################
    print('')
    print('Find the number of input channels..')
    F = determine_number_of_channels(input_features, pdb_list, length_dict)

    ################################################################################
    print('')
    print('Split into training and validation set (4%)..')
    split = int(0.0 * len(pdb_list))
    split2 = int(0.10 * len(pdb_list))
    split3 = int(0.70 * len(pdb_list))
    valid_pdbs = pdb_list[split:split2]
    train_pdbs = pdb_list[split2:]

    print('Total validation proteins = ', len(valid_pdbs))
    print('Total training proteins = ', len(train_pdbs))

    ################################################################################
    print('')
    print ('Prepare the validation input and outputs..')
    XVALID, YVALID = prepare_input_features_2D(valid_pdbs, input_features, distance_maps_cb, length_dict, F)
    print(XVALID.shape)
    print(YVALID.shape)

    print('')
    print ('Prepare the training input and outputs..')
    XTRAIN, YTRAIN = prepare_input_features_2D(train_pdbs, input_features, distance_maps_cb, length_dict, F)
    print(XTRAIN.shape)
    print(YTRAIN.shape)

    ################################################################################
    print('')
    print('Sanity check input features values..')
    print(' First validation protein:')
    print_max_avg_sum_of_each_channel(XVALID[0, :, :, :])
    print(' First traininig protein:')
    print_max_avg_sum_of_each_channel(XTRAIN[0, :, :, :])

    ################################################################################
    if flag_show_plots:
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        for i in range(4):
            print('')
            L = length_dict[valid_pdbs[i]]
            plot_input_output_of_this_protein(XVALID[i, 0:L, 0:L, :], YVALID[i, 0:L, 0:L, 0])

    ################################################################################
    print('')
    print('Build a model..')
    input = Input(shape = (256, 256, F))
    tower = BatchNormalization()(input)
    tower = Activation('relu')(tower)
    tower1 = Convolution2D(256, 3, padding = 'same')(tower)
    tower1 = LeakyReLU(alpha=0.05)(tower1)
    tower1=Dropout(0.05)(tower1)
    #tower1 = Activation('relu')(tower1)
    # tower1 = BatchNormalization()(tower1)
    tower2 = Convolution2D(128, 3, padding = 'same')(tower1)
    tower2 = LeakyReLU(alpha=0.05)(tower2)
    tower2=Dropout(0.05)(tower2)
#    tower2 = Activation('relu')(tower2)
    #tower2 = BatchNormalization()(tower2)
    d_rate=1
    tower3 = Convolution2D(64, 3, dilation_rate=(d_rate,d_rate),padding = 'same')(tower2)
#    tower3 = Activation('relu')(tower3)
    tower3 = LeakyReLU(alpha=0.05)(tower3)
    tower3=Dropout(0.05)(tower3)
    d_rate=2
    #tower3 = BatchNormalization()(tower3)
    tower4 = Convolution2D(64, 3,dilation_rate=(d_rate,d_rate),padding = 'same')(tower3)
    tower4 = LeakyReLU(alpha=0.05)(tower4)
    tower4=Dropout(0.05)(tower4)
#    tower4 = Activation('relu')(tower4)
    #tower4 = BatchNormalization()(tower4)
    d_rate=4
    tower5 = Convolution2D(64, 3,dilation_rate=(d_rate,d_rate), padding = 'same')(tower4)
    tower5 = LeakyReLU(alpha=0.05)(tower5)
    tower5=Dropout(0.05)(tower5)
#    tower5 = Activation('relu')(tower5)
    #tower5 = BatchNormalization()(tower5)
    tower6 = Convolution2D(32, 3, padding = 'same')(tower5)
    tower6 = LeakyReLU(alpha=0.05)(tower6)
    tower6=Dropout(0.05)(tower6)
#    tower6 = Activation('relu')(tower6)
    #tower4 = BatchNormalization()(tower4)
    tower7 = Convolution2D(32, 3, padding = 'same')(tower6)
    tower7 = LeakyReLU(alpha=0.05)(tower7)
    tower7=Dropout(0.05)(tower7)
#    tower7 = Activation('relu')(tower7)
    #tower5 = BatchNormalization()(tower5)


    tower7_1 = Deconvolution2D(32, 3, padding = 'same')(tower7)
    tower7_1 = Activation('relu')(tower7_1)
    tower7_1 = concatenate([tower7_1,tower6],axis=3)
    #tower5_1 = BatchNormalization()(tower5_1)
    tower6_1 = Deconvolution2D(32, 3, padding = 'same')(tower7_1)
    tower6_1 = Activation('relu')(tower6_1)
    tower6_1 = concatenate([tower6_1,tower5],axis=3)
    #tower4_1 = BatchNormalization()(tower4_1)
    tower5_1 = Deconvolution2D(64, 3, padding = 'same')(tower6_1)
    tower5_1 = Activation('relu')(tower5_1)
    tower5_1 = concatenate([tower5_1,tower4],axis=3)
    #tower5_1 = BatchNormalization()(tower5_1)
    tower4_1 = Deconvolution2D(64, 3, padding = 'same')(tower5_1)
    tower4_1 = Activation('relu')(tower4_1)
    tower4_1 = concatenate([tower4_1,tower3],axis=3)
    #tower4_1 = BatchNormalization()(tower4_1)
    tower3_1 = Deconvolution2D(64, 3, padding = 'same')(tower4_1)
    tower3_1 = Activation('relu')(tower3_1)
    tower3_1 = concatenate([tower3_1,tower2],axis=3)
    #tower3_1 = BatchNormalization()(tower3_1)
    tower2_1 = Deconvolution2D(128, 3, padding = 'same')(tower3_1)
    tower2_1 = Activation('relu')(tower2_1)
    tower2_1 = concatenate([tower2_1,tower1],axis=3)
    #tower2_1 = BatchNormalization()(tower2_1)
    tower = Deconvolution2D(256, 3, padding = 'same')(tower2_1)
    tower = Activation('relu')(tower)
    tower=Dense(32,activation='relu')(tower)
    tower=Dense(1,activation='relu')(tower)
    model = Model(input, tower)
    def male_loss_hh(gamma=2, alpha=0.75):
        def male_loss_fixed(y_true, y_pred):
            #pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            #pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    #        pt_1=tf.where(y_true<16, 1, 0)
     #       pt_2=tf.where(y_true<16, y_true, 16)
#            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
            t1=K.log(K.clip(y_pred,K.epsilon(),None)+1.)
            t2=K.log(K.clip(y_true,K.epsilon(),None)+1.)
            return K.mean(K.abs((t1-t2)/K.clip(K.abs(y_true/8+0.5),K.epsilon(),None)),axis=-1)
        return male_loss_fixed
#    model.compile(loss = 'logcosh', optimizer = 'rmsprop', metrics = ['mae'])
    model.compile(optimizer = 'rmsprop', loss =[male_loss_hh(gamma=2,alpha=0.25)], metrics = ['mae'])
    ################################################################################
#    model.compile(loss = 'msle', optimizer = 'rmsprop', metrics = ['mae'])
    print (model.summary())

    ################################################################################
    # a simple early stopping
    mc = ModelCheckpoint(modelfile, monitor = 'loss', mode = 'min', verbose = 1, save_best_only = True)
    es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 2, patience = 50)
    print('')
    print('Train the model..')
    history = model.fit(XTRAIN, YTRAIN, verbose = 1, batch_size = 2, epochs = 40, validation_data=(XVALID, YVALID), callbacks=[es, mc])

    ################################################################################
    print('')
    print('Cuves..')
    if flag_show_plots:
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        print(history.params)
        plt.clf()
        plt.plot(history.history['mean_absolute_error'], 'g', label='Training MAE')
        plt.plot(history.history['val_mean_absolute_error'], 'b', label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

    ################################################################################
    print('')
    print('Load the best weights..')
    model = load_model(modelfile, compile = False)

    ################################################################################
    print('')
    print('Make predictions..')
    P = model.predict(XVALID)
    print('')
    print('Compare the predictions with the truths (for some proteins) ..')
    if flag_show_plots:
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', frameon=True, edgecolor='k')
        I = 1
        for k in range(4):
            L = length_dict[pdb_list[k]]
            plt.subplot(4, 4, I)
            I += 1
            plt.grid(None)
            plt.imshow(P[k, 0:L, 0:L, 0], cmap='RdYlBu', interpolation='nearest')
        for k in range(4):
            L = length_dict[pdb_list[k]]
            plt.subplot(4, 4, I)
            I += 1
            plt.grid(None)
            plt.imshow(YVALID[k, 0:L, 0:L, 0], cmap='Spectral', interpolation='nearest')
        plt.show()

    ################################################################################
    print('')
    print('MAE of top L long-range distance predictions on the validation set..')
    calculate_mae(P, YVALID, valid_pdbs, length_dict)
    print('')
    print('Precision of top L long-range distance predictions on the validation set..')
    calculate_longrange_contact_precision(P, YVALID, valid_pdbs, length_dict)

    ################################################################################
    print('')
    print('Evaluate on the test dataset..')
    model = load_model(modelfile, compile = False)
    x = dirlocal + 'testset-input-features.npy'
    if not os.path.isfile(x):
        x = BytesIO(file_io.read_file_to_string(dirgcp + 'testset-input-features.npy', binary_mode=True))
    (pdb_list, length_dict,sequence_dict, input_features)  = np.load(x)
    x = dirlocal + 'testset-distance-maps-cb.npy'
    if not os.path.isfile(x):
        x = BytesIO(file_io.read_file_to_string(dirgcp + 'testset-distance-maps-cb.npy', binary_mode=True))
    (pdb_list_y, distance_maps_cb) = np.load(x)
    F = determine_number_of_channels(input_features, pdb_list, length_dict)
    XTEST, YTEST = prepare_input_features_2D(pdb_list, input_features, distance_maps_cb, length_dict, F)
    P = model.predict(XTEST)
    for pdb in length_dict:
        if length_dict[pdb] > 256:
            length_dict[pdb] = 256
    print('')
    print('MAE of top L long-range distance predictions on the test set..')
    calculate_mae(P, YTEST, pdb_list, length_dict)
    print('')
    print('Precision of top L long-range distance predictions on the test set..')
    calculate_longrange_contact_precision(P, YTEST, pdb_list, length_dict)
    # Only if building 3D models
    #write_predictions_to_file(P,YTEST, pdb_list, length_dict, dirpredictions)


################################################################################
if __name__ == "__main__":
    main()
