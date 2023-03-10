import tensorflow.keras as keras
import tensorflow as tf
from focal_loss import BinaryFocalLoss

import keras.backend as K
from keras import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import os

import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

import tensorflow as tf
#必须要下面这行代码
tf.compat.v1.disable_eager_execution()
print(tf.__version__)
 
#函数
def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))




class Metrics(Callback):
    def __init__(self, training_data,validation_data):
        self.validation_data = validation_data
        self.training_data = training_data

        
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.train_f1s = []
        self.train_recalls = []
        self.train_precisions = []

    def on_epoch_end(self,epoch, logs={}):
        #validation_data=(X[test], Y[test])

        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        val_targ = np.argmax(val_targ, axis=1)
        val_predict = np.argmax(val_predict, axis=1)

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        
        train_predict = (numpy.asarray(self.model.predict(
            self.training_data[0]))).round()
        train_targ = self.training_data[1]
        train_targ = np.argmax(train_targ, axis=1)
        train_predict = np.argmax(train_predict, axis=1)

        _train_f1 = f1_score(train_targ, train_predict)
        _train_recall = recall_score(train_targ, train_predict)
        _train_precision = precision_score(train_targ, train_predict)
        print('[-------------------._val_f1,_val_recall,_val_precision:',_val_f1,_val_recall,_val_precision,'----------------------]')
        print('[-------------------._train_f1,_train_recall,_train_precision:',_train_f1,_train_recall,_train_precision,'----------------------]')
        self.train_f1s.append(_train_f1)
        self.train_recalls.append(_train_recall)
        self.train_precisions.append(_train_precision)
        return
    

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    # index_best_model = hist_df['loss'].idxmin()
    index_best_model = hist_df['val_auc'].idxmax()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_auc',
                                          'best_model_val_auc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    # df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
    #                              columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
    #                                       'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_auc'] = row_best_model['auc']
    df_best_model['best_model_val_auc'] = row_best_model['val_auc']
    # df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    # df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses26396
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')
    plot_epochs_metric(hist, output_directory + 'epochs_recall.png',metric='recall')
    plot_epochs_metric(hist, output_directory + 'epochs_auc.png',metric='auc')
    plot_epochs_metric(hist, output_directory + 'epochs_precision.png',metric='precision')

    return df_metrics
#%%
'''
def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    # print('precision:',precision)
    return precision
    
def getRecall(y_true, y_pred):
    # print('np.shape(y_pred):',np.shape(y_pred))
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    # print('recall:',recall)
    return recallauc
'''

def create_mlp_model(input_shape):
    input_layer = keras.layers.Input(input_shape)
    # flatten/reshape because when multivariate all should be on the same axis 
    
    input_layer_flattened = keras.layers.Flatten()(input_layer)
    
    layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
    layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

    layer_2 = keras.layers.Dropout(0.2)(layer_1)
    layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

    layer_3 = keras.layers.Dropout(0.2)(layer_2)
    layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

    output_layer = keras.layers.Dropout(0.3)(layer_3)
    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
       
    auc = tf.keras.metrics.AUC() 
    recall = tf.keras.metrics.Recall()
    Precision = tf.keras.metrics.Precision()
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(learning_rate=0.001),
                  metrics=['accuracy',auc, recall, Precision])#"acc",
                   # metrics=[auc, recall, Precision, getRecall, getPrecision])#"acc",
            # metrics=[tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])#, keras_metrics.f1_score(), keras_metrics.precision(),keras_metrics.recall() ])
    return model

def create_cnn2D_model(input_shape):
    # padding = 'valid'
    input_layer = keras.layers.Input(input_shape)

    # if input_shape[0] < 60: # for italypowerondemand dataset
    padding = 'same'
    Droprate = 0.2
    regularrate = 0.01
    learning_rate = 0.0001
    conv1 = keras.layers.Conv2D(filters=16,kernel_size=(10,3),padding=padding,activation='relu',kernel_regularizer=keras.regularizers.l1(regularrate))(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.MaxPooling2D(pool_size=(10,3))(conv1)
    # conv1 = keras.layers.noise.GaussianNoise(0.1)(conv1)
    # conv1 = keras.layers.Dropout(Droprate)(conv1)

    conv2 = keras.layers.Conv2D(filters=32,kernel_size=(6,3),padding=padding,activation='relu',kernel_regularizer=keras.regularizers.l1(regularrate))(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.MaxPooling2D(pool_size=(6,3))(conv2)
    # conv2 = keras.layers.Dropout(Droprate)(conv2)
    
    conv3 = keras.layers.Conv2D(filters=64,kernel_size=(2,2),padding=padding,activation='relu',kernel_regularizer=keras.regularizers.l1(regularrate))(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv3)
    # conv3 = keras.layers.Dropout(Droprate)(conv3)
    

    flatten_layer = keras.layers.Flatten()(conv2)
    # GlobalAverage_layer = keras.layers.GlobalAveragePooling2D(conv3)
    # GlobalAverage_layer = keras.layers.Dropout(0.2)(GlobalAverage_layer)

    Dense_layer1 = keras.layers.Dense(units=32,activation='relu',kernel_regularizer=keras.regularizers.l2(regularrate))(flatten_layer)
    Dense_layer1 = keras.layers.Dropout(Droprate)(Dense_layer1)

    # Dense_layer2 = keras.layers.Dense(units=16,activation='relu')(Dense_layer1)
    # Dense_layer2 = keras.layers.Dropout(0.3)(Dense_layer2)

    # Dense_layer3 = keras.layers.Dense(units=8,activation='relu')(Dense_layer2)
    # Dense_layer3 = keras.layers.Dropout(0.3)(Dense_layer3)

    # Dense_layer4 = keras.layers.Dense(units=32,activation='relu')(Dense_layer3)
    # Dense_layer4 = keras.layers.Dropout(0.5)(Dense_layer4)
    
    
    output_layer = keras.layers.Dense(units=2,activation='softmax',kernel_regularizer=keras.regularizers.l2(regularrate))(Dense_layer1)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    # model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    #github.com/christinaversloot/mechine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
    #loss=BinaryFocalLoss(gamma=2)
    # BinaryFocalLoss(gamma=2)
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0),
                    metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    return model
    

from tensorflow import keras
from tensorflow.keras import layers

INPUT_SIZE = 224
CLASS_NUM = 2

# https://blog.csdn.net/liugan528/article/details/123719108
# stage_name=2,3,4,5; block_name=a,b,c
def ConvBlock(input_tensor, num_output, stride, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=stride, padding='same', name='res'+stage_name+block_name+'_branch2a')(input_tensor)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2a')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res'+stage_name+block_name+'_branch2b')(x)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2b')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2b_relu')(x)

    shortcut = layers.Conv2D(filter2, 1, strides=stride, padding='same', name='res'+stage_name+block_name+'_branch1')(input_tensor)
    shortcut = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch1')(shortcut)

    x = layers.add([x, shortcut], name='res'+stage_name+block_name)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_relu')(x)

    return x

def IdentityBlock(input_tensor, num_output, stage_name, block_name):
    filter1, filter2 = num_output

    x = layers.Conv2D(filter1, 3, strides=(1, 1), padding='same', name='res'+stage_name+block_name+'_branch2a')(input_tensor)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2a')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res'+stage_name+block_name+'_branch2b')(x)
    x = layers.BatchNormalization(name='bn'+stage_name+block_name+'_branch2b')(x)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_branch2b_relu')(x)

    shortcut = input_tensor

    x = layers.add([x, shortcut], name='res'+stage_name+block_name)
    x = layers.Activation('relu', name='res'+stage_name+block_name+'_relu')(x)

    return x

def create_resnet2D_model(input_shape, nb_classes):
    
    input = keras.Input(shape=input_shape, name='input')
    # conv1
    
    x = layers.Conv2D(12, 7, strides=(2, 2), padding='same', name='conv1')(input)  # 7×7, 64, stride 2  kernel_size=(3,3),
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)   # 3×3 max pool, stride 2

    # conv2_x
    x = ConvBlock(input_tensor=x, num_output=(12, 12), stride=(1, 1), stage_name='2', block_name='a')
    x = IdentityBlock(input_tensor=x, num_output=(12, 12), stage_name='2', block_name='b')

    # conv3_x
    # x = ConvBlock(input_tensor=x, num_output=(64, 64), stride=(2, 2), stage_name='3', block_name='a')
    # x = IdentityBlock(input_tensor=x, num_output=(64, 64), stage_name='3', block_name='b')

    # conv4_x
    # x = ConvBlock(input_tensor=x, num_output=(256, 256), stride=(2, 2), stage_name='4', block_name='a')
    # x = IdentityBlock(input_tensor=x, num_output=(256, 256), stage_name='4', block_name='b')

    # conv5_x
    # x = ConvBlock(input_tensor=x, num_output=(512, 512), stride=(2, 2), stage_name='5', block_name='a')
    # x = IdentityBlock(input_tensor=x, num_output=(512, 512), stage_name='5', block_name='b')

    # average pool, 1000-d fc, softmax
    x = keras.layers.AveragePooling2D((2, 2), strides=(1, 1))(x)#, name='pool5')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(nb_classes, activation='softmax', name='fc1000')(x)
   
    model = keras.Model(input, x, name='resnet2D')
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0),
                    metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    #model.summary()
    return model

# if __name__ == '__main__':
#     model = ResNet18((INPUT_SIZE, INPUT_SIZE, 3), CLASS_NUM)
#     print('Done.')
    
    

if __name__=="__main__":
    #%% read data
    x=np.array([[0,0],[1,0],[1,1],[0,1]])
    y=np.array([[0],[0],[1],[1]])
    
    switch_data_set = 3 #  
    file_path = '/data/preprocess_model_data/'
    # output_data_path = '/data/'
    out_result_path = '/data/model_result/minirocket_result/'
    .8345724907063197
    if switch_data_set == 0:
        train_test_path = file_path+ 'train_test_data/h5/'
        # # shape-less-data
        train_test_shape = [[[[3004,4321],[1288,4321]],[[3004,2401],[1288,2401]],[[3004,2401],[1288,2401]]],
        [[[6008,4321],[2576,4321]],[[6008,2401],[2576,2401]],[[6008,2401],[2576,2401]]]]
        NP_ratio = [1,3]

        
    elif switch_data_set == 1:
        train_test_path = file_path+ 'train_test_data/h5_120000/'
        #shape-more-data
        train_test_shape = [[[[16524,4321],[7082,4321]],[[16524,2401],[7082,2401]],[[16524,2401],[7082,2401]]],
        [[[31546,4321],[13520,4321]],[[31546,2401],[13520,2401]],[[31546,2401],[13520,2401]]],
        [[[61590,4321],[26396,4321]],[[61590,2401],[26396,2401]],[[61590,2401],[26396,2401]]]]
        NP_ratio = [10,20,40]
   
    elif switch_data_set == 2:
        train_test_path = file_path+ 'train_test_data/'
        thm_and_time_train_test_path = file_path+ 'train_test_data/'
        #shape-more-data
        train_test_shape = [[[[61254,4321],[20418,4321],[20418,4321]],[[61254,4321],[20418,4321],[20418,4321]],[[61254,4321],[20418,4321],[20418,4321]]]]
        thm_and_time_train_test_shape = 26396[[[61254,2],[20418,2],[20418,2]]]
        NP_ratio = [40]
        
    elif switch_data_set == 3:
        train_test_path = file_path+ 'train_test_data/'
        thm_and_time_train_test_path = file_path+ 'train_test_data/'
        #shape-more-data
        train_test_shape = [[[[65632,4321],[21878,4321],[21878,4321]],[[65632,4321],[21878,4321],[21878,4321]],[[65632,4321],[21878,4321],[21878,4321]]]]
        thm_and_time_train_test_shape = [[[65632,2],[21878,2],[21878,2]]]
        NP_ratio = [40]
    # h5_file_list = get_h5_file(train_test_path)

    predicted_data_path = '/data/preprocess_model_data/predicted_data/'
    train_test_transform_data_path = '/data/preprocess_model_data/train_test_transform_data/'
    # time1 = time.time()
        
    index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
    # 保存全部变量(18) all_data26396
    target_index_0 = ['Bx','By','Bz','B_theta','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','T_N_ratio']
    # 保存未计算的原始变量(10) initial_data
    target_index_1 = ['Bx','By','Bz','Ni','Ne','Ti','Te','Vx','Vy','Vz']
    # 保存判断bubble所依据的主要变量(10) judge_data
    target_index_2 = ['Bz','B_theta','Ni','Ne','Pm','Pp','Ti','Te','Vx_prep_B','T_N_ratio']
    target_index = [target_index_0,target_index_1,target_index_2]

    variance_type = ['all_var','initial_var','judge_var']
    normalized_type = ['non_normalized','max_min_normalized','mean_std_normalized']

    for i_NP_ratio in range(len(NP_ratio)-1,len(NP_ratio)):
        # for i_var in range(0,len(variance_type)-2):
        for i_var in range(0,len(variance_type)-2):
            for i_normalized in range(1,len(normalized_type)-1):
                train_sample_name = train_test_path+'train_data-add-2020-622-reevaluate-'+variance_type[i_var]+\
                    '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_NP_ratio][i_var][0][0])+\
                        '_'+str(train_test_shape[i_NP_ratio][i_var][0][1])+'-NP_ratio_'+str(NP_ratio[i_NP_ratio])+'.h5'

                test_sample_name = train_test_path+'test_data-add-2020-622-reevaluate-'+variance_type[i_var]+\
                    '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_NP_ratio][i_var][1][0])+\
                        '_'+str(train_test_shape[i_NP_ratio][i_var][1][1])+'-NP_ratio_'+str(NP_ratio[i_NP_ratio])+'.h5'
                
                val_sample_name = train_test_path+'validation_data-add-2020-622-reevaluate-'+variance_type[i_var]+\
                    '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_NP_ratio][i_var][1][0])+\
                        '_'+str(train_test_shape[i_NP_ratio][i_var][1][1])+'-NP_ratio_'+str(NP_ratio[i_NP_ratio])+'.h5'

                print('NP_ratio:',NP_ratio[i_NP_ratio])
                print('variance_type:',variance_type[i_var])
                print('normalized_type:',normalized_type[i_normalized])
                
                # model_name = 'cnn'
                model_name = 'resnet2D'


                output_directory='./model_result_train_test_validation_reevaluate/'+model_name+'_result_NP_ratio_'+str(NP_ratio[i_NP_ratio])+'/'
                
                y_train = pd.read_hdf(train_sample_name,key='df').values[:,0]
                y = y_train
                x_train = pd.read_hdf(train_sample_name,key='df').values[:,1:]
                # X_train_3D = np.reshape(X_train_inital,(np.shape(X_train_inital)[0],int(np.shape(X_train_inital)[1]/240),240))
                # X_train = convert(X_train_3D, from_type="numpy3D", to_type="pd-multiindex")
                y_test = pd.read_hdf(test_sample_name,key='df').values[:,0]
                x_test = pd.read_hdf(test_sample_name,key='df').values[:,1:]
                # X_test_3D = np.reshape(X_test_inital,(np.shape(X_test_inital)[0],int(np.shape(X_test_inital)[1]/240),240))
                # X_test = convert(X_test_3D, from_type="numpy3D", to_type="pd-multiindex")
                
                y_val = pd.read_hdf(val_sample_name,key='df').values[:,0]
                x_val = pd.read_hdf(val_sample_name,key='df').values[:,1:]
                
                nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

                # transform the labels from integers to one hot vectors
                enc = preprocessing.OneHotEncoder(categories='auto')
                enc.fit(np.concatenate((y_train, y_test,y_val), axis=0).reshape(-1, 1))
                y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
                y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
                y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

                # save orignal y because later we will use binary
                y_test_true = np.argmax(y_test, axis=1)
                y_val_true = np.argmax(y_val, axis=1)

                # if len(x_train.shape) == 2:  # if univariate
                #     # add a dimension to make it multivariate with one dimension 
                #     x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                #     x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
                # if len(x_train.shape) == 3:  # if univariate
                #     # add a dimension to make it multivariate with one dimension 
                #     x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
                #     x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],x_test.shape[2], 1))

                x_train = x_train.reshape(np.shape(x_train)[0],240,int(np.shape(x_train)[1]/240))
                x_test = x_test.reshape(np.shape(x_test)[0],240,int(np.shape(x_test)[1]/240))
                x_val = x_val.reshape(np.shape(x_val)[0],240,int(np.shape(x_val)[1]/240))
                if len(x_train.shape) == 2:  # if univariate
                    # add a dimension to make it multivariate with one dimension 
                    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
                    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
                if len(x_train.shape) == 3:  # if univariate
                    # add a dimension to make it multivariate with one dimension 
                    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
                    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],x_test.shape[2], 1))
                    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1],x_val.shape[2], 1))
 
                input_shape = x_train.shape[1:]
    
                nb_epochs = 100
                #%%
                start_time = time.time()
                # model = create_cnn1D_model(input_shape)
                # model = create_mlp_model(input_shape)
                model = create_resnet2D_model(input_shape,nb_classes)
                # tn_test, fp_test, fn_test, tp_test: 25604 144 131 517
                # tn_train,fp_train,fn_train,tp_train: 59909 183 66 1432
                # train_acuracy: 0.9959571358986848
                # train_recall: 0.9559412550066756
                # train_precision: 0.886687306501548
                # test_acuracy: 0.989581754811335
                # test_recall: 0.7978395061728395
                # test_precision: 0.7821482602118003
                
                #%% callback
                create_directory(output_directory)
                auc = tf.keras.metrics.AUC() 
                validation_data=(x_val, y_val)
                metrics = Metrics(training_data =(x_train,y_train),validation_data=(x_val,y_val))
                tf_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max',patience=5,restore_best_weights=True)#min_delta=0.00001,
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.000001)
                
                file_path = output_directory + 'best_model.hdf5'
                model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy',mode='max', 
                            save_best_only=True)
                tf_tensorboard = tf.keras.callbacks.TensorBoard('./tensorboard/'+ model_name +'/logs/NP_ratio_'+str(NP_ratio[i_NP_ratio]))
                #callbacks_list = [metrics] 
                callbacks=[metrics,model_checkpoint,tf_tensorboard,tf_earlystopping,reduce_lr]#,
                
                weight_array = compute_class_weight(class_weight='balanced',classes = np.unique(y),y=y)
                #weight_dict = dict(zip(np.unique(y),weight_array))
                weight_dict = {0.0: 1, 1.0: 10}
                # class_weight样本数多的类别权重低-20221105@fxd
                model.summary()
                flops = get_flops_params()
                print(flops)
                hist = model.fit(x_train, y_train,epochs=nb_epochs,
                    verbose=True, validation_data=(x_val,y_val),callbacks = callbacks)#,class_weight = weight_dict)#,class_weight = weight_dict)#, callbacks=self.callbacks)
                model.save(output_directory + 'last_model.hdf5')


                # model = keras.models.load_model(output_directory + 'best_model.hdf5')

                duration = time.time() - start_time

                y_train_pred = model.predict(x_train)
                y_train_pred = np.argmax(y_train_pred , axis=1)

                y_val_pred = model.predict(x_val)
                y_val_pred = np.argmax(y_val_pred , axis=1)
                
                y_test_pred = model.predict(x_test)
                y_test_pred = np.argmax(y_test_pred , axis=1)

                

                # print(y_val.T)
                # print(y_val_pred.T)
                tn_val, fp_val, fn_val, tp_val  = confusion_matrix(y_val[:,1],y_val_pred).ravel()
                tn_train,fp_train,fn_train,tp_train = confusion_matrix(y_train[:,1],y_train_pred).ravel()
                tn_test, fp_test, fn_test, tp_test  = confusion_matrix(y_test[:,1],y_test_pred).ravel()

                print("tn_train,fp_train,fn_train,tp_train:",tn_train,fp_train,fn_train,tp_train)
                print("tn_val, fp_val, fn_val, tp_val:",tn_val, fp_val, fn_val, tp_val)
                print("tn_test, fp_test, fn_test, tp_test:",tn_test, fp_test, fn_test, tp_test)

                train_accuracy = (tn_train+tp_train)/(tn_train+fp_train+fn_train+tp_train)
                train_recall = tp_train/(tp_train+fn_train)
                train_precision = tp_train/(tp_train+fp_train)
                
                val_accuracy = (tn_val+tp_val)/(tn_val+fp_val+fn_val+tp_val)
                val_recall = tp_val/(tp_val+fn_val)
                val_precision = tp_val/(tp_val+fp_val)
                
                test_accuracy = (tn_test+tp_test)/(tn_test+fp_test+fn_test+tp_test)
                test_recall = tp_test/(tp_test+fn_test)
                test_precision = tp_test/(tp_test+fp_test)
                
                target_names = ['non-bubble','bubble']
                # print(classification_report(y_test[:,1],y_pred,target_names = target_names))#该报告计算的精度和召回率也有问题
                print("train_accuracy:",train_accuracy)
                print("train_recall:",train_recall)
                print("train_precision:",train_precision)
                
                print("val_accuracy:",val_accuracy)
                print("val_recall:",val_recall)
                print("val_precision:",val_precision)
                
                print("test_accuracy:",test_accuracy)
                print("test_recall:",test_recall)
                print("test_precision:",test_precision)
                # print(y_val.T)
                print(y_val_pred.T)
                save_logs(output_directory, hist, y_val_pred, y_val_true, duration)
                
                