import tensorflow.keras as keras
import tensorflow as tf
from focal_loss import BinaryFocalLoss
import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Conv1D,MaxPooling1D
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

#%%
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
        # val_predict = np.argmax(val_predict, axis=1)

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
        # train_predict = np.argmax(train_predict, axis=1)

        _train_f1 = f1_score(train_targ, train_predict)
        _train_recall = recall_score(train_targ, train_predict)
        _train_precision = precision_score(train_targ, train_predict)
        print('[-------------------._val_f1,_val_recall,_val_precision:',_val_f1,_val_recall,_val_precision,'----------------------]')
        print('[-------------------._train_f1,_train_recall,_train_precision:',_train_f1,_train_recall,_train_precision,'----------------------]')
        self.train_f1s.append(_train_f1)
        self.train_recalls.append(_train_recall)
        self.train_precisions.append(_train_precision)
        return
#%%
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

#%%
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

    index_best_model = hist_df['loss'].idxmin()
    # index_best_model = hist_df['val_auc'].idxmax()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_auc',
                                          'best_model_val_auc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    # df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
    #                              columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
    #                                       'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    # df_best_model['best_model_train_auc'] = row_best_model['auc']
    # df_best_model['best_model_val_auc'] = row_best_model['val_auc']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png',metric='loss')
    plot_epochs_metric(hist, output_directory + 'epochs_accuracy.png',metric='accuracy')
    plot_epochs_metric(hist, output_directory + 'epochs_recall.png',metric='recall')
    plot_epochs_metric(hist, output_directory + 'epochs_precision.png',metric='precision')
    plot_epochs_metric(hist, output_directory + 'epochs_auc.png',metric='auc')

    return df_metrics
#%%
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

    conv1 = keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding=padding,activation='relu')(input_layer)
    conv1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv1 = keras.layers.Dropout(0.2)(conv1)

    conv2 = keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding=padding,activation='relu')(conv1)
    conv2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    
    # conv3 = keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding=padding,activation='relu')(conv2)
    # conv3 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv3)
    # conv3 = keras.layers.Dropout(0.2)(conv3)

    flatten_layer = keras.layers.Flatten()(conv2)
    # GlobalAverage_layer = keras.layers.GlobalAveragePooling2D(conv3)
    # GlobalAverage_layer = keras.layers.Dropout(0.2)(GlobalAverage_layer)

    Dense_layer1 = keras.layers.Dense(units=64,activation='relu')(flatten_layer)
    Dense_layer2 = keras.layers.Dense(units=32,activation='relu')(Dense_layer1)
    output_layer = keras.layers.Dense(units=2,activation='relu')(Dense_layer2)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    # model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    #github.com/christinaversloot/mechine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0),
                    metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    return model


def create_lstm_model(input_shape):
    # padding = 'valid'
    model = Sequential()
    model.add(LSTM(units = 128, activation = 'tanh', return_sequences = True, input_shape=input_shape))
    model.add(Dropout(0.5))
    # model.add(LSTM(units = 64, activation = 'tanh', return_sequences = True))
    # model.add(Dropout(0.4))
    model.add(LSTM(units = 64, activation = 'tanh', return_sequences = False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    #               metrics=['accuracy'])
    #github.com/christinaversloot/mechine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0),
    #                 metrics=['accuracy',tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    return model

    
if __name__=="__main__":
    #%% read data
    x=np.array([[0,0],[1,0],[1,1],[0,1]])
    y=np.array([[0],[0],[1],[1]])
    
    switch_data_set = 1 #  
    file_path = '/data/preprocess_model_data/'
    # output_data_path = '/data/'
    out_result_path = '/data/model_result/minirocket_result/'
    
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

    # h5_file_list = get_h5_file(train_test_path)

    predicted_data_path = '/data/preprocess_model_data/predicted_data/'
    train_test_transform_data_path = '/data/preprocess_model_data/train_test_transform_data/'
    # time1 = time.time()
        
    index =          ['time_B','Bx','By','Bz','Ni','Ne','plasma_beta','Pm','Pp','Ti','Te','Vx','Vy','Vz','Vx_prep_B','Vy_prep_B','Vz_prep_B','Pos_X','Pos_Y','Pos_Z']
    # 保存全部变量(18) all_data
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
        for i_var in range(2,len(variance_type)):
            for i_normalized in range(1,len(normalized_type)-1):
                train_sample_name = train_test_path+'train_data-'+variance_type[i_var]+\
                    '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_NP_ratio][i_var][0][0])+\
                        '_'+str(train_test_shape[i_NP_ratio][i_var][0][1])+'-NP_ratio_'+str(NP_ratio[i_NP_ratio])+'.h5'

                test_sample_name = train_test_path+'test_data-'+variance_type[i_var]+\
                    '-'+normalized_type[i_normalized]+'-shape_'+str(train_test_shape[i_NP_ratio][i_var][1][0])+\
                        '_'+str(train_test_shape[i_NP_ratio][i_var][1][1])+'-NP_ratio_'+str(NP_ratio[i_NP_ratio])+'.h5'

                print('NP_ratio:',NP_ratio[i_NP_ratio])
                print('variance_type:',variance_type[i_var])
                print('normalized_type:',normalized_type[i_normalized])
                
                # model_name = 'cnn'
                #model_name = 'mlp'
                model_name = 'lstm'
                #model_name = 'lstm_cnn'

                output_directory='./model_result/'+model_name+'_result_NP_ratio_'+str(NP_ratio[i_NP_ratio])+'/'
                
                y_train = pd.read_hdf(train_sample_name,key='df').values[:,0]
                y = y_train
                x_train = pd.read_hdf(train_sample_name,key='df').values[:,1:]
                # X_train_3D = np.reshape(X_train_inital,(np.shape(X_train_inital)[0],int(np.shape(X_train_inital)[1]/240),240))
                # X_train = convert(X_train_3D, from_type="numpy3D", to_type="pd-multiindex")
                y_test = pd.read_hdf(test_sample_name,key='df').values[:,0]
                x_test = pd.read_hdf(test_sample_name,key='df').values[:,1:]
                # X_test_3D = np.reshape(X_test_inital,(np.shape(X_test_inital)[0],int(np.shape(X_test_inital)[1]/240),240))
                # X_test = convert(X_test_3D, from_type="numpy3D", to_type="pd-multiindex")
                
                nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

                # transform the labels from integers to one hot vectors
                enc = preprocessing.OneHotEncoder(categories='auto')
                enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
                y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
                y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

                # save orignal y because later we will use binary
                y_true = np.argmax(y_test, axis=1)

                x_train = x_train.reshape(np.shape(x_train)[0],240,int(np.shape(x_train)[1]/240))
                x_test = x_test.reshape(np.shape(x_test)[0],240,int(np.shape(x_test)[1]/240))

                if len(x_train.shape) == 2:  # if univariate
                    # add a dimension to make it multivariate with one dimension 
                    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
                if len(x_train.shape) == 3:  # if univariate
                    # add a dimension to make it multivariate with one dimension 
                    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
                    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],x_test.shape[2], 1))
                    
                x_val = x_test
                y_val = y_test
               # input_shape = x_train.shape[1:]
                # input_length = x_train.shape[1]
                input_length_lstm = x_train.shape[1:3]
                nb_epochs = 500
                #%%
                start_time = time.time()
                # model = create_cnn1D_model(input_shape)
                # model = create_mlp_model(input_shape)
                #model = create_cnn2D_model(input_shape)
                model = create_lstm_model(input_length_lstm)
                #%% callback
                create_directory(output_directory)
                auc = tf.keras.metrics.AUC() 
                validation_data=(x_val, y_val)
                metrics = Metrics(training_data =(x_train,y_train),validation_data=(x_val,y_val))
                tf_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='accuracy',mode='max',patience=10)
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5, min_lr=0.000001)
                model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='accuracy',mode='max', 
                            save_best_only=True)
                tf_tensorboard = tf.keras.callbacks.TensorBoard('./tensorboard/'+ model_name +'/logs/NP_ratio_'+str(NP_ratio[i_NP_ratio]))
                #callbacks_list = [metrics] 
                callbacks=[metrics,tf_earlystopping,model_checkpoint,tf_tensorboard,reduce_lr]
                
                weight_array = compute_class_weight(class_weight='balanced',classes = np.unique(y),y=y)
                weight_dict = dict(zip(np.unique(y),weight_array))
                # class_weight样本数多的类别权重低-20221105@fxd
                hist = model.fit(x_train, y_train[:,1],epochs=nb_epochs,
                    verbose=True, validation_data=(x_val,y_val[:,1]),callbacks = callbacks)#,class_weight = weight_dict)#, callbacks=self.callbacks)
                duration = time.time() - start_time

                y_pred = model.predict(x_val)
                y_pred = np.argmax(y_pred , axis=1)

                y_train_predict = model.predict(x_train)
                y_test_predict = model.predict(x_test)
                y_train_predict = np.argmax(y_train_predict , axis=1)
                y_test_predict = np.argmax(y_test_predict , axis=1)

                # print(y_val.T)
                # print(y_pred.T)
                tn_test, fp_test, fn_test, tp_test  = confusion_matrix(y_val[:,1],y_pred).ravel()
                tn_train,fp_train,fn_train,tp_train = confusion_matrix(y_train[:,1],y_train_predict).ravel()

                print("tn_test, fp_test, fn_test, tp_test:",tn_test, fp_test, fn_test, tp_test)
                print("tn_train,fp_train,fn_train,tp_train:",tn_train,fp_train,fn_train,tp_train)
                
                train_accuracy = (tn_train+tp_train)/(tn_train+fp_train+fn_train+tp_train)
                train_recall = tp_train/(tp_train+fn_train)
                train_precision = tp_train/(tp_train+fp_train)
                
                test_accuracy = (tn_test+tp_test)/(tn_test+fp_test+fn_test+tp_test)
                test_recall = tp_test/(tp_test+fn_test)
                test_precision = tp_test/(tp_test+fp_test)
                target_names = ['non-bubble','bubble']
                # print(classification_report(y_val[:,1],y_pred,target_names = target_names))#该报告计算的精度和召回率也有问题
                print("train_acuracy:",train_accuracy)
                print("train_recall:",train_recall)
                print("train_precision:",train_precision)
                
                print("test_acuracy:",test_accuracy)
                print("test_recall:",test_recall)
                print("test_precision:",test_precision)
                # print(y_val.T)
                print(y_pred.T)
                save_logs(output_directory, hist, y_pred, y_true, duration)