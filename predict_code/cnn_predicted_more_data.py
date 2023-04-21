import numpy as np
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
# 2022-10-17@fxd
from sklearn.linear_model import LogisticRegressionCV
from focal_loss import BinaryFocalLoss

from sklearn.metrics import classification_report, confusion_matrix


from sktime.datatypes import convert
from pyspedas import time_double
from pyspedas import time_string

import tensorflow.keras as keras

import pandas as pd
import math
import time
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def get_h5_file(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith("h5") or file.endswith("hdf5"):
                file_list.append(os.path.join(root, file))


model_dir = '/home/dell/code/project1_MultiTS/model_result_train_test_validation_reevaluate/cnn_result_NP_ratio_40/val_87_87_test_84_82_choose/'
out_result_path = '/data/model_result/neural_network_result/'

file_path = '/data/preprocess_model_data/'
data_file_path = file_path+'/train_test_data/h5_120000/'
# output_data_path = '/data/'
train_test_path = file_path+ 'train_test_data/'
# h5_file_list = get_h5_file(train_test_path)

predicted_data_path = '/data/preprocess_model_data/predicted_data/'
train_test_transform_data_path = '/data/preprocess_model_data/train_test_transform_data/'
# time1 = time.time()
# store = pd.HDFStore(h5_file_list[9],mode='r')
# df1 = store.get('df')
# #store.close()

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

# shape
# train_test_shape = [[[[16524,4321],[7082,4321]],[[16524,2401],[7082,2401]],[[16524,2401],[7082,2401]]],
# [[[31546,4321],[13520,4321]],[[31546,2401],[13520,2401]],[[31546,2401],[13520,2401]]],
# [[[61590,4321],[26396,4321]],[[61590,2401],[26396,2401]],[[61590,2401],[26396,2401]]]]
# train_test_shape = [[[[61254,4321],[20418,4321],[20418,4321]],[[61254,4321],[20418,4321],[20418,4321]],[[61254,4321],[20418,4321],[20418,4321]]]]
# thm_and_time_train_test_shape = [[[61254,2],[20418,2],[20418,2]]]
train_test_shape = [[[[65632,4321],[21878,4321],[21878,4321]],[[65632,4321],[21878,4321],[21878,4321]],[[65632,4321],[21878,4321],[21878,4321]]]]
thm_and_time_train_test_shape = [[[65632,2],[21878,2],[21878,2]]]
# NP_ratio = [40]


# 1
#    all_var
#         non_normalized             train,test:[3004,4321],[1288,4321]-
#         max_min_normalized         train,test:[3004,4321],[1288,4321]
#         mean_std_normalized        train,test:[3004,4321],[1288,4321]
#    initial_var
#         non_normalized             train,test:[3004,2401],[1288,2401]-
#         max_min_normalized         train,test:[3004,2401],[1288,2401]  
#         mean_std_normalized        train,test:[3004,2401],[1288,2401]
#    judge_var
#         non_normalized             train,test:[3004,2401],[1288,2401]-
#         max_min_normalized         train,test:[3004,2401],[1288,2401]
#         mean_std_normalized        train,test:[3004,2401],[1288,2401]
# 3
#    all_var
#         non_normalized             train,test:[6008,4321],[2576,4321]-
#         max_min_normalized         train,test:[6008,4321],[2576,4321]
#         mean_std_normalized        train,test:[6008,4321],[2576,4321]
#    initial_var
#         non_normalized             train,test:[6008,2401],[2576,2401]-
#         max_min_normalized         train,test:[6008,2401],[2576,2401]
#         mean_std_normalized        train,test:[6008,2401],[2576,2401]
#    judge_var
#         non_normalized             train,test:[6008,2401],[2576,2401]-
#         max_min_normalized         train,test:[6008,2401],[2576,2401]
#         mean_std_normalized        train,test:[6008,2401],[2576,2401]
#%% 正负样本比例negative_positive_ratio
# NP_ratio = [1,3]
NP_ratio = [40]
minirocket_score_list = []
minirocket_time_list = []
rocket_score_list = []
rocket_time_list = []
result_output = pd.DataFrame(columns=['minirocket_score','minirocket_time'])#,'rocket_score','rocket_time'])
all_target_index = [target_index_0,target_index_1,target_index_2]
#j = 0

#%% predicted data 
year_list = ['2021']
thm = ['tha','thb','thc','thd','the']
step = [240]#[60,240]#3分钟为窗口的步长
stepname =['12min']#['3min','12min']

bubble_time_list_do_nothing = pd.DataFrame(columns= ["satellite","starttime","endtime"])

new_row = {'satellite':'','starttime':'','endtime':''}
new_row_satellite = []#={'satellite':'','starttime':'','endtime':''}
new_row_starttime = []
new_row_endtime = []
#new_row_do_nothing ={'satellite':'','starttime':'','endtime':''}
new_row_do_nothing_satellite = []
new_row_do_nothing_starttime = []
new_row_do_nothing_endtime = []
bubble_length = 12*60

#%%
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
            print('train_sample_name:',train_sample_name)
            print('test_sample_name:',test_sample_name)
            print('val_sample_name:',val_sample_name)

            y_train = pd.read_hdf(train_sample_name,key='df').values[:,0]
            X_train = pd.read_hdf(train_sample_name,key='df').values[:,1:]
            y_test = pd.read_hdf(test_sample_name,key='df').values[:,0]
            X_test = pd.read_hdf(test_sample_name,key='df').values[:,1:]
            y_val = pd.read_hdf(val_sample_name,key='df').values[:,0]
            X_val = pd.read_hdf(val_sample_name,key='df').values[:,1:]
            #%% 
            model_path = model_dir + '/best_model.hdf5'#ResNet-non-normalized
            #model_path = model_dir + '/FCN/non_normalized/'+'best_model.hdf5'#FCN-non-normalized
            model = keras.models.load_model(model_path)
            model.summary()
            y_train_predict = model.predict(X_train)            
            y_test_predict = model.predict(X_test)  
            y_val_predict = model.predict(X_val)  

            y_train_predict = np.argmax(y_train_predict , axis=1)
            y_test_predict = np.argmax(y_test_predict , axis=1)
            y_val_predict = np.argmax(y_val_predict , axis=1)

            # check_raise(X_train, mtype="np.ndarray")
            # 多变量时间序列数据


            tn_test, fp_test, fn_test, tp_test  = confusion_matrix(y_test,y_test_predict).ravel()
            tn_train,fp_train,fn_train,tp_train = confusion_matrix(y_train,y_train_predict).ravel()
            tn_val,fp_val,fn_val,tp_val = confusion_matrix(y_val,y_val_predict).ravel()

            print("tn_test, fp_test, fn_test, tp_test:",tn_test, fp_test, fn_test, tp_test)
            print("tn_train,fp_train,fn_train,tp_train:",tn_train,fp_train,fn_train,tp_train)
            print("tn_val,fp_val,fn_val,tp_val:",tn_val,fp_val,fn_val,tp_val)

            train_accuracy = (tn_train+tp_train)/(tn_train+fp_train+fn_train+tp_train)
            train_recall = tp_train/(tp_train+fn_train)
            train_precision = tp_train/(tp_train+fp_train)
            
            val_accuracy = (tn_val+tp_val)/(tn_val+fp_val+fn_val+tp_val)
            val_recall = tp_val/(tp_val+fn_val)
            val_precision = tp_val/(tp_val+fp_val)
            
            test_accuracy = (tn_test+tp_test)/(tn_test+fp_test+fn_test+tp_test)
            test_recall = tp_test/(tp_test+fn_test)
            test_precision = tp_test/(tp_test+fp_test)
            
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
            # mlp result NP-20:
            # tn_test, fp_test, fn_test, tp_test: 25590 158 260 388
            # tn_train,fp_train,fn_train,tp_train: 59746 346 584 914
            
            for i_step in range(0,len(step)):
                # if i_var==0 and i_step ==0:
                #     continue
                print("-------step:",step[i_step],"--------")
                bubble_time_list = pd.DataFrame(columns= ["satellite","starttime","endtime"])
                all_event_time_list = pd.DataFrame(columns= ["satellite","starttime","endtime","bubble_index"])

                new_row_satellite = []#={'satellite':'','starttime':'','endtime':''}
                new_row_starttime = []
                new_row_endtime = []
                #new_row_do_nothing ={'satellite':'','starttime':'','endtime':''}
                new_row_do_nothing_satellite = []
                new_row_do_nothing_starttime = []
                new_row_do_nothing_endtime = []
                new_row_do_nothing_bubble_index = []


                print('---------step:',step[i_step],'--------------')
                #for i_year in range(0,len(year_list)-1):#test:2020
                for i_year in range(0,len(year_list)):#test:2020
                    print("-------year:",year_list[i_year],"--------")
                    year = year_list[i_year]
                    source_file_time = [year+'-01-01-00-00-00',year+'-02-01-00-00-00',year+'-03-01-00-00-00',year+'-04-01-00-00-00',\
                        year+'-05-01-00-00-00',year+'-06-01-00-00-00',year+'-07-01-00-00-00',year+'-08-01-00-00-00',\
                        year+'-09-01-00-00-00',year+'-10-01-00-00-00',year+'-11-01-00-00-00',year+'-12-01-00-00-00']
                    predicted_data_full_path = predicted_data_path+year+'/'
                    for i_thm in range(0,len(thm)):#test:tha
                        print("-------thm:",thm[i_thm],"--------")
                        for i_month in range(0,len(source_file_time)):#读取每月数据
                            print("-------i_month:",i_month+1,"--------")

                            #for i_month in range(4,5):#读取每月数据
                            outputfilename = thm[i_thm]+'-'+year_list[i_year]+'-'+str(i_month+1)+'-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-'+stepname[i_step]+'.h5'
                            # predict_data_transform_name = 'transform_'+thm[i_thm]+'-'+year_list[i_year]+'-'+str(i_month+1)+'-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-'+stepname[i_step]+'.h5'
                            # predict_data_transform_fullfilename = predicted_data_full_path+predict_data_transform_name
                            outputfullfilename = predicted_data_full_path + outputfilename
                            thm_and_time_data = pd.read_hdf(outputfullfilename,key='df').values[:,0:2]
                            predict_data_inital = pd.read_hdf(outputfullfilename,key='df').values[:,2:].astype('float')
                            # predict_data_inital_3D = np.reshape(predict_data_inital,(np.shape(predict_data_inital)[0],int(np.shape(predict_data_inital)[1]/240),240))
                            # predict_data = convert(predict_data_inital_3D, from_type="numpy3D", to_type="pd-multiindex")
                            # predict_data_transform = minirocket_multi.transform(predict_data)
                            # predict_data_transform = pd.read_hdf(predict_data_transform_fullfilename,key='df')
                            # if os.path.exists(predict_data_transform_fullfilename):
                            #     continue
                            # # time3 = time.time()
                            # store = pd.HDFStore(predict_data_transform_fullfilename)
                            # store['df'] = pd.DataFrame(predict_data_transform)
                            # store.close()
                            
                            predict_result = model.predict(predict_data_inital)
                            y_predict_result = np.argmax(predict_result , axis=1)

                            # predict_probability_result = model._predict_proba_lr(predict_data_inital)
                            
                            # predict_result.to_csv()
                            # predict_probability_result.to_csv()
                            # #%% 
                            for l in range(0,len(y_predict_result)):
                                new_row_do_nothing_satellite.append(thm_and_time_data[l,0])
                                tmp_time = time_double(thm_and_time_data[l,1])
                                new_row_do_nothing_starttime.append(time_string(tmp_time,'%Y-%m-%d %H:%M:%S'))
                                new_row_do_nothing_endtime.append(time_string(tmp_time+720,'%Y-%m-%d %H:%M:%S'))  
                                tmp_bubble_index = 0
                                if y_predict_result[l] == 1:
                                    new_row_satellite.append(thm_and_time_data[l,0])
                                    tmp_time = time_double(thm_and_time_data[l,1])
                                    new_row_starttime.append(time_string(tmp_time,'%Y-%m-%d %H:%M:%S'))
                                    last_tmp_time = tmp_time
                                    new_row_endtime.append(time_string(tmp_time+720,'%Y-%m-%d %H:%M:%S'))  
                                    tmp_bubble_index = 1

                                new_row_do_nothing_bubble_index.append(tmp_bubble_index)
             
                bubble_time_list["satellite"] = new_row_satellite
                bubble_time_list["starttime"] = new_row_starttime
                bubble_time_list["endtime"] = new_row_endtime
                
                all_event_time_list["satellite"] = new_row_do_nothing_satellite
                all_event_time_list["starttime"] = new_row_do_nothing_starttime
                all_event_time_list["endtime"] = new_row_do_nothing_endtime
                all_event_time_list["bubble_index"] = new_row_do_nothing_bubble_index

                
                bubble_list_file_name = 'cnn_1D-20230110_predicted_bubble_list-NP-Ratio-'+str(NP_ratio[i_NP_ratio])+'-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-'+stepname[i_step]+'.csv'
                bubble_time_list.to_csv(out_result_path+bubble_list_file_name)
                
                all_bubble_list_file_name = 'all_cnn_1D-20230110_predicted_bubble_list-NP-Ratio-'+str(NP_ratio[i_NP_ratio])+'-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-'+stepname[i_step]+'.csv'
                all_event_time_list.to_csv(out_result_path+all_bubble_list_file_name)
                            
            # #%% Initialise ROCKET and Transform the Training Data
            # rocket_time1 = time.time()
            # rocket = Rocket()
            # rocket.fit(X_train)#[:40,:6]))
            # X_train_transform = rocket.transform(X_train)#[:40,:6]))

            # # Fit a Classifier
            # classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            # classifier.fit(X_train_transform, y_train)

            # # Load and Transform the Test Data
            # # X_test, y_test = load_basic_motions(split="test", return_X_y=True)
            # X_test_transform = rocket.transform(X_test)

            # #  Classify the Test Data
            # rocket_score = classifier.score(X_test_transform, y_test)
            # print(NP_ratio[i_NP_ratio],variance_type[i_var],normalized_type[i_normalized])
            # classifier.predict()
            # print('train_sample_name:',train_sample_name)
            # print('Rocket_result:',rocket_score)
            # rocket_time2 = time.time()
            
            # rocket_score_list.append(rocket_score)
            # rocket_time_list.append(rocket_time2-rocket_time1)
            # # result_output['rocket_score'][j] = classifier_score
            # # result_output['rocket_time'][j] = rocket_time2-rocket_time1