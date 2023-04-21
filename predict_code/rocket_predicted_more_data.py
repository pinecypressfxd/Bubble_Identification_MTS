import numpy as np
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
# 2022-10-17@fxd
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import classification_report, confusion_matrix
26396

from sktime.datatypes import convert
from pyspedas import time_double
from pyspedas import time_string


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


file_path = '/data/preprocess_model_data/'
data_file_path = file_path+'/train_test_data/h5_120000/'
# output_data_path = '/data/'
out_result_path = '/data/model_result/minirocket_result/'
train_test_path = file_path+ 'train_test_data/h5_120000/'
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
train_test_shape = [[[[16524,4321],[7082,4321]],[[16524,2401],[7082,2401]],[[16524,2401],[7082,2401]]],
[[[31546,4321],[13520,4321]],[[31546,2401],[13520,2401]],[[31546,2401],[13520,2401]]],
[[[61590,4321],[26396,4321]],[[61590,2401],[26396,2401]],[[61590,2401],[26396,2401]]]]


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
NP_ratio = [10,20,40]
minirocket_score_list = []
minirocket_time_list = []
rocket_score_list = []
rocket_time_list = []
result_output = pd.DataFrame(columns=['minirocket_score','minirocket_time'])#,'rocket_score','rocket_time'])
all_target_index = [target_index_0,target_index_1,target_index_2]
#j = 0

#%% predicted data 
year_list = ['2020','2021']
thm = ['tha','thb','thc','thd','the']
step = [60,240]#3分钟为窗口的步长
stepname =['3min','12min']

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
for i_NP_ratio in range(2,len(NP_ratio)):
    # for i_var in range(0,len(variance_type)-2):
    for i_var in range(0,len(variance_type)-2):
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

            y_train = pd.read_hdf(train_sample_name,key='df').values[:,0]
            # X_train_inital = pd.read_hdf(train_sample_name,key='df').values[:,1:]
            # X_train_3D = np.reshape(X_train_inital,(np.shape(X_train_inital)[0],int(np.shape(X_train_inital)[1]/240),240))
            # X_train = convert(X_train_3D, from_type="numpy3D", to_type="pd-multiindex")
            y_test = pd.read_hdf(test_sample_name,key='df').values[:,0]
            # X_test_inital = pd.read_hdf(test_sample_name,key='df').values[:,1:]
            # X_test_3D = np.reshape(X_test_inital,(np.shape(X_test_inital)[0],int(np.shape(X_test_inital)[1]/240),240))
            # X_test = convert(X_test_3D, from_type="numpy3D", to_type="pd-multiindex")
            
            # check_raise(X_train, mtype="np.ndarray")
            # 多变量时间序列数据

            #%% minirocket
            minirocket_time1 = time.time()
            # minirocket_multi = MiniRocketMultivariate()
            # minirocket_multi.fit(X_train)
            # X_train_transform = minirocket_multi.transform(X_train)
            # X_test_transform = minirocket_multi.transform(X_test)
            
            
            X_train_transform = pd.read_csv(train_test_transform_data_path+'X_train_transform_NP_'+str(NP_ratio[i_NP_ratio])+'.csv')
            X_test_transform = pd.read_csv(train_test_transform_data_path+'X_test_transform_NP_'+str(NP_ratio[i_NP_ratio])+'.csv')
            X_train_transform = X_train_transform.drop(['Unnamed: 0'],axis=1)
            X_test_transform = X_test_transform.drop(['Unnamed: 0'],axis=1)
            
            # classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            # classifier = LogisticRegressionCV(cv=10, random_state=0,max_iter=500,scoring='recall')
            #classifier = LogisticRegressionCV(cv=10, random_state=0,max_iter=500,n_jobs=16)
            # classifier = LogisticRegressionCV(cv=10, random_state=0)
            classifier = LogisticRegressionCV(Cs=[10,100,1000,10000], penalty ='l1',max_iter=100, solver='saga',cv=10, random_state=0)#Cs=1e-1, solver = saga,liblinear

            minirocket_model = classifier.fit(X_train_transform, y_train)
            # classifier.store(message_set, command, flag_list)
            # X_test, y_test = load_basic_motions(split="test", return_X_y=True)
            minirocket_model.save(out_result_path)
            minirocket_score = classifier.score(X_test_transform, y_test)
            # minirocket_time2 = time.time()
            y_test_predict = classifier.predict(X_test_transform)
            tn,fp,fn,tp = confusion_matrix(y_test,y_test_predict).ravel()
            print("tn,fp,fn,tp:",tn,fp,fn,tp)
           
            
            y_test_predict = classifier.predict(X_test_transform)
            y_train_predict = classifier.predict(X_train_transform)

            tn_test, fp_test, fn_test, tp_test  = confusion_matrix(y_test,y_test_predict).ravel()
            tn_train,fp_train,fn_train,tp_train = confusion_matrix(y_train,y_train_predict).ravel()

            print("tn_test, fp_test, fn_test, tp_test:",tn_test, fp_test, fn_test, tp_test)
            print("tn_train,fp_train,fn_train,tp_train:",tn_train,fp_train,fn_train,tp_train)

            y_test_predict_probability = classifier._predict_proba_lr(X_test_transform)
            y_test_predict_file_name = 'predicted_y_test_result-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'.csv'
            y_test_predict_probility_file_name = 'predicted_y_probility_test_result-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'.csv'

            np.savetxt(out_result_path+y_test_predict_file_name,[y_test_predict],delimiter=",")
            np.savetxt(out_result_path+y_test_predict_probility_file_name,[y_test_predict_probability.ravel()],delimiter=",")

            # # # np.sum(y_test_predict_probability[:,0]>0.7341)
            # minirocket_score_list.append(minirocket_score)
            # minirocket_time_list.append(minirocket_time2-minirocket_time1)
            
            print('train_sample_name:',train_sample_name)
            print('minirocket_score:', minirocket_score)
            # #for i_step in range(0,1):#len(step)):
            for i_step in range(0,len(step)):
                # if i_var==0 and i_step ==0:
                #     continue
                print("-------step:",step[i_step],"--------")
                bubble_time_list = pd.DataFrame(columns= ["satellite","starttime","endtime"])

                new_row_satellite = []#={'satellite':'','starttime':'','endtime':''}
                new_row_starttime = []
                new_row_endtime = []
                #new_row_do_nothing ={'satellite':'','starttime':'','endtime':''}
                new_row_do_nothing_satellite = []
                new_row_do_nothing_starttime = []
                new_row_do_nothing_endtime = []
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
                            predict_data_transform_name = 'transform_'+thm[i_thm]+'-'+year_list[i_year]+'-'+str(i_month+1)+'-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-'+stepname[i_step]+'.h5'
                            predict_data_transform_fullfilename = predicted_data_full_path+predict_data_transform_name
                            outputfullfilename = predicted_data_full_path + outputfilename
                            thm_and_time_data = pd.read_hdf(outputfullfilename,key='df').values[:,0:2]
                            # predict_data_inital = pd.read_hdf(outputfullfilename,key='df').values[:,2:].astype('float')
                            # predict_data_inital_3D = np.reshape(predict_data_inital,(np.shape(predict_data_inital)[0],int(np.shape(predict_data_inital)[1]/240),240))
                            # predict_data = convert(predict_data_inital_3D, from_type="numpy3D", to_type="pd-multiindex")
                            # predict_data_transform = minirocket_multi.transform(predict_data)
                            predict_data_transform = pd.read_hdf(predict_data_transform_fullfilename,key='df')
                            # if os.path.exists(predict_data_transform_fullfilename):
                            #     continue
                            # # time3 = time.time()
                            # store = pd.HDFStore(predict_data_transform_fullfilename)
                            # store['df'] = pd.DataFrame(predict_data_transform)
                            # store.close()
                            
                            predict_result = classifier.predict(predict_data_transform)
                            predict_probability_result = classifier._predict_proba_lr(predict_data_transform)
                            
                            # predict_result.to_csv()
                            # predict_probability_result.to_csv()
                            # #%% 
                            for l in range(0,len(predict_result)):
                                if predict_result[l] == 1:
                                    new_row_satellite.append(thm_and_time_data[l,0])
                                    tmp_time = time_double(thm_and_time_data[l,1])
                                    new_row_starttime.append(time_string(tmp_time,'%Y-%m-%d %H:%M:%S'))
                                    last_tmp_time = tmp_time
                                    new_row_endtime.append(time_string(tmp_time+720,'%Y-%m-%d %H:%M:%S'))               
                bubble_time_list["satellite"] = new_row_satellite
                bubble_time_list["starttime"] = new_row_starttime
                bubble_time_list["endtime"] = new_row_endtime
                
                bubble_list_file_name = 'Minirocket_predicted-20221109_bubble_list-NP-Ratio-'+str(NP_ratio[i_NP_ratio])+'-'+variance_type[i_var]+'-'+normalized_type[i_normalized]+'-'+stepname[i_step]+'.csv'
                bubble_time_list.to_csv(out_result_path+bubble_list_file_name)
                            
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

result_output['minirocket_score'] = minirocket_score_list
result_output['minirocket_time'] = minirocket_time_list
# result_output['rocket_score'] = rocket_score_list
# result_output['rocket_time'] = rocket_time_list
result_output.to_csv(out_result_path+'rocket_and_minirocket_result.csv')