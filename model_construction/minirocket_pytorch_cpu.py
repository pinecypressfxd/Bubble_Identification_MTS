import numpy as np

from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
from sktime.transformations.panel.rocket import MiniRocket, MiniRocketMultivariate

import sktime
from sktime.datatypes import convert

import pandas as pd
import math
import time
import h5py
import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##Moving threshold:
from sklearn.metrics import f1_score, precision_recall_curve
from matplotlib import pyplot
def to_labels(pos_probs,threshold):
    return (pos_probs>=threshold).astype('int')



train_test_transform_data_path = '/data/preprocess_model_data/train_test_validation_transform_data/'
file_path = '/data/preprocess_model_data/'
data_file_path = file_path+'/train_test_data/h5_120000/'
output_data_path = '/data/'
out_result_path = '/data/model_result/minirocket_result/'

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


variance_type = ['all_var','initial_var','judge_var']
normalized_type = ['non_normalized','max_min_normalized','mean_std_normalized']
switch_data_set = 3
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
    
    thm_and_time_train_test_shape = [[[61254,2],[20418,2],[20418,2]]]

    NP_ratio = [40]
elif switch_data_set == 3:
    train_test_path = file_path+ 'train_test_data/'
    thm_and_time_train_test_path = file_path+ 'train_test_data/'

    #shape-more-data
    train_test_shape = [[[[65632,4321],[21878,4321],[21878,4321]],[[65632,4321],[21878,4321],[21878,4321]],[[65632,4321],[21878,4321],[21878,4321]]]]
    
    thm_and_time_train_test_shape = [[[65632,2],[21878,2],[21878,2]]]

    NP_ratio = [40]

#%% 正负样本比例negative_positive_ratio
minirocket_score_list = []
minirocket_time_list = []
rocket_score_list = []
rocket_time_list = []
result_output = pd.DataFrame(columns=['minirocket_score','minirocket_time'])#,'rocket_score','rocket_time'])
all_target_index = [target_index_0,target_index_1,target_index_2]
#j = 0


for i_NP_ratio in range(len(NP_ratio)-1,len(NP_ratio)):
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
            start = time.time()
            y_train = pd.read_hdf(train_sample_name,key='df').values[:,0]
            # X_train_inital = pd.read_hdf(train_sample_name,key='df').values[:,1:]
            # X_train_3D = np.reshape(X_train_inital,(np.shape(X_train_inital)[0],int(np.shape(X_train_inital)[1]/240),240))
            # X_train = convert(X_train_3D, from_type="numpy3D", to_type="pd-multiindex")
            
            y_test = pd.read_hdf(test_sample_name,key='df').values[:,0]
            # X_test_inital = pd.read_hdf(test_sample_name,key='df').values[:,1:]
            # X_test_3D = np.reshape(X_test_inital,(np.shape(X_test_inital)[0],int(np.shape(X_test_inital)[1]/240),240))
            # X_test = convert(X_test_3D, from_type="numpy3D", to_type="pd-multiindex")
            
            y_val = pd.read_hdf(val_sample_name,key='df').values[:,0]
            # X_val_inital = pd.read_hdf(val_sample_name,key='df').values[:,1:]
            # X_val_3D = np.reshape(X_val_inital,(np.shape(X_val_inital)[0],int(np.shape(X_val_inital)[1]/240),240))
            # X_val = convert(X_val_3D, from_type="numpy3D", to_type="pd-multiindex")
            

            # check_raise(X_train, mtype="np.ndarray")
            # 
            #%% minirocket
            minirocket_time1 = time.time()
            # minirocket_multi = MiniRocketMultivariate()
            # minirocket_multi.fit(X_train)
            
            # picklefile_transform = out_result_path+'MiniRocketMultivariate-transform-20221215_with-val-'+variance_type[i_var]+'_'+'NP_ratio-'+str(NP_ratio[i_NP_ratio])+'.sav'
            # pickle.dump(minirocket_multi,open(picklefile_transform,'wb'))
            # transform_model = pickle.load(open(picklefile_transform,'rb'))
            
            
            # X_train_transform = transform_model.transform(X_train)
            # X_test_transform = transform_model.transform(X_test)
            # X_val_transform = transform_model.transform(X_val)
            train_data_transform_filename = train_test_transform_data_path+'X_train_transform_NP_'+variance_type[i_var]+'_'+str(NP_ratio[i_NP_ratio])+'.csv'
            test_data_transform_filename = train_test_transform_data_path+'X_test_transform_NP_'+variance_type[i_var]+'_'+str(NP_ratio[i_NP_ratio])+'.csv'
            val_data_transform_filename = train_test_transform_data_path+'X_val_transform_NP_'+variance_type[i_var]+'_'+str(NP_ratio[i_NP_ratio])+'.csv'
            X_train_transform = pd.read_csv(train_data_transform_filename)
            X_train_transform = X_train_transform.drop(['Unnamed: 0'],axis=1)
            X_test_transform = pd.read_csv(test_data_transform_filename)
            X_test_transform = X_test_transform.drop(['Unnamed: 0'],axis=1)
            X_val_transform = pd.read_csv(val_data_transform_filename)
            X_val_transform = X_val_transform.drop(['Unnamed: 0'],axis=1)
            end = time.time()
            # print('time = ',start-end,' s')

            # X_train_transform = pd.read_csv(train_test_transform_data_path+'X_train_transform_NP_'+variance_type[i_var]+'_'+str(NP_ratio[i_NP_ratio])+'.csv')
            # X_test_transform = pd.read_csv(train_test_transform_data_path+'X_test_transform_NP_'+variance_type[i_var]+'_'+str(NP_ratio[i_NP_ratio])+'.csv')
            # X_train_transform = X_train_transform.drop(['Unnamed: 0'],axis=1)
            # X_test_transform = X_test_transform.drop(['Unnamed: 0'],axis=1)
            # comments
            #classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            #classifier = LogisticRegressionCV(cv=10, random_state=0)
            
            
            # print('classifier start')
            # classifier = LogisticRegressionCV(Cs=10, penalty ='l1',max_iter=100,scoring='roc_auc', solver='saga',cv=10, random_state=0)#Cs=1e-1, solver = saga,liblinear
            # print('classifier end')
           


            # minirocket_model = classifier.fit(X_train_transform, y_train)
            picklefile = out_result_path+'minirocket_classifier-20221215_with-val-reevaluate-'+variance_type[i_var]+'_'+'NP_ratio-'+str(NP_ratio[i_NP_ratio])+'.sav'
            # pickle.dump(minirocket_model,open(picklefile,'wb'))
            loaded_model = pickle.load(open(picklefile,'rb'))
            minirocket_score = loaded_model.score(X_test_transform, y_test)
            minirocket_time2 = time.time()
            minirocket_score_list.append(minirocket_score)
            minirocket_time_list.append(minirocket_time2-minirocket_time1)
            
            print('train_sample_name:',train_sample_name)
            print('minirocket_score:', minirocket_score)
            
            y_test_predict = loaded_model.predict(X_test_transform)
            y_train_predict = loaded_model.predict(X_train_transform)
            y_val_predict = loaded_model.predict(X_val_transform)

            y_test_predict_probability = loaded_model._predict_proba_lr(X_test_transform)
            y_train_predict_probability = loaded_model._predict_proba_lr(X_train_transform)
            y_val_predict_probability = loaded_model._predict_proba_lr(X_val_transform)

            probs_train = y_train_predict_probability[:,1]
            probs_test = y_test_predict_probability[:,1]
            probs_val = y_val_predict_probability[:,1]
            
            ##% get best threshold
            thresholds = np.arange(0,1,0.001)
            scores_train=[f1_score(y_train,to_labels(probs_train,t)) for t in thresholds]
            scores_test =[f1_score(y_test ,to_labels(probs_test, t)) for t in thresholds]
            scores_val  =[f1_score(y_val  ,to_labels(probs_val,  t)) for t in thresholds]
            ix_train = np.argmax(scores_train)
            ix_test  = np.argmax(scores_test ) 
            ix_val   = np.argmax(scores_val  ) 

            print("Train dataset Threshold %.3f, F-Score=%.5f"%(thresholds[ix_train],scores_train[ix_train]))
            print("Test  dataset Threshold %.3f, F-Score=%.5f"%(thresholds[ix_test ],scores_test[ix_test ]))
            print("Val   dataset Threshold %.3f, F-Score=%.5f"%(thresholds[ix_val  ],scores_val[ix_val  ]))

            tn_test, fp_test, fn_test, tp_test  = confusion_matrix(y_test,y_test_predict).ravel()
            tn_train,fp_train,fn_train,tp_train = confusion_matrix(y_train,y_train_predict).ravel()
            tn_val,fp_val,fn_val,tp_val = confusion_matrix(y_val,y_val_predict).ravel()

            print("tn_test, fp_test, fn_test, tp_test:",tn_test, fp_test, fn_test, tp_test)
            print("tn_train,fp_train,fn_train,tp_train:",tn_train,fp_train,fn_train,tp_train)
            print("tn_val,fp_val,fn_val,tp_val:",tn_val,fp_val,fn_val,tp_val)

            ##% presicion recall curve
            
            train_precision, train_recall, train_thresholds = precision_recall_curve(y_train,probs_train)
            test_precision, test_recall, test_thresholds = precision_recall_curve(y_test,probs_test)
            val_precision, val_recall, val_thresholds = precision_recall_curve(y_val,probs_val)

            train_fscore = (2*train_precision*train_recall)/(train_precision+train_recall)
            test_fscore = (2*test_precision*test_recall)/(test_precision+test_recall)
            val_fscore = (2*val_precision*val_recall)/(val_precision+val_recall)

            ix_train = np.argmax(train_fscore)
            ix_test = np.argmax(test_fscore)
            ix_val = np.argmax(val_fscore)

            # plot the roc curve for the model
            no_skill = len(y_test[y_test==1])/len(y_test)
            pyplot.plot([0,1],[no_skill,no_skill],linestyle='--',label='No skill')
            pyplot.plot(train_recall, train_precision, marker='.', color='red',label='train Logistic')
            pyplot.plot(test_recall, test_precision, marker='.',color='green', label='test Logistic')
            pyplot.plot(val_recall, val_precision, marker='.',color='yellow', label='val Logistic')
            pyplot.scatter(train_recall[ix_train],train_precision[ix_train],marker='o',color='black',label='Train best')
            pyplot.scatter(test_recall[ix_test],test_precision[ix_test],marker='o',color='black',label='Test best')
            pyplot.scatter(val_recall[ix_val],val_precision[ix_val],marker='o',color='black',label='Val best')
            pyplot.xlabel('Recall')
            pyplot.ylabel('Precision')
            pyplot.legend()
            pyplot.show()
            exit(0)

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
result_output.to_csv('minirocket_result_more_data_recall-20221215-with-validation.csv')