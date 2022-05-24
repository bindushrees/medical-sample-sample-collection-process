# -*- coding: utf-8 -*-
"""
Created on Fri May 13 22:32:32 2022

@author: bindu
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from feature_engine.outliers import Winsorizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_excel("C:/Users/bindu/OneDrive/Desktop/Data_Science_Course/0. Project/estimating_delivery_of sample_collection/sampla_data_08_05_2022(final).xlsx")
data.dtypes

cols = list(data.columns)

data.isna().sum()

'''data = pd.get_dummies(data, columns = ["Patient_Gender",
                                       "Test_Name",
                                       "Sample",
                                       "Way_Of_Storage_Of_Sample",
                                       "Cut_off_Schedule",
                                       "Traffic_Conditions",
                                       "Mode_Of_Transport"],
                      drop_first = True)'''

labelencoder = LabelEncoder()
data['Patient_Gender']= labelencoder.fit_transform(data['Patient_Gender'])
data['Test_Name']= labelencoder.fit_transform(data['Test_Name'])
data['Sample']= labelencoder.fit_transform(data['Sample'])
data['Way_Of_Storage_Of_Sample']= labelencoder.fit_transform(data['Way_Of_Storage_Of_Sample'])
data['Cut_off_Schedule']= labelencoder.fit_transform(data['Cut_off_Schedule'])
data['Traffic_Conditions']= labelencoder.fit_transform(data['Traffic_Conditions'])
data['Mode_Of_Transport']= labelencoder.fit_transform(data['Mode_Of_Transport'])

data = data.drop(["Patient_ID","Agent_ID","Test_Booking_Date",
                  "Test_Booking_Time_HH_MM",
                  "Scheduled_Sample_Collection_Time_HH_MM",
                  "Time_Taken_To_Reach_Patient_MM",
                  "Mode_Of_Transport","Agent_Location_KM",
                  "Sample_Collection_Date"],axis =1)


sns.boxplot(data.Patient_ID)    #no outlier
sns.boxplot(data.Patient_Age)
#sns.boxplot(data.Patient_Gender_Male)  
#sns.boxplot(data.Test_Name_CBC)
#sns.boxplot(data.Sample)    
#sns.boxplot(data.Way_Of_Storage_Of_Sample)
sns.boxplot(data.Test_Booking_Date)    #no outlier
sns.boxplot(data.Test_Booking_Time_HH_MM)
sns.boxplot(data.Sample_Collection_Date)   
sns.boxplot(data.Scheduled_Sample_Collection_Time_HH_MM)    #no outlier
#sns.boxplot(data.Cut_off_Schedule)
sns.boxplot(data.Cut_off_time_HH_MM)
sns.boxplot(data.Agent_ID)
#sns.boxplot(data.Traffic_Conditions)
sns.boxplot(data.Agent_Location_KM)
sns.boxplot(data.Time_Taken_To_Reach_Patient_MM)
sns.boxplot(data.Time_For_Sample_Collection_MM)
sns.boxplot(data.Lab_Location_KM)
sns.boxplot(data.Time_Taken_To_Reach_Lab_MM)
#sns.boxplot(data.Mode_Of_Transport)

list= ['Patient_Age','Cut_off_time_HH_MM',
       'Time_For_Sample_Collection_MM',
       'Lab_Location_KM','Time_Taken_To_Reach_Lab_MM']

for i in list:
    winsor = Winsorizer(capping_method='iqr', 
                              tail='both',
                              fold=1.5,
                              variables=[i])
    data[i] = winsor.fit_transform(data[[i]])
    

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

data = data.iloc[:,[0,6,8,9,10,1,2,3,4,5,7,11]]
df_norm = norm_func(data.iloc[:,:5])

data_final = pd.concat([df_norm, data.iloc[:,5:11]], axis=1, ignore_index=False)
    
describe = data_final.describe()

predictors = data_final
#data.loc[:, data.columns!="Reached_On_Time"]  
target = data["Reached_On_Time"]    

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

#####################################################################

import xgboost as xgb
xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)
xgb_clf.fit(x_train, y_train)

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))
result = classification_report(y_test,xgb_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, xgb_clf.predict(x_train))
accuracy_score(y_train, xgb_clf.predict(x_train))

######################################################################

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(x_train, y_train)

# Evaluation on Testing Data
confusion_matrix(y_test, rf_clf.predict(x_test))
accuracy_score(y_test, rf_clf.predict(x_test))
result1 = classification_report(y_test,rf_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, rf_clf.predict(x_train))
accuracy_score(y_train, rf_clf.predict(x_train))

#####################################################################
# GridSearchCV

from sklearn.model_selection import GridSearchCV

#rf_clf_grid = RandomForestClassifier(n_estimators=5000, n_jobs=-1)
rf_clf_grid = RandomForestClassifier( random_state=42)

#param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}
param_grid = {'n_estimators':[200,500,5000],
    'max_features': ['auto', 'sqrt', 'log2'],
    "min_samples_split": [2, 3, 10],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, cv_rf_clf_grid.predict(x_test))
accuracy_score(y_test, cv_rf_clf_grid.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, cv_rf_clf_grid.predict(x_train))
accuracy_score(y_train, cv_rf_clf_grid.predict(x_train))


#####################################################################
# saving the model
# importing pickle
import pickle
pickle.dump(grid_search, open('model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(data_final.iloc[0:1,:])
list_value

print(model.predict(list_value))








