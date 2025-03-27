#!/usr/bin/env python
# coding: utf-8

# 
# # Stroke Predication

# In[1]:


# Importing Libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# for displaying all feature from dataset:
pd.pandas.set_option('display.max_columns', None)


# In[3]:


# Reading Dataset:
dataset = pd.read_csv("Stroke_data.csv")
# Top 5 records:
dataset.head()


# ### Attribute Information
# 1) id: unique identifier <br>
# 2) gender: "Male", "Female" or "Other" <br>
# 3) age: age of the patient <br>
# 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension <br>
# 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease <br>
# 6) ever_married: "No" or "Yes" <br>
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed" <br>
# 8) Residence_type: "Rural" or "Urban" <br>
# 9) avg_glucose_level: average glucose level in blood <br>
# 10) bmi: body mass index <br>
# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"* <br>
# 12) stroke: 1 if the patient had a stroke or 0 if not <br>
# *Note: "Unknown" in smoking_status means that the information is unavailable for this patient

# In[4]:


# Dropping unneccsary feature :
dataset = dataset.drop('id', axis=1)


# In[5]:


# Shape of dataset:
dataset.shape


# In[6]:


# Cheaking Missing (NaN) Values:
dataset.isnull().sum()


# In[7]:


# Filling NaN Values in BMI feature using mean:
dataset['bmi'] = dataset['bmi'].fillna(dataset['bmi'].median())


# In[8]:


# After filling Missing (NaN) Values in BMI feature:
dataset.isnull().sum()


# In[9]:


# Description:
dataset.describe()


# In[10]:


# Datatypes:
dataset.dtypes


# In[11]:


# Target feature:
print("Stroke People     : ", dataset['stroke'].value_counts()[1])
print("Non-Stroke People : ", dataset['stroke'].value_counts()[0])


# - By seeing target feature, We clearly say we have **imbalenced dataset**.

# In[12]:


# Gender
dataset['gender'].value_counts()


# In[13]:


dataset[dataset['gender']=='Other']


# - We Seen that in our **Gender** feature, we have only one **Other** gender, So instead of taking we drop that record.

# In[14]:


# Dropping Other gender
Other_gender = dataset[dataset['gender'] == 'Other'].index[0]
dataset = dataset.drop(Other_gender, axis=0)


# In[15]:


# Gender:
print("Male    : ", dataset['gender'].value_counts()[1])
print("female  : ", dataset['gender'].value_counts()[0])


# In[16]:


# Hypertension:
print("Hypertension People     : ", dataset['hypertension'].value_counts()[1])
print("Non-hypertension People : ", dataset['hypertension'].value_counts()[0])


# In[17]:


# Heart Disease:
print("Heart Disease People     : ", dataset['heart_disease'].value_counts()[1])
print("Non-Heart Disease People : ", dataset['heart_disease'].value_counts()[0])


# In[18]:


# Single VS Married:
print("Single    : ", dataset['ever_married'].value_counts()[1])
print("Married   : ", dataset['ever_married'].value_counts()[0])


# In[19]:


# Work Type:
print("Private         : ", dataset['work_type'].value_counts()[0])
print("Self-employed   : ", dataset['work_type'].value_counts()[1])
print("children        : ", dataset['work_type'].value_counts()[2])
print("Govt_job        : ", dataset['work_type'].value_counts()[3])
print("Never_worked    : ", dataset['work_type'].value_counts()[4])


# In[20]:


# Rename some names in worktype feature for simplacity nothing else:
dataset.replace({'Self-employed' : 'Self_employed'}, inplace=True)


# In[21]:


# Residence Type:
print("Urban   : ", dataset['Residence_type'].value_counts()[0])
print("Rural   : ", dataset['Residence_type'].value_counts()[1])


# In[22]:


# Smokers:
print("Never Smoked      : ", dataset['smoking_status'].value_counts()[0])
print("Unknown           : ", dataset['smoking_status'].value_counts()[1])
print("Formerly Smoked   : ", dataset['smoking_status'].value_counts()[2])
print("Smokes            : ", dataset['smoking_status'].value_counts()[3])


# In[23]:


# Rename some names in smokers feature for simplacity nothing else:
dataset.replace({'never smoked':'never_smoked', 'formerly smoked':'formerly_smoked'}, inplace=True)


# In[24]:


never_smoked, Unknown, formerly_smoked, Smokes


# In[24]:


# Boxplot:
plt.figure(figsize=(15,12))
dataset.plot(kind='box', subplots=True, layout=(2,3), figsize=(20, 10))
plt.show()


# In[25]:


# Correlation using Heatmap:
plt.figure(figsize=(12,8))
sns.heatmap(dataset.corr(), annot=True, cmap='YlGnBu')
plt.show()


# In[26]:


dataset.head()


# In[27]:


# Dependent & Independent Feature:
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[28]:


X.head()


# In[29]:


# Label Encoding:
X['ever_married'] = np.where(X['ever_married']=='Yes',1,0)   ## If married replace with by 1 otherwise 0.
X['Residence_type'] = np.where(X['Residence_type']=='Rural',1,0)    ## If residence type is Rural replace it by 1 otherwise 0.


# In[30]:


# One Hot Encoding:
X = pd.get_dummies(X, drop_first=True)


# In[31]:


X.head()


# In[32]:


X.columns


# In[33]:


# Rearranging the columns for better understanding
X = X[['gender_Male','age', 'hypertension', 'heart_disease', 'ever_married',
       'Residence_type', 'avg_glucose_level', 'bmi', 
       'work_type_Never_worked', 'work_type_Private','work_type_Self_employed', 'work_type_children',
       'smoking_status_formerly_smoked', 'smoking_status_never_smoked','smoking_status_smokes']]


# In[34]:


X.head()


# In[35]:


# Train-Test Split:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[36]:


print(X_train.shape)
print(X_test.shape)


# In[37]:


# Importing Performance Metrics:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[38]:


# RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[39]:


# AdaBoostClassifier:
from sklearn.ensemble import AdaBoostClassifier
AdaBoost = AdaBoostClassifier()
AdaBoost = AdaBoost.fit(X_train,y_train)

# Predictions:
y_pred = AdaBoost.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[40]:


# GradientBoostingClassifier:
from sklearn.ensemble import GradientBoostingClassifier
GradientBoost = GradientBoostingClassifier()
GradientBoost = GradientBoost.fit(X_train,y_train)

# Predictions:
y_pred = GradientBoost.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #### RandomizedSearchCV

# In[41]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200,20)]
# Minimum number of samples required to split a node
min_samples_split = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)

rf = RandomForestClassifier()
rf_randomcv = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=10,verbose=2,
                               random_state=0,n_jobs=-1)
### fit the randomized model
rf_randomcv.fit(X_train,y_train)


# In[42]:


rf_randomcv.best_params_


# In[44]:


RandomForest_RandomCV = RandomForestClassifier(criterion='gini', n_estimators=100, max_depth=130, max_features='auto', min_samples_split=14, min_samples_leaf=16)
RandomForest_RandomCV = RandomForest_RandomCV.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest_RandomCV.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # ----------------------------------------------------------------------------------------

# ## SMOTE

# In[47]:


import delayed


# In[48]:


from imblearn.combine import SMOTETomek
smote = SMOTETomek()
X_smote, y_smote = smote.fit_resample(X,y)


# In[49]:


from collections import Counter
print('Before SMOTE : ', Counter(y))
print('After SMOTE  : ', Counter(y_smote))


# In[50]:


# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(X_smote,y_smote, test_size=0.2, random_state=0)

# RandomForestClassifier:
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# ## Over Sampling

# In[51]:


from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler(0.4)
x_oversampler, y_oversampler = oversampler.fit_resample(X,y)


# In[52]:


print('Before RandomOverSampler : ', Counter(y))
print('After RandomOverSampler  : ', Counter(y_oversampler))


# - #### We make 60-40% data.

# In[53]:


# Train Test Split:
X_train, X_test, y_train, y_test = train_test_split(x_oversampler,y_oversampler, test_size=0.2, random_state=0)

# RandomForestClassifier:
RandomForest = RandomForestClassifier()
RandomForest = RandomForest.fit(X_train,y_train)

# Predictions:
y_pred = RandomForest.predict(X_test)

# Performance:
print('Accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:




