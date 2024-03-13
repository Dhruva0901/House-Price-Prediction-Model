#!/usr/bin/env python
# coding: utf-8

# In[249]:


import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import streamlit as st
import pickle


# In[250]:


dataset=pd.read_csv("Bengaluru_House_Data_new_features.csv")
dataset.shape


# In[251]:


dataset.head()


# In[252]:


dataset.drop_duplicates()
dataset.shape


# In[253]:


dataset.info()


# In[254]:


#to check null values
dataset.isnull().sum()


# In[255]:


d1 = dataset[dataset['location'].isna()]
d1.head()


# In[256]:


d2 = d1 = dataset[dataset['society']=='Grare S']
d2.head()


# In[257]:


#to fill NAN value in location column
dataset['location'].replace(to_replace=np.NaN,value="Anantapura",inplace=True)


# In[258]:


dataset.dropna(subset=['size'],axis='rows',inplace=True)
dataset = dataset.reset_index(drop = True)


# In[259]:


dataset.shape


# In[260]:


#adding bedroom column
dataset['bedroom'] = dataset['size'].apply(lambda x: int(x.split(' ')[0]))


# In[261]:


lst_bed=[]
lst_bath=[]
for i in range(len(dataset)):
    if dataset['bedroom'][i]<10:
        lst_bed.append(dataset['bedroom'][i])
        lst_bath.append(dataset['bath'][i])
plt.scatter(lst_bed,lst_bath)
plt.xlabel("number of bedroom")
plt.ylabel('number of bathroom')
plt.show();


# In[262]:


#to fill NAN values in bath column
for i in range(0,len(dataset)):
    if pd.isnull(dataset['bath'][i])==True:
        if dataset['bedroom'][i] == 1:
            dataset['bath'][i]=rd.choice([1,2])
        elif dataset['bedroom'][i] == 2:
            dataset['bath'][i]=rd.randint(1,4)
        elif dataset['bedroom'][i] == 3:
            dataset['bath'][i]=rd.randint(1,6)
        elif dataset['bedroom'][i] == 4:
            dataset['bath'][i]=rd.randint(1,8)
        elif dataset['bedroom'][i] == 5:
            dataset['bath'][i]=rd.randint(2,8)
        elif dataset['bedroom'][i] == 6:
            dataset['bath'][i]=rd.randint(3,9)
        elif dataset['bedroom'][i] == 7:
            dataset['bath'][i]=rd.randint(2,9)
        elif dataset['bedroom'][i] == 8:
            dataset['bath'][i]=rd.randint(3,12)
        elif dataset['bedroom'][i] == 9:
            dataset['bath'][i]=rd.randint(5,14)
        else:
            dataset['bath'][i]=dataset['bedroom'][i]      


# In[263]:


dataset.dropna(subset=['balcony'],axis='rows',inplace=True)
dataset = dataset.reset_index(drop = True)


# In[264]:


#to fill NAN value in balcony column
dataset.shape


# In[265]:


#adding new feature parking_facility
park=[]
for i in range(len(dataset)):
    if (dataset['bedroom'][i] >3 and dataset['street_type'][i] in ['Gravel','Paved']):
        park.append('Yes')
    elif (dataset['bedroom'][i] == 3 and dataset['dist_mainroad'][i] < 25):
        park.append('Yes')
    elif (dataset['bedroom'][i] in [2,3] and dataset['street_type'][i]=='Gravel'):
        park.append('Yes')
    else:
        park.append('No')


# In[266]:


dataset.insert(7,'park_facility',park,True)
dataset.head()


# In[267]:


dataset['park_facility'].value_counts()


# In[268]:


# to check total_sqft column
dataset['total_sqft'].unique()


# In[269]:


def tofloat(x):
    try:
        float(x)
    except:
        return False
    return True

dataset[~dataset['total_sqft'].apply(tofloat)]

def convert_sqft_tonum(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None
    
dataset['total_sqft']=dataset['total_sqft'].apply(convert_sqft_tonum)


# In[270]:


dataset['total_sqft'].fillna(value=(dataset['total_sqft'].mean()),inplace=True)


# In[271]:


#cleaning data in availability column by reducing different headings
for i in range(len(dataset)):
    if (dataset['availability'][i]=='Ready To Move'):
        dataset['availability'][i] ='Ready To Move'
    elif (dataset['availability'][i]=='Immediate Possession'):
        dataset['availability'][i] ='Immediate Possession'
    else:
        dataset['availability'][i] = 'Others'


# In[272]:


#remove leading and ending extra spaces
dataset['location']=dataset['location'].apply(lambda x: x.strip())


# In[273]:


loc = dataset.groupby('location')['location'].agg('count').sort_values(ascending=False)
loc


# In[274]:


len(loc[loc <= 10])


# In[275]:


loc_less_then10 = loc[loc<=10]
dataset['location'] = dataset['location'].apply(lambda x: 'other' if x in loc_less_then10 else x)
print(len(dataset['location'].unique()))


# In[276]:


dataset['area_type'].unique()


# In[277]:


#turning string data to numeric data
#assigning 0 to Super built-up  Area
#assigning 1 to Plot  Area
#asigning 2 to Built-up  Area
#assigning 3 to Carpet  Area
dataset['area_type'].replace(['Super built-up  Area'],'0',inplace=True)
dataset['area_type'].replace(['Plot  Area'],'1',inplace=True)
dataset['area_type'].replace(['Built-up  Area'],'2',inplace=True)
dataset['area_type'].replace(['Carpet  Area'],'3',inplace=True)


# In[278]:


dataset['availability'].unique()


# In[279]:


#turning string data to numeric data
#assigning 0 to Immediate Possession
#assigning 1 to Ready To Move
#asigning 2 to Others
dataset['availability'].replace(['Immediate Possession'],'0',inplace=True)
dataset['availability'].replace(['Ready To Move'],'1',inplace=True)
dataset['availability'].replace(['Others'],'2',inplace=True)


# In[280]:


dataset['street_type'].unique()


# In[281]:


#turning string data to numeric data
#assigning 0 to Gravel
#assigning 1 to Paved
#asigning 2 to No Access
dataset['street_type'].replace(['Gravel'],'0',inplace=True)
dataset['street_type'].replace(['Paved'],'1',inplace=True)
dataset['street_type'].replace(['No Access'],'2',inplace=True)


# In[282]:


dataset['park_facility'].unique()


# In[283]:


#turning string data to numeric data
#assigning 0 to Yes
#assigning 1 to No
dataset['park_facility'].replace(['Yes'],'0',inplace=True)
dataset['park_facility'].replace(['No'],'1',inplace=True)


# In[284]:


dataset['price']=dataset['price']*10000


# In[285]:


price_per_sqft=[]
for i in range(len(dataset)):
    a=(dataset['price'][i])/dataset['total_sqft'][i]
    price_per_sqft.append(a)

dataset['price_per_sqft'] = price_per_sqft


# In[286]:


d=dict(dataset['location'].value_counts())
lst0=[]
lst1=[]
lst2=[]
for i in d.keys():
    if d[i]>1000:
        lst0.append(i)
    elif (d[i]>=70 and d[i]<600):
        lst1.append(i)
    else:
        lst2.append(i)
for i in range(len(dataset)):
    if (dataset['location'][i] in lst0):
        dataset['location'][i]=0
    elif (dataset['location'][i] in lst1):
        dataset['location'][i]=1
    else:
        dataset['location'][i]=2


# In[287]:


temp=['area_type','availability','location']
for i in temp:
    print("**********value count in",i,"***********")
    print(dataset[i].value_counts())
    print("")


# In[288]:


cleaned_dataset = dataset.drop(['size','society'],axis='columns')
cleaned_dataset


# In[304]:


cleaned_dataset['price_per_sqft'].describe()


# In[296]:


cleaned_dataset.to_csv('Bengaluru_House_Data_cleaned.csv',index = False)


# # Using Random Forest

# In[289]:


X = cleaned_dataset.drop(['price'],axis='columns')
Y = cleaned_dataset['price']


# In[290]:


#lets divide the data into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.3)


# In[291]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[292]:


from sklearn.ensemble import RandomForestRegressor
import gzip

regressor_rf = RandomForestRegressor(n_estimators=100, random_state=0)
regressor_rf.fit(X_train, Y_train)

#pickle.dump(regressor_rf,open('model.pkl','wb'))

y_pred_test_rf= regressor_rf.predict(X_test)


# In[293]:


from sklearn import metrics

print('Mean Absolute Error: ',metrics.mean_absolute_error(Y_test,y_pred_test_rf))
print('mean squared Error: ', metrics.mean_squared_error(Y_test,y_pred_test_rf))
print('Root mean squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test,y_pred_test_rf)))


# In[294]:


from sklearn.metrics import r2_score 
r2_score_rf = (r2_score(Y_test,y_pred_test_rf))
print(r2_score_rf)


# In[295]:


##dump the model into a file
with open("model.bin", 'wb') as f_out:
    pickle.dump(regressor_rf, f_out) # write final_model in .bin file
    f_out.close()  # close the file 


# In[ ]:




