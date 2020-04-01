from functions2 import *

import warnings
warnings.filterwarnings('ignore')



data= loaddataset('dataset.csv')

data1 ={}
data2 ={}
Arima ={}
RF ={}
SVM ={}


for x in data:
    
    data1[x] = preprocess(data[x], 0)
    data2[x]  =preprocess(data[x], 1)



for x in data:
    ax = data1[x].plot(figsize=(15,8))

for x in data:
     Arima[x] = prediction(data1[x],1,0,0) ##Arima

for x in data:

    X = data2[x].drop('sugarValue', axis=1)
    y = data2[x]['sugarValue']
    RF[x] = prediction(data2[x],2,X,y) ##RF
    SVM[x] = prediction(data2[x],3,X,y) ##svm


j=0
for x in data:
    
    boxplot( Arima[x], "Arima",j, 3, 36)
    boxplot( RF[x], "RF",j, 3, 36)
    boxplot( SVM[x], "SVM",j, 3, 36)
    joinI(j)
    j= j+1