from functions2 import *

import warnings
warnings.filterwarnings('ignore')



data= loaddataset('dataset.csv')

data1 ={}
data2 ={}
Arima ={}
RF ={}
SVM ={}
j=1

for x in data:
    
    data1[x] = preprocess(data[x], 0)
    data2[x]  =preprocess(data[x], 1)




for x in range(1 ,len(data)):
#for x in range(1,2):
    print("Arima "+str(x)+": ")
    Arima[x] = prediction(data1[x],1,0,0,j) ##Arima
 

    X = data2[x].drop('sugarValue', axis=1)
    y = data2[x]['sugarValue']
    print("RF "+str(x)+": ")
    RF[x] = prediction(data2[x],2,X,y,j) ##RF
    print("SVM "+str(x)+": ")
    SVM[x] = prediction(data2[x],3,X,y,j) ##svm
    boxplot(Arima[x], "Arima", x)
    boxplot(RF[x], "RF", x)
    boxplot(SVM[x], "SVM", x)
    #joinO("SVM",j)
    joinI(j)
    j= j+1