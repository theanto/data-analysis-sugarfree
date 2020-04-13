from functions import *

import warnings
warnings.filterwarnings('ignore')

PSW=[36,72,144, 288, 432]  
inter= 4
orderArima= (3, 1, 3)



data= loadFullDf('dataset.csv')

data1 ={}
data2 ={}
Arima ={}
RF ={}
SVM ={}
j=1


for x in range(1 ,len(data)):

    data1[x] = preprocess(data[x], 0)
    data2[x]  =preprocess(data[x], 1)
    X = data2[x].drop('sugarValue', axis=1)
    y = data2[x]['sugarValue']

#for x in range(1,2):

    print("Arima "+str(x)+": ")
    Arima[x] = predictionArima(data1[x]) 
    boxplot(Arima[x], "Arima"+str(x))
    
    print("RF "+str(x)+": ")
    RF[x] = predictionRFSVM(data2[x],X,y, RandomForestClassifier()) ##RF
    boxplot(RF[x], "RF"+str(x))

    print("SVM "+str(x)+": ")
    SVM[x] = predictionRFSVM(data2[x],X,y, SVC()) ##svm
    boxplot(SVM[x], "SVM"+str(x))

    Arima[x].to_csv("Arima "+str(x)+".csv", sep='\t')    
    RF[x].to_csv("RF "+str(x)+".csv", sep='\t')
    SVM[x].to_csv("SVM "+str(x)+".csv", sep='\t')

    
    
    joinI("Arima"+str(x),"RF"+str(x),"SVM"+str(x))
    j= j+1