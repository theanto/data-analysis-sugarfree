import os
import pandas as pd
from external import *


def pathos():
    cwd = os.getcwd()  
    os.chdir( '..' )
    cwd = os.getcwd()  
    os.chdir( '..' )
    os.chdir(cwd) 
    # Get the current working directory (cwd)
    #files = os.listdir(cwd)  # Get all the files in that directory
    print(cwd)




path = os.getcwd()
xlsx= []
for dir in os.walk(path+"/Dati CGM/"):
    os.chdir(dir[0])
    files = os.listdir(dir[0])
    files_xls = [f for f in files if f[-4:] == 'xlsx']
    files_xls
    xlsx.append(files_xls) 

xlsx.sort() 

pathos()

i=1

d = {}
        
sys = os.getcwd()
for f in xlsx:
    
    for x in f:
        
      
        path = x[:-8]
        number = x[6:-8]
        user = str(x[6:-5])
        print (path, user)

        
        d[user] = pd.DataFrame()
        data1 = pd.read_excel(str(path)+'/'+str(x) , header=11, parse_dates=['Timestamp'])
        #data1 = pd.read_excel('Dati CGM/' +str(path) +'/'+str(x) , header=11, parse_dates=['Timestamp'])
        d[user] = d[user].append(data1)
        d[user] = d[user][['Timestamp', 'BG Reading (mg/dL)','ISIG Value', 'Sensor Glucose (mg/dL)']]
        d[user]  = d[user].dropna(thresh=3)
        d[user] = d[user].rename({'Sensor Glucose (mg/dL)': 'sugarValue'}, axis=1)

        
        data2= preprocess(d[user], 1)

        X = data2.drop('sugarValue', axis=1)
        y = data2['sugarValue']

        Arima = prediction(d[user],1,1,0,0,5,0,2)
        RF = prediction(data2,2,1, X, y,0,0,0)
        SVM = prediction(data2,3,1, X, y,0,0,0)

        Arima.to_csv("Arima "+str(user)+".csv", sep='\t')
        
        RF.to_csv("RF "+str(user)+".csv", sep='\t')
        
        SVM.to_csv("SVM "+str(user)+".csv", sep='\t')

        boxplot(Arima,user,  "Arima", 3, 72)
        boxplot(RF, user, "RF", 3, 72)
        boxplot(SVM,user, "SVM",3,72)

        joinI(user)