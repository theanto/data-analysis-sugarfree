#!/usr/bin/python
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv
import datetime
from pmdarima import auto_arima
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from statsmodels.tools.eval_measures import rmse
from sklearn.svm import SVR

def boxplot(df, name,min,max):
    plt.figure(figsize=(30, 10))
    plt.title(name+ " RMSE from  "+ str(min) +"h PSW to " + str(max) +"h PWS")
    plt.ylabel("RMSE")
    plt.xlabel("")


    box_plot = sns.boxplot(x="Interval" ,y="RMSE" ,data=df, showfliers = False)

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(lines[4+cat*5].get_ydata()[0],3) 
        
        ax.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=10,
            color='white',
            bbox=dict(facecolor='#445A64'))
        

    box_plot.figure.tight_layout()


def loaddataset(file, user):
    series = read_csv(file, header=0, index_col=0, parse_dates=True, squeeze=True)
    data= series.loc[user]
    data = data.reset_index()
    data = data.drop(data[data.sugarValue > 300].index, inplace=False) 
    data = data.fillna(0)
    data = data.drop(data[data.sugarValue < 1].index, inplace=False)
    return data  

def preprocess(data, fun):
    if(fun ==0):
        data = data[['sugarValue', 'time']]
        data['time']=pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
        data.set_index(['time'], inplace=True)

    if(fun ==1):
        data = data.drop(['uid','time'], axis=1)
        float_col = data.select_dtypes(include=['float64']) # This will select float columns only
        #list(float_col.columns.values)
        for col in float_col.columns.values:
                data[col] = data[col].astype('int64')


    return data  


def app(window,  train, test, pred, interval, windo):
    window = window.append({'Current train': "From: "+str(train.index[0]) +" to: " + str(train.index[-1]) , 
                                        'Current test':"From: "+str(test.index[0]) +" to: " + str(test.index[-1]), 
                                        'MSE': np.square(np.subtract(test , pred)).mean(),
                                        'RMSE': rmse(test, pred),
                                        'Interval': str(interval) + "Min" + " "+ str(int(round(windo)))+ "PSW" },ignore_index=True)

    return window


def ARIMA(window, df ,x, n,v, interval, windo):

   
    train = df.iloc[x:n]
    test = df.iloc[n:n+v+1]

    #indces for the train test split
    start = len(train)
    end = start + len(test)-1

    model = SARIMAX(df['sugarValue'], order=(0, 1, 3), seasonal_order=(0, 0, 0, 12), enforce_invertibility=False).fit()
    pred = model.predict(n, n+v)
    
    window = app(window, train['sugarValue'], test['sugarValue'] ,pred, interval, windo)

    return window

def RFSVM(window, df, X, y ,x, n,v, interval, windo,fun):

   
    X_train = X.iloc[x:n]
    X_test = X.iloc[n:n+v+1]
    y_train = y.iloc[x:n]
    y_test = y.iloc[n:n+v+1]
   
    if (fun ==2):
        rfc = RandomForestClassifier()
        rfc.fit(X_train,y_train)
        # predictions
        pred = rfc.predict(X_test)
    if (fun ==3):
        
        svclassifier = SVR()
        #svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train)

        pred = svclassifier.predict(X_test)

    window = app(window, y_train, y_test ,pred, interval, windo)

    return window


def prediction(df, fun,freq,*args, **kwargs):
    for ar in args:
        pass
    
    X= args[0]
    y= args[1]

    start_time = time.time()

    window = pd.DataFrame(columns=['Current train', 'Current test','MSE', 'RMSE', 'Interval'])
   
    list = [12,24,48,96,144] 
    inter = 4
    
    if(freq ==2):
        list = [192,240,288]  ##high PSW
        inter = 12

    for z in list:
        for v in range(0,inter):
            n=z ##3h
            
            interval = (v+1)*15
            windo= n/4
            
            for x in range((len(df)-n-v)):
            
                if fun == 1:                   
                    window = ARIMA(window, df, x, n, v, interval, windo)
                if fun == 2:
                    
                    window = RFSVM(window, df, X,y,x, n, v, interval, windo,fun)
                if fun == 3:
                    window = RFSVM(window, df, X,y,x, n, v, interval, windo,fun)
                n= n+1
                
                
            v= v+1

        
        
    print("--- %s Seconds for computation ---" % (time.time() - start_time))
    return window