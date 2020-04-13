#!/usr/bin/python
from __future__ import division
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
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from statsmodels.tools.eval_measures import rmse
from sklearn.svm import SVR
from PIL import Image
import os



import warnings
warnings.filterwarnings('ignore')

## Global variable


PSW=[36,72 , 144, 288, 432]  
inter= 4
orderArima= (5, 1, 2)


def boxplot(df, name):
    
    sns.set(font_scale=2)

    g = sns.catplot(kind='box', data=df, x='Prediction Horizon (minutes)', y='Glycemia prediction RMSE (mg/dl)', col='PSW', height=10, aspect=0.8, palette='Greys', showfliers = False)

    g.fig.subplots_adjust(wspace=0)

    plt.subplots_adjust(top=0.8)
    
    g.fig.suptitle(str(name)+' Past sliding window (PSW)') 

    # remove the spines of the axes (except the leftmost one)
    # and replace with dasehd line
    for ax  in g.axes.flatten()[0:]:
        ax.spines['left'].set_visible(False)
        [tick.set_visible(True) for tick in ax.yaxis.get_major_ticks()]
        xmin,xmax = ax.get_xlim()
        ax.axvline(xmin, ls='--', color='k')

        
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
                size=15,
                color='white',
                bbox=dict(facecolor='#445A64'))
            
        for i, ax in enumerate(g.axes.flat):
            g.axes[0,i].set_xlabel('')

        g.axes[0,3].set_xlabel('Predictive Horizon (minutes)')   
    
    plt.savefig(str(name) +".jpg")


def loaddataset(file, user):
    series = read_csv(file, header=0, index_col=0, parse_dates=True, squeeze=True)
    data= series.loc[user]
    data = data.reset_index()
    data = data.drop(data[data.sugarValue > 300].index, inplace=False) 
    data = data.fillna(0)
    data = data.drop(data[data.sugarValue < 1].index, inplace=False)
    return data  

def cleanCGM(df, path, x):
    
    data1 = pd.read_excel(str(path)+'/'+str(x) , header=11, parse_dates=['Timestamp'])
    #data1 = pd.read_excel('Dati CGM/' +str(path) +'/'+str(x) , header=11, parse_dates=['Timestamp'])
    df = df.append(data1)
    df = df[['Timestamp', 'BG Reading (mg/dL)','ISIG Value', 'Sensor Glucose (mg/dL)']]
    df  = df.dropna(thresh=3)
    df =df.rename({'Sensor Glucose (mg/dL)': 'sugarValue','Timestamp': 'time'}, axis=1)
    
    return df

def loadFullDf(file):
    
    series = read_csv(file, header=0, index_col=0, parse_dates=True, squeeze=True)
    dfusers = pd.DataFrame()
    dfusers = series.reset_index() 
    uid= dfusers.uid.unique()
    
    dataf = {}
    ind= 0
    for x in uid: 
        #if (os.path.isdir('uid/'+str(ind)+'/')  == False) :
            #os.mkdir('uid/'+str(ind))
        print("User "+str(ind),  x)
        dataf[ind] = pd.DataFrame()
        dataf[ind]= series.loc[x]
        dataf[ind] = dataf[ind].reset_index()
        #dataf[ind] = dataf[ind].drop(dataf[ind][dataf[ind].sugarValue > 300].index, inplace=False) 
        dataf[ind] = dataf[ind].fillna(0)
        dataf[ind] = dataf[ind].drop(dataf[ind][dataf[ind].sugarValue < 1].index, inplace=False)
        ind = ind+1

    return dataf

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
    
    if(fun ==2):
        data = data.drop(['time'], axis=1)
        data = data.fillna(0)
        float_col = data.select_dtypes(include=['float64']) # This will select float columns only
        #list(float_col.columns.values)
        for col in float_col.columns.values:
                data[col] = data[col].astype('int64')

    return data  


def app(window,  train, test, pred, interval, windo):

    window = window.append({'Current test': test.values, 
                                        'Current prediction': pred, 
                                        'MSE': np.square(np.subtract(test.values , pred)).mean(),
                                        'Glycemia prediction RMSE (mg/dl)': rmse(test.values, pred),
                                        'PSW': int(round(windo)) ,
                                         'Prediction Horizon (minutes)': interval },ignore_index=True)

    return window


def joinI(arima, rf, svm):
    

    im1 = Image.open(arima+'.jpg')
    im2 = Image.open(rf+'.jpg')
    im3 = Image.open(svm+'.jpg')

    imageL = [im1, im2, im3]
    min_width = min(im.width for im in imageL)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)),resample=Image.BICUBIC)
                      for im in imageL]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    
    dst.save("Result"+arima+rf+svm+".jpg")
    
    os.remove(arima+'.jpg')
    os.remove(rf+'.jpg')
    os.remove(svm+'.jpg')



### new  prediction 


def predictionArima(df):
    start_time = time.time()
    window = pd.DataFrame(columns=['Current test', 'Current prediction','MSE', 'Glycemia prediction RMSE (mg/dl)', 'PSW', 'Prediction Horizon (minutes)'])

    for n in PSW:
        for v in range(0,inter):
                   
            interval = (v+1)*15
            windo= n/12
            
            for x in range((len(df)-n-v)):

                #print(v, x)            
                train = df.iloc[x:n+x]
                test = df.iloc[n+x:n+x+v+1]

               
                model = SARIMAX(train, order=orderArima    ,enforce_stationarity=False , enforce_invertibility=False).fit()
        
                

                #pred = result.predict(start= n+x, end= n+x+v, exog= test['sugarValue'])
                pred = model.forecast(step = v+1)
                pred= pred.values
                #model = SARIMAX(df['sugarValue'], order=(0, 1, 3), seasonal_order=(0, 0, 0, 12), enforce_invertibility=False).fit()
                #pred = result.predict(n, n+v)
                
                
                window = app(window, train, test['sugarValue'] ,pred, interval, windo)
             
            v= v+1
        
    print("--- %s Seconds for computation ---" % (time.time() - start_time))
    return window


def predictionRFSVM(df, X, y, classi):
    start_time = time.time()
    window = pd.DataFrame(columns=['Current test', 'Current prediction','MSE', 'Glycemia prediction RMSE (mg/dl)', 'PSW', 'Prediction Horizon (minutes)'])


    for n in PSW:
        for v in range(0,inter):
            
            #n=z ##3h
            
            interval = (v+1)*15
            windo= n/12
            
            for x in range((len(df)-n-v)):

                X_train = X.iloc[x:n+x]
                X_test = X.iloc[n+x:n+x+v+1]
                y_train = y.iloc[x:n+x]
                y_test = y.iloc[n+x:n+x+v+1]

                #rfc = RandomForestClassifier()
                #svclassifier = SVR()
                classifit = classi
                classifit.fit(X_train,y_train)
                # predictions
                pred = classifit.predict(X_test)
                
                
                window = app(window, y_train, y_test ,pred, interval, windo)
                x=x+1
                        
            v= v+1
        n=n+1
    print("--- %s Seconds for computation ---" % (time.time() - start_time))
    return window


