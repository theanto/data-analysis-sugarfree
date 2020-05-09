
from functions import *

import warnings
warnings.filterwarnings('ignore')


data= loadFullDf('dataset.csv')

d ={}


for x in range(0 ,len(data)):
    l= [([0])]
    d[x] = preprocess(data[x], 0)
     
    d[x]= d[x].reset_index()

    
    
    
   