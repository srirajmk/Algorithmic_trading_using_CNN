import pandas as pd
import numpy as np
import math
import csv 
import pandas_ta as ta
import time
import matplotlib.pyplot as plt

etf_name=['1WMT','2GS','3MSFT','4JNJ','5AAPL']
for etf in etf_name: 
 #Labelling
  df = pd.read_csv('/content/drive/My Drive/ATuGAFCNN/Phase1Extraction/{}_phase1.csv'.format(etf), sep=',')
  closeVal=df['Close'].values
  label=[0]*len(closeVal)
  rowCounter=0
  windowSize=15 #days
  while(rowCounter<len(closeVal)):
    if(rowCounter>windowSize):
      windowBeginIndex=rowCounter-windowSize
      windowEndIndex=rowCounter
      windowMidIndex=(windowBeginIndex+windowEndIndex)//2
      
      max=-math.inf
      min=math.inf

      for i in range(windowBeginIndex,windowEndIndex):
        num=closeVal[i]
        if(num>max):
          maxIndex=i
          max=num
        if(num<min):
          minIndex=i
          min=num
      if maxIndex==windowMidIndex:
        label[maxIndex]=2
        #print("Max val is ",closeVal[maxIndex],"at position",maxIndex)
      elif minIndex==windowMidIndex:
        label[minIndex]=1
        #print("Min val is ",closeVal[minIndex],"at position",minIndex)
      else:
        label[rowCounter]=0
    rowCounter+=1
  df=df.assign(Label=label)
  df.to_csv('/content/drive/My Drive/ATuGAFCNN/Phase2Labelling/{}_phase2.csv'.format(etf), index=False)