
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
import glob
import random
import os
import csv
import json
import sys


def calculateEma(price, interval = 9, startEma = -1):
    #https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    # (Closing price-EMA(previous day)) x multiplier + EMA(previous day)
    k = 2/(interval + 1)
    if startEma > 0:
        return reduce(lambda x,y: x + [ (y - x[-1]) * k + x[-1] ], price, [startEma])
    else:        
        subset = price[0:interval]
        sma = sum(subset) / len(subset)
        start = [sma] * interval    
        return reduce(lambda x,y: x + [ (y - x[-1]) * k + x[-1] ], price[interval:], start) 

def rewindEma(price, interval, startEma):
    k = 2/(interval + 1)   
    return reduce(lambda x,c: x + [ (-c*k + x[-1])  / (-k+1) ], price, [startEma])

def priceChangeToPrice(data, initial = 100):    
    return list(reduce(lambda x,y:  x + [ x[-1]+(x[-1]*y) ], data, list([initial]) ) )

def rewindPriceChangeToPrice(data, initial = 100):
    return list(reduce(lambda x,y:  x + [ x[-1] / (y+1.0) ], data, list([initial]) ) )


def genSeries(data, timeDomains = [1,5,15,30]):
    sample1Min = data[0:180]
    sample1Min = sample1Min[::2]  # only want price
    
    enterPrice = 100.0
    # we need to rewind these values through time now.
    rewindPrice1 = rewindPriceChangeToPrice(sample1Min[::-1], initial=enterPrice)

    graph1 = priceChangeToPrice(sample1Min, initial=rewindPrice1[-1])
    
    series = [graph1]
    ind = 1
    for t in filter(lambda x: x != 1,timeDomains):
        start = (ind*180)
        end = ((ind+1)*180)
        sampleXMin = data[start:end]
        sampleXMin = sampleXMin[::2]
        minutes = 1
        remainder = (60+minutes) % t
        #print("x: "+str(60+time.minute))
        #print("remainder: " + str(remainder) )
        #print(graph1[-(remainder+1)])

        rewindPriceX = rewindPriceChangeToPrice(sampleXMin[::-1], initial=graph1[-(remainder+1)])
        extra = 90*t - 90*timeDomains[ind-1]
        series = [([None] * extra) + x for x in series]

        graphX = priceChangeToPrice(sampleXMin, initial=rewindPriceX[-1])
        graphX = [[x]*t for x in graphX]
        graphX = [val for sublist in graphX for val in sublist][remainder:]
        series.append(graphX)
        ind = ind+1
    return series

def debugPlot(data, timeDomains = [1,5,15,30]):
    sample = random.randint(0,data.shape[0]-1)
    te = data[sample,:]
    series = genSeries(te, timeDomains)
    for x in series:
        plt.plot(x) 
    plt.show()  

def gridPlot(data, timeDomains = [1,5,15,30]):
    #plt.figure(figsize=(6,6))
    for k in range(12):
         sample = random.randint(0,data.shape[0]-1)
         series = genSeries(data[sample,:], timeDomains)
         plt.subplot(4, 3, k+1)
         for x in series:
            plt.plot(x) 
    plt.tight_layout()
    plt.show()
 

def main():
    # print command line arguments
    for arg in sys.argv[1:]:
        print(arg)
    
    # load the np array from the passed in file location
    path = sys.argv[1:][0]
    print("path: " + path)
    #obj = json.load(open(path))
    data = np.loadtxt(path, delimiter="|")
    print(data.shape)
    gridPlot(data, [1,5])

if __name__ == "__main__":
    main()