import calendar
import csv
import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
import os
import scipy.stats
import sys
import wget


global dateRef; dateRef = datetime.date(1900, 1, 1)

def downloadData(hemi = "north", dataSet = "OSISAF-v2p1"):


    # Retrieving the data source file
    # -------------------------------

    hemi_region = {"south": "Antarctic",
                   "north": "Arctic"   ,
                  }
    hemshort = hemi[0] + hemi[-1]

    if dataSet == "NSIDC-G02135":
        rootdir = "ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/" + hemi + \
                   "/daily/data/"
        filein  = hemi[0].upper() + "_" + "seaice_extent_daily_v3.0.csv"
    elif dataSet == "OSISAF-v2p1":
        rootdir = "ftp://osisaf.met.no/prod_test/ice/index/v2p1/" + hemshort + "/"
        filein = "osisaf_" + hemshort + "_sie_daily.txt"
    else:
        sys.exit("Dataset unknown")

    if os.path.exists("./data/" + filein):
        os.remove("./data/" + filein)
        wget.download(rootdir + filein, out = "./data/")
    else:
        wget.download(rootdir + filein, out = "./data/")


def loadData(hemi = "north", dataSet = "OSISAF-v2p1"):
    # Reading the data
    # ----------------
    
    # Reading and storing the data. We are going to archive the daily extents
    # in a 1-D numpy array. Missing data are still recorded but as NaNs
    # Creating NaNs for missing data is useful because it makes the computation
    # of autocorrelation easier later on.
    # The 29th of February of leap years are excluded for 
    # ease of analysis
    hemshort = hemi[0] + hemi[-1]
    
    if dataSet == "NSIDC-G02135":
        filein  = hemi[0].upper() + "_" + "seaice_extent_daily_v3.0.csv"
        delimiter = ","
        nbIgnoreLines = 1
    elif dataSet == "OSISAF-v2p1":
        filein = "osisaf_" + hemshort + "_sie_daily.txt"
        delimiter = " "
        nbIgnoreLines = 7
        
    # Index for looping through rows in the input file
    j = 0
    
    

    rawData = list()
    
    
    if dataSet == "NSIDC-G02135":
      with open("./data/" + filein, 'r') as csvfile:
        obj = csv.reader(csvfile, delimiter = delimiter)
        for row in obj:
          if j <= nbIgnoreLines:
            pass
          else:
            thisDate = datetime.date(int(row[0]), int(row[1]), int(row[2]))
            thisValue = float(row[3])
            timeElapsed = (thisDate - dateRef).days
            
            if thisDate.month != 2 or thisDate.day != 29:
              rawData.append(
              [timeElapsed, thisDate, thisValue])
            
          j += 1
            
    elif dataSet == "OSISAF-v2p1":
      with open("./data/" + filein, 'r') as csvfile:
        obj = csv.reader(csvfile, delimiter = delimiter)
        for row in obj:
          if j <= nbIgnoreLines:
              pass
          else:
            if row[5]!= "MISSING":
              thisDate = datetime.date(int(row[1]), int(row[2]), int(row[3]))
              thisValue = float(row[4]) / 1e6              
              timeElapsed = (thisDate - dateRef).days

            # Only append if not 29 Feb
              if thisDate.month != 2 or thisDate.day != 29:
                rawData.append(
                [timeElapsed, thisDate, thisValue])
          j += 1   
  
            

        
        
    # Now that we have the raw dates, we can create 
    # a list of items for each date, even those for which there is no data.
        

    # Create list of all dates except 29th of Feb between the first and
    # last dates of rawData
    
    thisDate = rawData[0][1]
    allDates = list()
    
    while thisDate <= rawData[-1][1]:
        
        if thisDate.day != 29 or thisDate.month != 2:
            
            allDates.append(thisDate)

        thisDate += datetime.timedelta(days = 1)
    
    
    
    # Finally, go throught allDates and dump rawData if exists for that date
    counterRaw = 0
    outData = list()
    import time
    
    for d in allDates:
        
        timeElapsed = (d - dateRef).days
        if rawData[counterRaw][1] == d:
            #If there is a match, record it
            thisValue = rawData[counterRaw][2]
            counterRaw += 1
        else:

            thisValue = np.nan
        
        outData.append([timeElapsed, d, thisValue])

    return outData



#downloadData(dataSet = "NSIDC-G02135")
data = loadData(    dataSet = "NSIDC-G02135")


leadTimeMax = 365 # Max lead time in days
dateFirstHindcast = datetime.date(1990, 1, 1)

for d in data:
  thisDate = d[1]
  thisYear = thisDate.year
  thisMonth= thisDate.month
  thisDay  = thisDate.day

  thisSIE = d[2]

  if thisDate >= dateFirstHindcast and ~np.isnan(thisSIE):
    print("Doing " + str(d[1]))
    # For each day of the period we are going to make a deterministic forecast and store it
  
    # 1. Collect as predictors all SIE from previous years matching the day and month 
    #    of the current date. We only consider those dates

    
    predictions = np.full(leadTimeMax, np.nan)

    for leadTime in np.arange(0, leadTimeMax):

      targetDate = thisDate + timedelta(days = float(leadTime))
      targetMonth=targetDate.month
      targetDay  = targetDate.day


      # Suppose we are the 22nd March 1998 and we want to make a prediction out to 400 days
      # We need to take as predictors all the SIE that fell a 22nd of March but such that 400 days past this date
      # is still before the current date
      
      predictors = np.array([dd[2] for dd in data if     dd[1].month == thisMonth   and dd[1].day == thisDay  \
                                         and    dd[1] + timedelta(days = float(leadTime)) <= thisDate])
      
      predictands= np.array([dd[2] for dd in data if     dd[1].month == targetMonth and dd[1].day == targetDay \
                                         and    dd[1] <= thisDate])

      idx = np.isfinite(predictors) * np.isfinite(predictands) # Indexes that are not nan

      p = np.polyfit(predictors[idx], predictands[idx], 1) # Linear regression

      # Make prediction
      prediction = np.polyval(p, thisSIE)
   
      predictions[leadTime] = prediction


    stop()
    # Write hindcast
      
