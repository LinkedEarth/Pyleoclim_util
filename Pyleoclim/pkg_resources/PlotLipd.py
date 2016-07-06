# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:14:26 2016

@author: deborahkhider

Plot a timeseries

"""

from lipd.start import *
import matplotlib.pyplot as plt

class MakePlot(object):
    
    def __init__(self):
        self.time_series = extractTimeSeries() #Extract a time series object
    
        # create the directory for the figure
        if not os.path.exists(path+'/figures'):
            os.makedirs(path+'/figures')
    
    def plot_aLiPD(self, filename = "", \
        plot_style = 'seaborn-ticks', x_min = [], x_max = [], \
        saveFig = True):
        # Check for the file name or ask for one    
        if not filename:
            print("Below are the available time series")
            list_record = []
            for key in self.time_series.keys():
                list_record.append(key)
            for record in list_record:
                print(list_record.index(record), "\t", record)
            num_ans = int(input("Enter the number of the record" +
                " you would like to analyze: "))
            filename = list_record[num_ans]
   
        # Extract the data into several DataFrames        
        dfs_ts = ts_to_df(self.time_series,
                                               filename)

        #Look for the variable name in the columns of the dataframe
        y_axis = [col for col in dfs_ts['paleoData'].columns \
            if filename.split('_')[2] in col][0]
        
        x_axis = [col for col in dfs_ts['paleoData'].columns \
            if 'age' in col or 'year' in col or 'depth' in col]

        #Check for time representation
        if len(x_axis)>1:
            print("The time series object you have selected has several" +
                " time representation.")
            for choice in x_axis:
                print(x_axis.index(choice), "\t", choice)
            user_choice = int(input("Enter the number of the time"+
            " representation you would like to use: "))
            x_axis = x_axis[user_choice]
    
        # Make the plot
        plt.style.use(plot_style)
        plt.plot(dfs_ts['paleoData'][x_axis], dfs_ts['paleoData'][y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        xmin, xmax = plt.xlim()
        if not x_min:
            x_min = xmin
        if not x_max:
            x_max = xmax
        plt.xlim([x_min,x_max]) 

        #Save the figure
        if saveFig == True:           
            plt.savefig(path+'/figures/plot_'+filename+'.eps',\
                bbox_inches='tight',pad_inches = 0.25)
        
    
    

    