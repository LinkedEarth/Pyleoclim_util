# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:00:07 2016

@author: deborahkhider

Use cartopy to make a map of the LiPD files. 

"""

# Import

from lipd.start import *
import matplotlib.pyplot as plt
import numpy as np
import cartopy
import cartopy.crs as ccrs
import sys

class MakeMap(object):

    def __init__(self):
        # Organize the data        

        files = os.listdir(path)
        lipd_in_directory = [i for i in files if i.endswith('.lpd')]
                     
        self.data = pd.DataFrame({'A': np.array([0] * len(lipd_in_directory)),
                     'B': np.array([0] * len(lipd_in_directory)),
                     'C': 'proxy',
                     'D':'name'})                   

        index = 0

        for i in lipd_in_directory:
            d = getMetadata(i)
            self.data.iloc[index,0]=d['geo']['geometry']['coordinates'][1]
            self.data.iloc[index,1]=d['geo']['geometry']['coordinates'][0]
            #TODO: Fix LiPD. For now override ice core as glacier ice
            if d['archiveType'] == 'ice core':
                self.data.iloc[index,2]='glacier ice'
            else:
                self.data.iloc[index,2]=d['archiveType']
            self.data.iloc[index,3]=i
            index+=1       
        
        # Create the directory for the figures
        if not os.path.exists(path+'/figures'):
            os.makedirs(path+'/figures')
            
        # Get the default paletter by proxy type
        self.plot_default = {'borehole': ['grey','p'],
                'coral': ['orange','o'],
                'documents':['#6B4226','o'],
                'glacier ice':['#74BBFB', 'd'],
                'hybrid': ['k','*'],
                'lake sediment': ['m','s'],
                'marine sediment': ['#6B4226', 's'],
                'sclerosponge' : ['m','o'],
                'speleothem' : ['r','d'], 
                'tree' : ['g','^']}    
                
    
    def map_all(self, markersize = int(50), saveFig = True):
        """        
        Make a map of all available records
        """
        ax = plt.axes(projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black', \
            facecolor = ( 0.9, 0.9, 0.9))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.75, box.height])
    
        already_plotted = [] # Check if the data has already been plotted

        for archiveType in self.data['C']:
            if archiveType in self.plot_default and \
            archiveType not in already_plotted:
                ax.scatter(self.data[self.data['C']==archiveType]['A'],
                    self.data[self.data['C']==archiveType]['B'],
                    s = markersize,
                    facecolor = self.plot_default[archiveType][0],
                    edgecolor = 'k',
                    marker = self.plot_default[archiveType][1],
                    transform=ccrs.Geodetic(),
                    label = archiveType)
                already_plotted.append(archiveType)
            elif archiveType not in plot_default:
                ax.scatter(self.data[self.data['C']==archiveType]['A'],
                    self.data[self.data['C']==archiveType]['B'],
                    s = markersize,
                    facecolor = 'k',
                    edgecolor = 'k',
                    marker = 'o',
                    transform=ccrs.Geodetic(),
                    label = 'other')    
        
        ax.legend(loc = 'center', bbox_to_anchor=(1.25,0.5),scatterpoints = 1,
                   frameon = False, fontsize = 8, markerscale = 0.7)
        if saveFig == True:           
            plt.savefig(path+'/figures/map_all_LiPDs.eps',\
                bbox_inches='tight',pad_inches = 0.25)
        plt.show()

    def map_aLiPD(self, name="",gridlines = False, borders = True, \
        topo = True, markersize = int(50), marker = "Default", \
        saveFig = True):
        """
        Mapp one particular record. 
        """
        # Check whether the record name was provided
        if not name:
            print("Below are the available records.")
            print(self.data.loc[:,'D'])
            answer = int(input("Enter the number of the record"+
            " you wish to analyze: "))
            name = self.data.iloc[answer]['D']
    
        if self.data[self.data['D']==name].empty:
            print("ERROR: The name you have entered is " +
            "not in the current directory. Make sure you entered "+
            "the name with the .lpd extension.")
            sys.exit(0)
    
        record = self.data[self.data['D']==name]
        ax = plt.axes(projection=ccrs.Orthographic(record['A'].iloc[0], \
                                               record['B'].iloc[0]))
        if topo == True:
            ax.stock_img()
            ax.add_feature(cartopy.feature.LAND, 
               edgecolor='black', facecolor='none')
        else:
            ax.add_feature(cartopy.feature.LAND, 
               edgecolor='black', facecolor=[0.9375, 0.9375, 0.859375],\
               zorder = 0)               
        if borders == True:
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-')
        if gridlines == True:   
            ax.gridlines()
        ax.set_global()
        if marker == "Default":
            if record['C'].iloc[0] in self.plot_default:    
                ax.scatter(record['A'].iloc[0], record['B'].iloc[0],
                    s = markersize,
                    facecolor = self.plot_default[record['C'].iloc[0]][0],
                    edgecolor = 'k',
                    marker = self.plot_default[record['C'].iloc[0]][1],
                    transform=ccrs.Geodetic()) 
            else:
                ax.scatter(record['A'].iloc[0], record['B'].iloc[0],
                    s = markersize,
                    facecolor = 'k',
                    edgecolor = 'k',
                    marker = 'o',
                    transform=ccrs.Geodetic())
        else:
            ax.scatter(record['A'].iloc[0], record['B'].iloc[0],
                    s = markersize,
                    facecolor = marker[0],
                    edgecolor = 'k',
                    marker = marker[1],
                    transform=ccrs.Geodetic())
        if saveFig == True:            
            plt.savefig(path+'/figures/map_'+os.path.splitext(name)[0]+'.eps',\
                bbox_inches='tight',pad_inches = 0.25)
        plt.show()