#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:08:22 2017

@author: deborahkhider

Pyleoclim Bchron Module

"""

import numpy as np
import pandas as pd
import datetime
import rpy2.robjects as robjects
from itertools import chain
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
import matplotlib.patches as mpatches
import sys


def chooseCalCurves():
    """ Prompt for a calibration curve if not given by the user.
   
    Prompt the user for the name of a calibration curve used to run the Bchron 
    software package. The user can enter either enter only one name that will
    be applied to each age or a list of names of different ages. To enter a list, 
    separate each name with a comma. No quotation marks needed. 
    
    Returns:
        A list of calibration curves to be applied  
    
    """
    
    # Prompt the user
    print('Please select a calibration curve')
    print('If you want to use the same calibration curve ' + 
          'for all data points, enter the name once.')
    print('If you need to use a different calibration curve, enter each one' +
          ' separated by a coma. The list needs to be the same length' +
          ' as the number of datapoints')
    answer = input('Enter either "intcal13", "shcal13", "marine13", or "normal": ')
    
    # Transform the answer into a list
    CalCurves = answer.split(sep = ',')
    # Remove white space if any
    CalCurves = [i.strip(' ') for i in CalCurves]
    
    return CalCurves
    
def verifyCalCurves(CalCurves):    
    """ Verify that the list of calibration curves contain valid entries for BChron
    
    Performs a verification that the names entered in the list of calibration
    curves are conformed to the names used by Bchron. Will prompt the user to
    replace the name if needed. 
    
    Agrs:
        CalCurves (list): A list of strings containing the name(s) of the
            calibration curves to be used
    
    Returns:
        CalCurves - Checked list of calibration curves names        
    
    """
    
    # Check for the right input
    print("Checking that list contains valid calibration curves ...")
    for idx, item in enumerate(CalCurves):
        if item != 'intcal13' and item != 'shcal13' and item != 'marine13' \
        and item!= 'normal':
            print(item + " is not a valid selection in the list.")
            new_answer = input('Enter either "intcal13", "shcal13", '+
                   '"marine13", or "normal": ')
            while new_answer != 'intcal13' and new_answer != 'shcal13' \
            and new_answer != 'marine13' and new_answer != 'normal':
                new_answer = input('Enter either "intcal13", "shcal13", '+
                   '"marine13", or "normal": ')
            CalCurves[idx] = new_answer
    print("Verification complete!") 
    
    return CalCurves               
            
def reservoirAgeCorrection():            
    """ Estimate reservoir age correction

    Assists in estimating the reservoir age correction for marine records.
    If unknown, will direct the user to copy and paste the table available
    on the 14Chrono Marine Reservoir database: http://intcal.qub.ac.uk/marine/
    
    Returns:
        ageCorr -  the DeltaR for the site. \n
        ageCorrStd - The error on DeltaR estimated as the standard error on the 
        mean if using the 14Chrono Marine Reservoir database.
    
    """
    print("The marine13 calibration curve includes the standard reservoir " +
         "age correction for the ocean of 400 years. If you do not wish "+
         "to enter an additional reservoir age correction or if the "+
         "reservoir age correction has already been added to the "+
         "reported ages, answer no to the question below to "+
         "skip this step")
    answer = input("Do you wish to add an additional correction [Y/N]: ")
    while answer != "Y" and answer != "N":
        answer = input("Enter either Y or N: ")
    if answer == 'N':
        ageCorr = 0 
        ageCorrStd = 0
    else:
        prompt = input("Do you know the correction and associated "+
                     "uncertainty? [Y/N]: ") 
        while prompt != "Y" and prompt != "N":
          prompt = input("Enter either Y or N: ")
        if prompt == "Y":
            ageCorr = float(input("Enter the reservoir age correction: "))
            ageCorrStd = float(input("Enter the reservoir age "+
                               "correction uncertainty: "))
        else:
            print("We recommend using the 14Chrono Marine Reservoir "+
                "Database: http://intcal.qub.ac.uk/marine/")
            print("Enter the coordinate for your site and click 'Get Data'")
            print("Copy the table (WITH ALL THE LINES AND COLUMNS, "+
                                 "including headers)")
            print("When ready, press Enter at the prompt. This will copy "+
                "the table into Python and calculate the mean and "+
                "standard deviation for you.")
            input("Press Enter to continue...")
          
            data = pd.read_clipboard()
          
            # Check the work
            print(data)
            correct = input("Is this correct? [Y/N]: ")
            while correct != "Y":
                print("Try again. Make sure to include all the columns, "+
                    "including the blank cell next to MapNo.")
                input("Press Enter to continue...")
                data = pd.read_clipboard()
                print(data)
                correct = input("Is this correct? [Y/N]: ")
              
            # Calculate mean and std deviation  
            ageCorr =  data.mean()['DeltaR']
            ageCorrStd =  data.std()['DeltaR']
     
    return ageCorr, ageCorrStd                         

def runBchron(ages, agesStd, positions, rejectAges = None,\
           positionsThickness = None, \
           calCurves = None, reservoirAgeCorr = None,
           outlierProbs = None,\
           predictPositions = None, iterations = 10000,\
           burn = 2000, thin = 8, \
           extractDate = 1950-datetime.datetime.now().year,\
           maxExtrap = 500, thetaMhSd = 0.5, muMhSd = 0.1, psiMhSd = 0.1, \
           ageScaleVal = 1000, positionScaleVal = 100):
    
    """ Age model for Tie-Point chronologies
    
    Fits a non-parametric chronology model to age/position data according to
    the Compound Poisson-Gamma model defined by Haslett and Parnell (2008). 
    This version used a slightly modified Markov chain Monte-Carlo fitting
    algorithm which aims to converge quicker and requires fewer iterations.
    It also a slightly modified procedure for identifying outliers.
    
    The Bchronology functions fits a compounf Poisson-Gamma distribution to the
    incrememnts between the dated levels. This involves a stochastic linear
    interpolation step where the age gaps are Gamma distributed, and the position
    gaps are Exponential. Radiocarbon and non-radiocarbon dates (including outliers)
    are updated within the fucntion also by MCMC.
    
    Args:
        ages (array): A vector of ages (most likely 14C)
        ageSds (array): A vector of 1-sigma values for the ages given above
        positions (array): Position values (e.g. depths) for each age
        rejectAges (vector): A vector of 1/0 where 1 include the dates to be rejected. 
            Default it None.
        positionsThickness (array): (Optional) Thickness values for each of the positions.
            The thickness values should be the full thickness value of the
            slice. By default set to zero.
        calCurves (list): (Optional) A vector of values containing either 'intcal13',
            'marine13', 'shcal13', or 'normal'. If none is provided, will
            prompt the user. Should be either of length =1 if using the same
            calibration for each age or the same length as the vector of ages.
        reservoirAgeCorr (array): (Optional) A list (matrix) of two floats that correspond to the
            DeltaR and DeltaR uncertainty. If already added to the ages and
            ages standard deviation, then enter [0,0] to bypass the prompt.
            Will only be applied if CalCurves is set to 'marine13'. Otherwise,
            leave to none.
        outlierProbs (array): (Optional) A vector of prior outlier probabilities,
            one for each age. Defaults to 0.01
        predictPositions (array): (Optional) a vector of positions 
            (e.g. depths) at which predicted age values are required. 
            Defaults to a sequence of length 100 from the top position to the
            bottom position.
        iterations (int): (Optional) The number of iterations to start the procedure. 
            Default and minimum should be 10000.
        burn (int): (Optional) The number of starting iterations to discard.
            Default is 200
        thin (int): (Optional) The step size for every iteration to keep beyond
            the burnin. Default is 8.
        extractDate (float): (Optional) The top age of the core. Used for
            extrapolation purposes so that no extrapolated ages go beyond the
            top age of the core. Defaults to the current year.
        maxExtrap (int): (Optional) The maximum number of extrapolations to
            perform before giving up and setting the predicted ages to NA. 
            Useful for when large amounts of extrapolation are required, i.e.
            some of the predictPositions are a long way from the dated
            positions. Defaults to 500. 
        thetaMhSd (float):  (Optional)  The Metropolis-Hastings standard
            deviation for the age parameters. Defaults to 0.5.
        muMhSd (float): (Optional)  The Metropolis-Hastings standard deviation
            for the compound Poisson-Gamma Scale. Defaults to 0.1
        psiMhSd (float): (Optional) The Metropolis-Hastings standard deviation 
            for the Compound Poisson-Gamma Scale.
        ageScaleVal (int): (Optional) A scale value for the ages. 
            Bchronology works best when the ages are scaled to be 
            approximately between 0 and 100.
            The default value is thus 1000 for ages given in years.
        positionScaleVal (int):  (Optional) A scale value for the positions. 
            Bchronology works best when the positions are scaled to be 
            approximately between 0 and 100. The default value is thus
            100 for positions given in cm.
    
        Returns:
            depth - the predicted positions (either same as the user or the default) \n
            chron -  a numpy array of possible chronologies in each column.
                The number of rows is the same as the length of depth
            ageDist - the distribution of ages around each dates.
            run - the full R object containing the outputs of the Bchron run
        
        Warnings:
            This function requires R and the Bchron package and all its
                dependencies to be installed on the same machine.
                
        Reference:
            - Haslett, J., and Parnell, A. C. (2008). A simple monotone 
                process with application to radiocarbon-dated depth 
                chronologies. Journal of the Royal Statistical Society, 
                Series C, 57, 399-418. DOI:10.1111/j.1467-9876.2008.00623.x
            - Parnell, A. C., Haslett, J., Allen, J. R. M., Buck, C. E., 
                and Huntley, B. (2008). A flexible approach to assessing 
                synchroneity of past events using Bayesian reconstructions
                of sedimentation history. Quaternary Science Reviews, 
                27(19-20), 1872-1885. DOI:10.1016/j.quascirev.2008.07.009     
    
    """
       
    #r = robjects.r 
    
    # Install necessary R packages
    utils = importr("utils")
    #Choose a CRAN Mirror for download in necessary
    utils.chooseCRANmirror(ind=1)
    # Install the various packages if needed
    try:
        #Import BChron 
        Bchron = importr('Bchron')
    except:
        packnames = ('Bchron', 'stats', 'graphics')
        utils.install_packages(robjects.vectors.StrVector(packnames))
        #Import Bchron
        Bchron = importr('Bchron')  
    
    ages = np.array(ages, dtype='float64')
    agesStd = np.array(agesStd, dtype='float64')
    positions = np.array(positions, dtype='float64')

    #Make sure that the vectors are of the same length
    assert np.size(positions) == np.size(ages) == np.size(agesStd)
    
    if positionsThickness:
        positionsThickness = np.array(positionsThickness, dtype='float64')
        assert np.size(positionsThickness)==np.size(positions)
           
    # Get the calCurve if not given
    if not calCurves:
        calCurves = chooseCalCurves()
        calCurves = verifyCalCurves(calCurves)
        if len(calCurves) == 1 and len(positions)!=1:
            calCurves = list(chain(*[[i]*len(positions) for i in calCurves]))        
    elif len(calCurves)==1 and len(positions)!=1:
        calCurves = verifyCalCurves(calCurves)
        calCurves = list(chain(*[[i]*len(positions) for i in calCurves]))
    else:
        calCurves = verifyCalCurves(calCurves) 
    
    assert len(calCurves) == len(positions)
    # Make sure it's an array
    calCurves = np.array(calCurves)
    
    # Ask for the reservoir age correction
    # Get the indices where calCurves is marine13
    idx = [i for i, j in enumerate(calCurves) if j == 'marine13']
    
    if idx:
        if type(reservoirAgeCorr) != np.ndarray:
            if type(reservoirAgeCorr) == list:
                assert len(reservoirAgeCorr) == 2
                reservoirAgeCorr = np.array(reservoirAgeCorr)
                ageCorr = reservoirAgeCorr[0]
                ageCorrStd = reservoirAgeCorr[1]
                for i in idx:
                    ages[i] = np.floor(ages[i]-ageCorr)
                    agesStd[i] = np.floor(np.sqrt(agesStd[i]**2 + ageCorrStd**2))
                         
            else:
                ageCorr, ageCorrStd =  reservoirAgeCorrection()
        else:
            s = np.shape(reservoirAgeCorr)
            if len(s) > 1:
                assert s[1] == 2
                if s[0] > 1:
                    assert s[0] == len(positions)
                ageCorr = reservoirAgeCorr[:,0]
                ageCorrStd = reservoirAgeCorr[:,1]
                for i in idx:
                    ages[i] = np.floor(ages[i]-ageCorr[i])
                    agesStd[i] = np.floor(np.sqrt(agesStd[i]**2 + ageCorrStd[i]**2))
            else:
                assert s[0] == 2
                ageCorr = reservoirAgeCorr[0]
                ageCorrStd = reservoirAgeCorr[1]          
                for i in idx:
                    ages[i] = np.floor(ages[i]-ageCorr)
                    agesStd[i] = np.floor(np.sqrt(agesStd[i]**2 + ageCorrStd**2))
        
    # Make sure that there is at least 10000 iterations
    if iterations <10000:
        iterations = 10000        
    
    # Get the thickness
    if positionsThickness == None:
        positionsThickness = np.zeros(np.shape(positions))        
    
    # Get the outlier probs
    if outlierProbs == None:
        outlierProbs = 0.01*np.ones(np.shape(positions))        
            
    # Get the predicted positions are give the default
    if predictPositions == None:
        predictPositions = np.linspace(np.min(positions), np.max(positions), 100)        
    
    #Deal with rejected ages if needed
    if rejectAges:
        assert np.size(rejectAges) == np.size(ages) #make sure it's the right size
        # Remove the rejected ages
        ages = ages[np.where(rejectAges==0)]
        agesStd = agesStd[np.where(rejectAges==0)]
        positions = agesStd[np.where(rejectAges==0)]
        ageCorr = ageCorr[np.where(rejectAges==0)]
        ageCorrStd = ageCorrStd[np.where(rejectAges==0)]
        positionsThickness = positionsThickness[np.where(rejectAges==0)]
        calCurves = calCurves[np.where(rejectAges==0)]
        outlierProbs = outlierProbs[np.where(rejectAges==0)]
        
    # Export for R
    
    ages_r = robjects.FloatVector(ages)
    sd_r = robjects.FloatVector(agesStd)
    pos_r = robjects.FloatVector(positions)
    calCurves_r = robjects.vectors.StrVector(calCurves)   
    positionsThickness_r = robjects.FloatVector(positionsThickness)    
    outlierProbs_r =  robjects.FloatVector(outlierProbs)
    predictPositions_r = robjects.FloatVector(predictPositions)  
    
    # Run Bchronology from Bchron
    
    run = Bchron.Bchronology(ages = ages_r, ageSds = sd_r, positions = pos_r,\
                             positionThicknesses = positionsThickness_r, \
                             calCurves = calCurves_r, \
                             outlierProbs = outlierProbs_r,\
                             predictPositions = predictPositions_r,\
                             iterations = iterations, burn = burn,
                             thin = thin, extractDate = extractDate,\
                             maxExtrap = maxExtrap, thetaMhSd = thetaMhSd,\
                             muMhSd = muMhSd, psiMhSd = psiMhSd, \
                             ageScaleVal = ageScaleVal, \
                             positionScaleVal = positionScaleVal) 
    
    # Extract the needed information for plotting purposes
    # theta is the posterior distribution for each of the ages (distribution tables)
    ageDist = np.array(run[0])
    
    # Theta predict containst the interpolated age models at the predicted positions
    chron = np.transpose(np.array(run[4]))
    
    # Get the depths
    depth = predictPositions
    
    return depth, chron, ageDist, run

def plotBchron(depth, chron, positions, ageDist, flipCoor= False,\
               xlabel = 'Depth', ylabel = 'Age', xlim = None, ylim = None, \
               violinColor = '#8B008B', medianLineColor = 'black',
               medianLineWidth = 2.0, CIFillColor = 'Silver', \
               samplePaths = True, samplePathNumber  = 10,\
               alpha = 0.5, figsize = [4,8], ax = None):
    """ Plot a Bchron output
    
    This function creates a plot showing the calibrated calendar ages and
    associated 95% confidence interval as error bars, the 95% ensemble from
    the produced age model as well as randomly drawn members of the ensemble.
    
    Args:
        depth (array): the positions in the archive (often referred to as
            depth) where the age model was interpolated to. Should be a vector
        chron (array): The possible age models returned by BChron. The number
            of rows should be the same length as the depth vector, with each
            possible realization stored in the columns.
        positions (array): The depth on the archive at which chronological
            measurements have been made. Should be a vector 
        agesDist (array): The distribution of ages for each chronological tie
            points. The number of columns should correspond to the number of
            chronological tie points available.
        flipCoor (bool): If True, plots depth on the y-axis.     
        xlabel (str): The label for the x-axis
        ylabel (str): The label for the y-axis
        xlim (list): Limits for the x-axis. Default corresponds to the min/max
            of the depth vector.
        ylim (list): Limits for the y-axis. Default set by matplotlib
        violinColor (str): The color for the violins. Default is purple
        medianLineColor (str): The color for the median line. Default is black.
        medianLineWidth (float): The width for the median line
        CIFillColor (str): Fill color in between the 95% confidence interval.
            Default is silver.
        samplePaths (bool): If True, draws sample paths from the distribution.
            Use the same color as the violins. 
        samplePathNumber (int): The number of sample paths to draw. Default is 10.
            Note: samplePaths need to be set to True. 
        alpha (float): The violins' transparency. Number between 0 and 1
        figsize (list): The figure size. Default is [4,8]
        ax: Default is None. Useful to set for subplots. 
            
    Returns:
        - fig: the figure.      
          
    """
    
    # Make sure np.arrays are given
    depth = np.array(depth)
    chron = np.array(chron)
    positions = np.array(positions)
    ageDist = np.array(ageDist)
    
    # Make a few assertion
    if len(depth)!=np.shape(chron)[0]:
        sys.exit("The number of rows in chron should match the length of depth")
    if len(positions)!=np.shape(ageDist)[1]:
        sys.exit("The number of columns in ageDist should match the length of positions")    
    
    # Get the various quantiles for the plot
    chron_Q = mquantiles(chron, prob=[0.025, 0.5, 0.975], axis=1)
    nchrons = chron.shape[1]

    # Set the figure display
    if not ax:
        fig,ax = plt.subplots(figsize=figsize)
    # Set to ggplot style
    plt.style.use('ggplot')    
    
    if flipCoor is True:
        # PLot with depth on y-axis. 
        plt.fill_betweenx(depth, chron_Q[:,0], chron_Q[:,2], facecolor=CIFillColor,\
                     edgecolor=CIFillColor,lw=0.0, zorder = 1)
        CI = mpatches.Patch(color=CIFillColor) # create proxy artist for labeling
        med, = ax.plot(chron_Q[:,1],depth,color = medianLineColor, \
                       lw=medianLineWidth, zorder=5)
        lbl = ['95% CI','median']
    
        #plot a few random paths
        if samplePaths is True:
            nl = samplePathNumber
            idx = np.random.randint(nchrons+1, size=nl)
            l = ax.plot(chron[:,idx],depth,lw=0.5,color = violinColor,zorder=2)
            lbl.append('sample paths')
    
        # Plot the age tie points 
        parts = ax.violinplot(ageDist,positions,vert = False,\
                              points=200,widths=0.5,showmedians=False,\
                          showextrema=False, showmeans = False)
        for pc in parts['bodies']:
            pc.set_facecolor(violinColor)
            pc.set_edgecolor('black')
            pc.set_alpha(alpha)
            pc.set_zorder(200)
        lbl.append('chronological tiepoints') 
        lbl = tuple(lbl)
        # plot the legend
        if samplePaths is True:
            lg = plt.legend((CI,med,l[1], pc),lbl,loc='upper right'); lg.draw_frame(False)
        else:
            lg = plt.legend((CI,med, pc),lbl,loc='upper right'); lg.draw_frame(False)
        
        # Work on the axes
        ax.grid(axis='y'); 
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xlabel(xlabel,fontsize=14) 
        plt.ylabel(ylabel,fontsize=14)
        if xlim==None:
            ax.set_xlim([np.min(chron_Q[:,1]),np.max(chron_Q[:,1])])
        else:
            ax.set_xlim(xlim)    
        if ylim==None:
            ax.set_ylim([np.min(depth),np.max(depth)])
        else:    
           ax.set_ylim([ylim]) 
        plt.gca().invert_yaxis()
    else:
          # Plot the median and confidence interval
        plt.fill_between(depth, chron_Q[:,0], chron_Q[:,2], facecolor=CIFillColor,\
                         edgecolor=CIFillColor,lw=0.0, zorder = 1)
        CI = mpatches.Patch(color=CIFillColor) # create proxy artist for labeling
        med, = ax.plot(depth,chron_Q[:,1],color = medianLineColor, \
                       lw=medianLineWidth, zorder=5)
        lbl = ['95% CI','median']
    
        #plot a few random paths
        if samplePaths is True:
            nl = samplePathNumber
            idx = np.random.randint(nchrons+1, size=nl)
            l = ax.plot(depth,chron[:,idx],lw=0.5,color = violinColor,zorder=2)
            lbl.append('sample paths')
    
        # Plot the age tie points 
        parts = ax.violinplot(ageDist,positions,points=200,widths=0.5,showmedians=False,\
                          showextrema=False, showmeans = False)
        for pc in parts['bodies']:
            pc.set_facecolor(violinColor)
            pc.set_edgecolor('black')
            pc.set_alpha(alpha)
            pc.set_zorder(200)
        lbl.append('chronological tiepoints') 
        lbl = tuple(lbl)
        # plot the legend
        if samplePaths is True:
            lg = plt.legend((CI,med,l[1], pc),lbl,loc='upper right'); lg.draw_frame(False)
        else:
            lg = plt.legend((CI,med, pc),lbl,loc='upper right'); lg.draw_frame(False)
        
        # Work on the axes
        ax.grid(axis='y'); 
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xlabel(xlabel,fontsize=14) 
        plt.ylabel(ylabel,fontsize=14)
        if xlim==None:
            ax.set_xlim([np.min(depth),np.max(depth)])
        else:
            ax.set_xlim(xlim)    
        if ylim==None:
            ax.set_ylim([np.min(chron_Q[:,1]),np.max(chron_Q[:,1])])
        else:    
           ax.set_ylim([ylim]) 
        plt.gca().invert_yaxis()
    
    return ax                        
