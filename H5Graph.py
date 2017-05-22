#!/usr/bin/env python
#Filename: H5Graph


    #################################
    #### Import needed libraries ####
    #################################

import logging
import glob
import h5py
import matplotlib
import matplotlib.pyplot as plt

    ##################################
    #### Import Data from H5 file ####
    ##################################

def importH5Data( H5fileString ):
    attrs = []
    dataList = []
    
    for thing in H5fileString:
        
        H5file = thing.split( '.h5', 1 )
    
        #print(H5file)
    
        H5file[0] += '.h5'
    
        #import h5 file as read only
        file = h5py.File( H5file[0] , 'r')
        
        #grab data from .dat file inside h5 stuct and save as array
        dataArray = file[ H5file[1] ] 
	
	attrs.append(dataArray.attrs)
        
        dataList.append(dataArray)
    
    #retun that array
    return [dataList, attrs]
    
    
    ##########################################
    #### Set defualt formatting for plots ####
    ##########################################
    
def defaultFormat(plt, metaData, title):
    
    #construct x-axis labels and check if they are all the same
    xlab = [thing[0] for thing in metaData]
    if all( x==xlab[0] for x in xlab):
        xlab = xlab[0]
        
    #construct y-axis labels and check if they are all the same
    ylab = [thing[1] for thing in metaData]
    if all( x==ylab[0] for x in ylab):
        ylab = ylab[0]
        
    arrayTitle = []
    for thing in title:
        arrayTitle.append(thing)
        
    titleFinal = ",\n ".join(arrayTitle)
    
    plt.xlabel(xlab,fontsize = 18)
    plt.ylabel(ylab,fontsize = 18)
    plt.yticks(fontsize=16) 
    plt.xticks(fontsize=16)
    plt.title(titleFinal , fontsize = 15)
   
    
    return
    
    ######################################################
    #### Gather MetaData information from attrs array ####
    ######################################################
    
def getMetaData(attrs, col1, col2, col3=False):
    MetaData = []
    
    MetaData.append(attrs['Legend'][col1])
    MetaData.append(attrs['Legend'][col2])
    if col3 == 3:
        MetaData.append(attrs['Legend'][col3])
        MetaData[2] = '$' + MetaData[2] + '$'
    #add $ on both sides to display LaTex format
    MetaData[0] = '$' + MetaData[0] + '$'
    MetaData[1] = '$' + MetaData[1] + '$'
    
    
    return MetaData
    
    ###################################################################
    #### Take the h5 output data and format it like this:          ####
    #### item in newData where: item[0] = data, item[1] = metaData ####
    ###################################################################
    
def formatData(Data):
    
    newData = []
    length = len(Data[0])
    for i in range (0, length):
        duo = []
        duo.append(Data[0][i])
        duo.append(Data[1][i])
        newData.append(duo)
        
        
    return newData
    
    
    #########################################################################################
    #### obtain columns given by command line or if no columns are given then obtain     ####
    #### all columns in each .dat file. Then append to newData for each item in new Data ####
    #########################################################################################
    
def getColumns(newData, args):
    logger = logging.getLogger('verbose_data')
    if not args.c:
        logger.warn('*** No columns selected. Will graph all columns to column one for each file.')
        #print('*** No columns selected. Will graph all columns to column one for each file.')
        for thing in newData:
            columns = []
            for i in range (1, len(thing[1]['Legend'])):
                columns.append([1 , i + 1]) 
        
            thing.append(columns)
    
    else:
        for item in newData:
            columns = []
            for thing in args.c:
                columns.append(thing.split(':'))
                
            item.append(columns)
    return newData
      
      
    ##################################################################################################
    #### Input: newData: array with each item containing an array with data, metaData and columns ####
    ####        h5idir: title of the subplot                                                      ####
    #### Output: create a subplot and given newData and format if given a title                   ####
    ##################################################################################################
def plotData(newData, h5dir):

    #create plot figure/subplots
    fig = plt.figure(figsize=(12.0, 9.0))
    ax = fig.add_subplot(111)
    
    totalMeta = []
    for item in newData:
        for num in item[2]:
            col1 = int(num[0]) -1
            col2 = int(num[1]) -1
        
            #get Meta Data
            metaData = getMetaData(item[1], col1, col2) 
            
            #plot data
            ax.plot(item[0][:,col1], item[0][:,col2], label = metaData[1] + ' vs ' + metaData[0])
            totalMeta.append(metaData)
            
    if not h5dir:
        #logger.warn('*** No default formating will be done')
        print('*** No default formating will be done')
    else:
        defaultFormat(plt, totalMeta, h5dir)
    
    return 
    

    #####################################################################
    #### Show legend in specified location and with specified names. ####
    #### No names will result in defualts being chosen.              ####
    #####################################################################

def legend(args):
    logger = logging.getLogger('verbose_data')
    #add legend
    if  not args.legend:
        logger.warn('*** Legend will not be displayed')
        #print('*** Legend will not be displayed')
    else:
        locleg = int(''.join(args.legend[0]))
        plt.legend(loc=locleg)
        
        if len(args.legend) <= 1:
            logger.warn('*** Default legend names will be used')
            #print('*** Default legend names will be used')
        else:
            del args.legend[0]
            plt.legend(args.legend, loc=locleg)
    return

    ########################################################
    #### Show x and y axis labels with specified names. ####
    #### No names will result in defualts being chosen. ####
    ########################################################

def labXY(args):
    logger = logging.getLogger('verbose_data')
    if not args.X:
        logger.warn('*** No X axis labels were given. Will print default.')
        #print('*** No X axis labels were given. Will print default.')
    else:
        plt.xlabel(''.join(args.X), fontsize = 18)
        
    if not args.Y:
        logger.warn('*** No Y axis labels were given. Will print default.')
        #print('*** No Y axis labels were given. Will print default.')
    else:
        plt.ylabel(''.join(args.Y), fontsize = 18)
        
    return
    
def wildFile(args):
    logger = logging.getLogger('verbose_data')
    globDatFiles = False
    globHorizons = False  
    files = []
    totalDat = []
    totalDir = []
    #Grab data from each .h5 file/folder
    for thing in args:
        H5file = thing.split( '.h5', 1 )    
        H5file[0] += '.h5'
        
        #check for wild cards in .dir files        
        dirstring = H5file[1].split( '/', 2)
        if '?' in dirstring[1] or '*' in dirstring[1]:
            globHorizons = True
            
        if '?' in dirstring[2] or '*' in dirstring[2]:
            dirstring[2] = dirstring[2].split('*',1)
            globDatFiles = True
            
            
        globfile = glob.glob(H5file[0])
        
        #import h5 file as read only and construct the full file names if wild cards were used
        for thing in globfile:
            file = h5py.File( thing , 'r' )
            if globDatFiles:
                horizon = file.items()
                datFiles = horizon[0][1].items()
                for data in datFiles:
                    
                    if dirstring[2][0] in data[0] and dirstring[2][1] in data[0]:
                        foundDatFile = ''.join(data[0])
                        totalDat.append(str(foundDatFile))
            else:
                totalDat.append(dirstring[2])
            
            if globHorizons:   
                horizons = file.items()
                for dotdir in horizons:
                    dirstring[1] = dotdir[0]
                    totalDir.append(str(dirstring[1]))
  
            
            else:
                totalDir.append(dirstring[1])
            for direc in totalDir:
                for dat in totalDat:
                    files.append(str(thing) +'/'+ str(direc) + '/'+ str(dat))
    
    logger.warn('*****' + str(files) + ' will be graphed.')
    #print '*****' + str(files) + ' will be graphed.'
    #print files
    return files
    
    
    ##################################################################################################
    #### Input: newData: array with each item containing an array with data, metaData and columns ####
    ####        h5idir: title of the subplot                                                      ####
    #### Output: create a subplot in 3d and given newData and format if given a title             ####
    ##################################################################################################
def threeDPlot(newData, h5dir):
    logger = logging.getLogger('verbose_data')
    fig = plt.figure(figsize=(12.0, 9.0))
    ax = fig.add_subplot(111, projection='3d')
    totalMeta = []
    for item in newData:
        
        #get Meta Data
        metaData = getMetaData(item[1], 1, 2, 3) 
            
        #plot data
        ax.plot(item[0][:,1], item[0][:,2], item[0][:,3])
        totalMeta.append(metaData)
    
    if not h5dir:
        logger.warn('*** No default formating will be done')
        #print('*** No default formating will be done')
    else:
        #construct x-axis labels and check if they are all the same
        xlab = [thing[0] for thing in totalMeta]
        if all( x==xlab[0] for x in xlab):
            xlab = xlab[0]
        
        #construct y-axis labels and check if they are all the same
        ylab = [thing[1] for thing in totalMeta]
        if all( x==ylab[0] for x in ylab):
            ylab = ylab[0]
           
        #construct z-axis labels and check if they are all the same
        zlab = [thing[2] for thing in totalMeta]
        if all ( x==zlab[0] for x in zlab):
            zlab = zlab[0] 
              
        arrayTitle = []
        for thing in h5dir:
            arrayTitle.append(thing)
            
        titleFinal = ",\n ".join(arrayTitle)
        
        ax.set_xlabel(xlab,fontsize = 18)
        ax.set_ylabel(ylab,fontsize = 18)
        ax.set_zlabel(zlab,fontsize = 18)
        plt.yticks(fontsize=16) 
        plt.xticks(fontsize=16)
        #plt.zticks(fontsize=16)
        plt.title(titleFinal , fontsize = 15)
        
    return
