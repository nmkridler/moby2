""" genTrainMetrics.py

	This file generates the training metrics
"""


import numpy as np
import pylab as pl

import metrics
import plotting
import fileio
import templateManager
import cv2
reload(fileio)
reload(metrics)


def makeMetrics(tmplFile,train=None,old=None,params={},trainOutFile='out.csv'):
	""" Make the metrics """
	maxTime = 60 # Number of time slice metrics
	##################### BUILD A TemplateManager OBJECT ####################
	tmpl = templateManager.TemplateManager(fileName=tmplFile, 
		trainObj=old, params=params)

	########################### CREATE THE HEADER ###########################
	outHdr = metrics.buildHeader(tmpl)

	###################### LOOP THROUGH THE FILES ###########################
	hL = []
	for i in range(train.numH1):
		P, freqs, bins = train.H1Sample(i,params=params)
		out = metrics.computeMetrics(P, tmpl, bins, maxTime)
		hL.append([1, i] + out)
	for i in range(train.numH0):
		P, freqs, bins = train.H0Sample(i,params=params)
		out = metrics.computeMetrics(P, tmpl, bins, maxTime)
		hL.append([0, i] + out)
	hL = np.array(hL)
	file = open(trainOutFile,'w')
	file.write("Truth,Index,"+outHdr+"\n")
	np.savetxt(file,hL,delimiter=',')
	file.close()

def main():
	baseDir = '/Users/nkridler/Desktop/whale/' # Base directory

	############################## PARAMETERS ###############################
	dataDir = baseDir+'data2/'				   # Data directory
	oldDir = baseDir+'data/'				   # Data directory
	params = {'NFFT':256, 'Fs':2000, 'noverlap':192} # Spectogram parameters

	######################## BUILD A TrainData OBJECT #######################
	train = fileio.TrainData(dataDir+'train2.csv',dataDir+'train2/')
	old = fileio.TrainData(oldDir+'train.csv',oldDir+'train/')

	###################### SET OUTPUT FILE NAME HERE ########################
	trainOutFile = baseDir+'workspace/trainMetricsBase.csv'
	tmplFile = baseDir+'moby2/templateBase.csv'
	makeMetrics(tmplFile,train=train,old=old,params=params,trainOutFile=trainOutFile)

	###################### SET OUTPUT FILE NAME HERE ########################
	tmplFile = baseDir+'moby2/manyMoreTemplates.csv'
	trainOutFile = baseDir+'moby2/trainMetricsMoreTemplates.csv'
	makeMetrics(tmplFile,train=train,old=old,params=params,trainOutFile=trainOutFile)


if __name__ == "__main__":
	main()
