""" genTestMetrics.py

	This file generates the test metrics
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

def makeMetrics(tmplFile,test=None,old=None,params={},testOutFile='out.csv'):
	""" Make the metrics """
	maxTime = 60 # Number of time slice metrics
	##################### BUILD A TemplateManager OBJECT ####################
	tmpl = templateManager.TemplateManager(fileName=tmplFile, 
		trainObj=old, params=params)

	########################### CREATE THE HEADER ###########################
	outHdr = metrics.buildHeader(tmpl)

	hL = []
	####################### LOOP THROUGH THE FILE ###########################
	for i in range(test.nTest):
		P, freqs, bins = test.TestSample(i,params=params)
		out = metrics.computeMetrics(P, tmpl, bins, maxTime)
		hL.append(out)			
	hL = np.array(hL)

	########################## WRITE TO FILE ################################
	file = open(testOutFile,'w')
	file.write(outHdr+"\n")
	np.savetxt(file,hL,delimiter=',')
	file.close()
		

def main():
	baseDir = '/Users/nkridler/Desktop/whale/' # Base directory

	############################## PARAMETERS ###############################
	dataDir = baseDir+'data2/'				   # Data directory
	oldDir = baseDir+'data/'				   # Data directory
	params = {'NFFT':256, 'Fs':2000, 'noverlap':192} # Spectogram parameters

	######################## BUILD A TestData OBJECT #######################
	test = fileio.TestData2(dataDir+'test2.csv',dataDir+'test2/')
	old = fileio.TrainData(oldDir+'train.csv',oldDir+'train/')

	###################### SET OUTPUT FILE NAME HERE ########################
	testOutFile = baseDir+'workspace/testMetricsBase.csv'
	tmplFile = baseDir+'moby2/templateBase.csv'
	makeMetrics(tmplFile,test=test,old=old,params=params,testOutFile=testOutFile)

	###################### SET OUTPUT FILE NAME HERE ########################
	tmplFile = baseDir+'moby2/manyMoreTemplates.csv'
	testOutFile = baseDir+'moby2/testMetricsMoreTemplates.csv'
	makeMetrics(tmplFile,test=test,old=old,params=params,testOutFile=testOutFile)


if __name__ == "__main__":
	main()
