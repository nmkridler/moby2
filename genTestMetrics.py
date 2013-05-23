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
def main():
	###################### WORKING DIRECTORY ########################
	baseDir = '/Users/nkridler/Desktop/whale/'

	###################### SET OUTPUT FILE NAME HERE ########################
	testOutFile = baseDir+'workspace/testMetricsLowFreq2.csv'

	############################## PARAMETERS ###############################
	dataDir = baseDir+'data2/'			   # Data directory
	oldDir = baseDir +'data/'
	params = {'NFFT':256, 'Fs':2000, 'noverlap':192} # Spectogram parameters
	maxTime = 60 # Number of time slice metrics

	######################## BUILD A TestData OBJECT #######################
	#train = fileio.TrainData(dataDir+'train.csv',dataDir+'train/')
	test = fileio.TestData2(dataDir+'test2.csv',dataDir+'test2/')

	##################### BUILD A TemplateManager OBJECT ####################
	tmplFile = baseDir+'moby2/templateReduced.csv'
	old = fileio.TrainData(oldDir+'train.csv',oldDir+'train/')
	tmpl = templateManager.TemplateManager(fileName=tmplFile, 
		trainObj=old, params=params)

	################## VERTICAL BARS FOR HIFREQ METRICS #####################
	bar_ = np.zeros((12,9),dtype='Float32')
	bar1_ = np.zeros((12,12),dtype='Float32')
	bar2_ = np.zeros((12,6),dtype='Float32')
	bar_[:,3:6] = 1.
	bar1_[:,4:8] = 1.
	bar2_[:,2:4] = 1.

	########################### CREATE THE HEADER ###########################
	outHdr = metrics.buildHeader(tmpl)

	hL = []
	####################### LOOP THROUGH THE FILE ###########################
	for i in range(test.nTest):
		P, freqs, bins = test.TestSample(i,params=params)
		out = metrics.computeMetrics(P, tmpl, bins, maxTime)
		#out += metrics.highFreqTemplate(P, bar_)
		#out += metrics.highFreqTemplate(P, bar1_)
		#out += metrics.highFreqTemplate(P, bar2_)
		hL.append(out)			
	hL = np.array(hL)

	########################## WRITE TO FILE ################################
	file = open(testOutFile,'w')
	file.write(outHdr+"\n")
	np.savetxt(file,hL,delimiter=',')
	file.close()
		

if __name__ == "__main__":
	main()
