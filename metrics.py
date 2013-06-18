import random
import numpy as np
import cv2
import cv2.cv as cv
import pylab as pl
from scipy.signal import convolve2d
from scipy.stats import skew, kurtosis

def buildHeader(tmpl,maxT=60):
	""" Build a header

		Build the header for the metrics

		Args:
			tmpl: templateManager object

		Returns:
			header string as csv
	"""
	hdr_ = []
	prefix_ = ['max','xLoc','yLoc']
	for p_ in prefix_:
		for i in range(tmpl.size):
			hdr_.append(p_+'_%07d'%tmpl.info[i]['file'])
	for p_ in prefix_:
		for i in range(tmpl.size):
			hdr_.append(p_+'H_%07d'%tmpl.info[i]['file'])

	# Add time metrics
	for i in range(maxT):
		hdr_ += ['centTime_%04d'%i]
	for i in range(maxT):
		hdr_ += ['bwTime_%04d'%i]
	for i in range(maxT):
		hdr_ += ['skewTime_%04d'%i]
	for i in range(maxT):
		hdr_ += ['tvTime_%04d'%i]

	return ','.join(hdr_)


def computeMetrics(P, tmpl, bins, maxT):
	""" Compute a bunch of metrics

		Perform template matching and time stats

		Args:
			P: 2-d numpy array
			tmpl: templateManager object
			bins: time bins
			maxT: maximum frequency slice for time stats

		Returns:
			List of metrics
	"""

	Q = slidingWindowV(P,inner=3,maxM=40,maxT=bins.size)
	W = slidingWindowH(P,inner=3,outer=32,maxM=60,maxT=bins.size)

	out = templateMetrics(Q, tmpl)	
	out += templateMetrics(W, tmpl)	
	out += timeMetrics(P,bins,maxM=maxT)
	return out

def matchTemplate(P, template):
	""" Max correlation and location

		Calls opencv's matchTemplate and returns the
		max correlation and location

		Args:
			P: 2-d numpy array to search
			template: 2-d numpy array to match

		Returns:
			maxVal: max correlation
			maxLoc: location of the max
	"""
	m, n = template.shape
	mf = cv2.matchTemplate(P.astype('Float32'), template, cv2.TM_CCOEFF_NORMED)
	minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mf)
	return maxVal, maxLoc[0], maxLoc[1]


def slidingWindowV(P,inner=3,outer=64,maxM=50,minM=7,maxT=59,norm=True):
	""" Enhance the constrast vertically (along frequency dimension)

		Cut off extreme values and demean the image
		Utilize numpy convolve to get the mean at a given pixel
		Remove local mean with inner exclusion region

		Args:
			P: 2-d numpy array image
			inner: inner exclusion region 
			outer: length of the window
			maxM: size of the output image in the y-dimension
			norm: boolean to cut off extreme values

		Returns:
			Q: 2-d numpy contrast enhanced vertically
	"""
	Q = P.copy()
	m, n = Q.shape
		
	if norm:
		mval, sval = np.mean(Q[minM:maxM,:maxT]), np.std(Q[minM:maxM,:maxT])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
		Q[:minM,:] = mval
	wInner = np.ones(inner)
	wOuter = np.ones(outer)
	for i in range(maxT):
		Q[:,i] = Q[:,i] - (np.convolve(Q[:,i],wOuter,'same') - np.convolve(Q[:,i],wInner,'same'))/(outer - inner)
	Q[Q < 0] = 0.

	return Q[:maxM,:]

def slidingWindowH(P,inner=3,outer=32,maxM=50,minM=7,maxT=59,norm=True):
	""" Enhance the constrast horizontally (along temporal dimension)

		Cut off extreme values and demean the image
		Utilize numpy convolve to get the mean at a given pixel
		Remove local mean with inner exclusion region

		Args:
			P: 2-d numpy array image
			inner: inner exclusion region 
			outer: length of the window
			maxM: size of the output image in the y-dimension
			norm: boolean to cut off extreme values

		Returns:
			Q: 2-d numpy contrast enhanced vertically
	"""
	Q = P.copy()
	m, n = Q.shape
	if outer > maxT:
		outer = maxT

	if norm:
		mval, sval = np.mean(Q[minM:maxM,:maxT]), np.std(Q[minM:maxM,:maxT])
		fact_ = 1.5
		Q[Q > mval + fact_*sval] = mval + fact_*sval
		Q[Q < mval - fact_*sval] = mval - fact_*sval
		Q[:minM,:] = mval

	wInner = np.ones(inner)
	wOuter = np.ones(outer)
	if inner > maxT:
		return Q[:maxM,:]
		
	for i in range(maxM):
		Q[i,:maxT] = Q[i,:maxT] - (np.convolve(Q[i,:maxT],wOuter,'same') - np.convolve(Q[i,:maxT],wInner,'same'))/(outer - inner)
	Q[Q < 0] = 0.
	return Q[:maxM,:]

def timeMetrics(P, b,maxM=50):
	""" Calculate statistics for a range of frequency slices

		Calculate centroid, width, skew, and total variation
			let x = P[i,:], and t = time bins
			centroid = sum(x*t)/sum(x)
			width = sqrt(sum(x*(t-centroid)^2)/sum(x))
			skew = scipy.stats.skew
			total variation = sum(abs(x_i+1 - x_i))

		Args:
			P: 2-d numpy array image
			b: time bins 

		Returns:
			A list containing the statistics

	"""
	m, n = P.shape
	cf_ = [np.sum(P[i,:b.size]*b)/np.sum(P[i,:b.size]) for i in range(maxM)]
	bw_ = [np.sum(P[i,:b.size]*(b - cf_[i])*(b - cf_[i]))/np.sum(P[i,:b.size]) for i in range(maxM)]
	sk_ = [skew(P[i,:b.size]) for i in range(maxM)]
	tv_ = [np.sum(np.abs(P[i,1:b.size] - P[i,:b.size-1])) for i in range(maxM)]
	return cf_ + bw_ + sk_ + tv_

def templateMetrics(P, tmpl):
	""" Template matching

		Perform template matching for a list of templates

		Args:
			P: 2-d numpy array image
			tmpl: templateManager object

		Returns:
			List of correlations, x and y pixel locations of the max 
	"""
	maxs, xs, ys = [], [], []
	for k in range(tmpl.size):
		mf, y, x  = matchTemplate(P,tmpl.templates[k])
		maxs.append(mf)
		xs.append(x)
		ys.append(y)
	return maxs + xs + ys



