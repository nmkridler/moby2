import classifier 
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
reload(classifier)

import random
def main():
	baseDir = '/Users/nkridler/Desktop/whale/'

	params = {'max_depth':6, 'subsample':0.5, 'verbose':2, 'random_state':1337,
		'min_samples_split':25, 'min_samples_leaf':25, 'max_features':15,
		'n_estimators': 12000, 'learning_rate': 0.002}
	clf = GradientBoostingClassifier(**params)	
	test = classifier.Classify(baseDir+'workspace/trainMetricsBase.csv',useCols=cols)
	test.testAndOutput(clf=clf,testFile=baseDir+'workspace/testMetricsBase.csv',outfile='base.csv')

	test = classifier.Classify(baseDir+'workspace/trainMetricsBaseMoreTemplates.csv',useCols=cols)
	test.testAndOutput(clf=clf,testFile=baseDir+'workspace/testMetricsMoreTemplates.csv',outfile='more.csv')

	more_ = np.loadtxt('more.csv')
	base_ = np.loadtxt('base.csv')
	np.savetxt('final.csv',0.5*more_ + 0.5*base_)

if __name__=="__main__":
	main()