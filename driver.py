import classifier 
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
reload(classifier)

def main():
	baseDir = '/Users/nkridler/Desktop/whale/'

	params = {'max_depth':6, 'subsample':0.5, 'verbose':2, 'random_state':1337,
		'min_samples_split':25, 'min_samples_leaf':25, 'max_features':15,
		'n_estimators': 12000, 'learning_rate': 0.002}
		#'n_estimators': 500, 'learning_rate': 0.05}
	clf = GradientBoostingClassifier(**params)	
	cols = np.array(range(77,447))
	cols = np.array(range(270)) #151
	oFile = baseDir+'moby2/serial32.csv'
	test = classifier.Classify(baseDir+'workspace/trainMetricsLowerFreq3.csv')#,useCols=cols)
	#test.validate(clf,nFolds=6,featureImportance=True,outFile='gbmLowerFreq3.csv')
	test.testAndOutput(clf=clf,testFile=baseDir+'workspace/testMetricsLowFreq3.csv',outfile='607.csv')
if __name__=="__main__":
	main()