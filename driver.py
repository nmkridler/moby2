import classifier 
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
reload(classifier)

def main():
	baseDir = '/Users/nkridler/Desktop/whale/'

	params = {'max_depth':5, 'subsample':0.5, 'verbose':0, 'random_state':1337,
		'min_samples_split':25, 'min_samples_leaf':25, 'max_features':10,
		'n_estimators': 12000, 'learning_rate': 0.002}
		#'n_estimators': 500, 'learning_rate': 0.05}
	clf = GradientBoostingClassifier(**params)	
	cols = np.array(range(77,447))
	cols = np.array(range(270)) #151
	test = classifier.Classify(baseDir+'workspace/trainMetricsLowerFreq2.csv')#,useCols=cols)
	#test.validate(clf,nFolds=6,featureImportance=False,outFile='gbmLowerFreq2.csv')
	test.testAndOutput(clf=clf,testFile=baseDir+'workspace/testMetricsLowFreq2.csv',outfile='523.csv')
if __name__=="__main__":
	main()