
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import ensemble
from sklearn.metrics import roc_auc_score as auc
import time
from sklearn.ensemble import ExtraTreesClassifier


# # Load santandertrain.csv file

trainDataFrame = pd.read_csv("D:/Users/annieuwe/Documents/Documents/Nieuw/Data Science/Data Sets/Santander/santandertrain.csv")
#trainDataFrame.head(5)


# # Summary of tabel

print len(trainDataFrame)
#trainDataFrame.columns


# # Basic statistics

trainDataFrame.describe()

trainDataFrame.groupby('TARGET').mean()


# # Show histogram of satisfied customer vs unsatisfied customer (TARGET score 1 vs 0)

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

trainDataFrame.TARGET.hist()
plt.title('satisfied customer vs unsatisfied customer')
plt.xlabel('satisfaction (1 = unsatisfied)')
plt.ylabel('Frequency')


# # Build an estimator trying to predict the target for each feature individually

# remove constant columns
colsToRemove = []
for col in trainDataFrame.columns:
    if trainDataFrame[col].std() == 0:
        colsToRemove.append(col)

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = trainDataFrame.columns
for i in range(len(columns)-1):
    v = trainDataFrame[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,trainDataFrame[columns[j]].values):
            colsToRemove.append(columns[j])

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)


# # Create two seperate dataframes for the label (0/1) and the features

trainLabels2 = trainDataFrame['TARGET']
trainLabels2[:5]

trainFeatures = trainDataFrame.drop(['ID','TARGET'], axis=1)
trainFeatures[:5]

targetcolumn = ['TARGET']
trainLabels = pd.DataFrame(trainDataFrame, columns = targetcolumn)
trainLabels[:5]

trainFeatures2 = trainDataFrame(['var3','var15'], axis=1)
trainFeatures2[:5]

X = trainDataFrame.iloc[:,:-1]
y = trainDataFrame.TARGET


# # Show percentage of satisfied customers and number of satisfied vs unsatisfied customer (0/1)

print trainLabels.mean()
print trainLabels.TARGET.value_counts()


# # Feature selection

# # Variance Threshold - Exclude all low-variance features X

# Features with a training-set variance lower than this threshold will be removed. The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples.

featurelist = X.columns.tolist()
featurearray = np.asarray(featurelist)
featurearray[:5]

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)[:5]


# # Tree based feature selection

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
print trainFeatures.shape

ExtraTreesModel = ExtraTreesClassifier()
ExtraTreesModel.fit(trainFeatures, trainLabels)
#print ExtraTreesModel.feature_importances_  

print zip(map(lambda x: round(x, 4), ExtraTreesModel.feature_importances_), featurearray)[:5]
print "Features sorted by their score:"
ExtraTreesModel_sorted_array = sorted(zip(map(lambda x: round(x, 4), ExtraTreesModel.feature_importances_), featurearray),
             reverse=True)[:35]
print ExtraTreesModel_sorted_array[:5]

model = SelectFromModel(ExtraTreesModel, prefit=True)
trainFeatures_tree = model.transform(trainFeatures)
print trainFeatures_tree.shape

# Convert the array from the output to a table
from astropy.table import Table, Column

ExtraTreesModel_table = Table(rows=ExtraTreesModel_sorted_array, names=('feature_importance_score', 'feature',))
print "Table of Feature selection with Extra Trees Cleassifier:"
ExtraTreesModel_table


# Note: With the Extra Trees Classifier 35 variables are rated as important out of 306

# # Gradient Boosting feature selection

print trainFeatures.shape

GradientBoostingModel = ensemble.GradientBoostingClassifier()
GradientBoostingModel.fit(trainFeatures, trainLabels)
#print GradientBoostingModel.feature_importances_  
print zip(map(lambda x: round(x, 4), ExtraTreesModel.feature_importances_), featurearray)[:5]
print "Features sorted by their score:"
GradientBoostingModel_sorted_array = sorted(zip(map(lambda x: round(x, 4), ExtraTreesModel.feature_importances_), featurearray),
             reverse=True)[:56]
print GradientBoostingModel_sorted_array[:5]

model = SelectFromModel(GradientBoostingModel, prefit=True)
trainFeatures_gradientboosting = model.transform(trainFeatures)
print trainFeatures_gradientboosting.shape

# Convert the array from the output to a table
from astropy.table import Table, Column

GradientBoostingModel_table = Table(rows=GradientBoostingModel_sorted_array, names=('feature_importance_score', 'feature',))
print "Table of Feature selection with Gradient Boosting Cleassifier"
GradientBoostingModel_table[:5]


# Note: With the Gradient Boosting Classifier 56 variables are rated as important out of 306

# # Feature importance based on AUC with Gradient Boosting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score as auc
import time

#%% look at single feature performance

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels2, test_size=0.5, random_state=1)
        
startTime = time.time()
singleFeatureAUC_list = []
singleFeatureAUC_dict = {}
for feature in X_train.columns:
    trainInputFeature = X_train[feature].values.reshape(-1,1)
    validInputFeature = X_valid[feature].values.reshape(-1,1)
    GradientBoostingModel.fit(trainInputFeature, y_train)
    
    trainAUC = auc(y_train, GradientBoostingModel.predict_proba(trainInputFeature)[:,1])
    validAUC = auc(y_valid, GradientBoostingModel.predict_proba(validInputFeature)[:,1])
        
    singleFeatureAUC_list.append(validAUC)
    singleFeatureAUC_dict[feature] = validAUC
        
#validAUC = np.array(singleFeatureAUC_list)
timeToTrain = (time.time()-startTime)/60
print("(min,mean,max) AUC = (%.3f,%.3f,%.3f). took %.2f minutes" %(validAUC.min(),validAUC.mean(),validAUC.max(), timeToTrain))

#print singleFeatureAUC_list[:5]
singleFeatureAUC_dict


# # Single feature importance based on AUC with Gradient Boosting¶

# create a table with features sorted according to AUC
singleFeatureTable = pd.DataFrame(index=range(len(singleFeatureAUC_dict.keys())), columns=['feature','AUC'])
for k,key in enumerate(singleFeatureAUC_dict):
    singleFeatureTable.ix[k,'feature'] = key
    singleFeatureTable.ix[k,'AUC'] = singleFeatureAUC_dict[key]
singleFeatureTable = singleFeatureTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)

singleFeatureTable.ix[:20,:]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.feature_selection import SelectFromModel

train = pd.read_csv("D:/Users/annieuwe/Documents/Documents/Nieuw/Data Science/Data Sets/Santander/santandertrain.csv")
test = pd.read_csv("D:/Users/annieuwe/Documents/Documents/Nieuw/Data Science/Data Sets/Santander/santandertest.csv")

# clean and split data

# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

#split
test_id = test.ID
test = test.drop(["ID"],axis=1)

X2 = train.drop(["TARGET","ID"],axis=1)
y2 = train.TARGET.values

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.20)
print(X_train2.shape, X_test2.shape, test.shape)

#Feature selection
sclf = ExtraTreesClassifier(n_estimators=47,max_depth=47)
selector = sclf.fit(X_train2, y_train2)
fs = SelectFromModel(selector, prefit=True)

X_train2 = fs.transform(X_train2)
X_test2 = fs.transform(X_test2)
test = fs.transform(test)

print(X_train2.shape, X_test2.shape, test.shape)

#loop
names = ["etsc","dtc","etc","abc","gbc"]
clfs = [
ExtraTreesClassifier(n_estimators=100,max_depth=5),
DecisionTreeClassifier(max_depth=5),
ExtraTreeClassifier(max_depth=5),
AdaBoostClassifier(n_estimators=100),
GradientBoostingClassifier(n_estimators=100,max_depth=5)
]

plt.figure()
for name,clf in zip(names,clfs):
    clf.fit(X_train2,y_train2)
    y_proba = clf.predict_proba(X_test2)[:,1]
    print("Roc AUC:"+name, roc_auc_score(y_test2, clf.predict_proba(X_test2)[:,1],average='macro'))
    fpr, tpr, thresholds = roc_curve(y_test2, y_proba)
    plt.plot(fpr, tpr, label=name)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('1.png')
plt.show()          

#probs = xgb.predict_proba(test)
#submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})
#submission.to_csv("submission.csv", index=False)

from sklearn.linear_model import LogisticRegression

# flatten y into a 1-D array so that scikit-learn will properly understand it as the response variable
y = np.ravel(y)

# instantiate a logistic regression model, and fit with X and y
LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(trainFeatures, trainLabels)

# check the accuracy on the training set
LogisticRegressionModel.score(trainFeatures, trainLabels)


# # Hier blijven haken: join ExtraTreesModel met trainDataFrame
# Transpose trainDataFrame en maak een variable van de index column.
# Join op deze column.
# 
# https://www.kaggle.com/selfishgene/santander-customer-satisfaction/basic-feature-exploration/comments

ExtraTreesModel_sorted_dataframe = pd.DataFrame(ExtraTreesModel_sorted_array)
print ExtraTreesModel_sorted_dataframe

trainDataFrameT = trainDataFrame.transpose()
trainDataFrameT[:5]

trainDataFrameT.index.names = ['feature']
trainDataFrameT2 = trainDataFrameT.reindex(trainDataFrameT.index.rename('1'))

trainDataFrameT2[:5]

