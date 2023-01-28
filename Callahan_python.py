#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:54:50 2023

@author: jamescallahan
"""

#Importing everything required
# data manipulation
import pandas as pd
import numpy as np

# data viz
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix

# apply some cool styling
plt.style.use("ggplot")
rcParams['figure.figsize'] = (12,  6)
#importing CSV


df = pd.read_csv('/Users/jamescallahan/Desktop/Ferrara ML Class/Wine assignment 1/assignment1_winequality.csv')
print(df)


#Exploratory data analysis

    

df.head()
df.tail()

df.describe()
df.info()




#Density distr.
df.density.value_counts().plot(kind="bar")
plt.title("Value counts of the density variable")
plt.xlabel("density type")
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.show()

#Creating correlation matrix
import seaborn as sn
correlations=df.corr()
dfo = pd.DataFrame(df)

corr_matrix = dfo.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()


#Pairplot time 
#sns.pairplot(dfo)


#Setting "quality" to 0 if <7 and to 1 if >=7

dfo['quality'] = np.where(dfo['quality'] >= 7, 1, 0)






#Getting rid of outliers
for x in ['fixed acidity']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan


for x in ['volatile acidity']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    
for x in ['citric acid']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    
for x in ['residual sugar']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan

for x in ['chlorides']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    
for x in ['free sulfur dioxide']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    

for x in ['total sulfur dioxide']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    
for x in ['density']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    
for x in ['pH']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    
for x in ['sulphates']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    
for x in ['alcohol']:
    q75,q25 = np.percentile(dfo.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(5*intr_qr)
    min = q25-(5*intr_qr)
 
    dfo.loc[dfo[x] < min,x] = np.nan
    dfo.loc[dfo[x] > max,x] = np.nan
    


dfo.head()
dfo.tail()

dfo.describe()
dfo.info()


#Changing NA into mean
antiNAclub=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality" ]
for column in antiNAclub:
#dataset [column] = dataset [column].replace (0, np NaN)
    mean = int (dfo [column].mean (skipna=True))
    dfo [column] = dfo [column]. replace (np.NaN, mean)


#Let's do some scaling

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


#I scaled the test and training set separately, but did scale the entire set separatelyl, for data vizualisation purposes. 

#df = pandas.read_csv("data.csv")

#dfonarrow=dfo[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality" ]]
#dfonarrow=dfo[["chlorides"]]

#X = dfonarrow

#scaledX = scale.fit_transform(X)

#print(scaledX)

#Scaled correlation matrix
#dfoscaled = pd.DataFrame(scaledX)

#corr_matrix = dfoscaled.corr()
#sn.heatmap(corr_matrix, annot=True)
#plt.show()


#Pairplot time (again!)
#sns.pairplot(dfoscaled)

#Notice how much narrower the value range is for things such as chloride once the outliers have been removed!




#We're  going to construct our KNN model first, followed by the Bernoulli Naive Bayes

#Importing some tools to prepare to run the model
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn. preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score



# splitting our dataset
X = dfo.iloc[:, 0:11]
y = dfo.iloc[:, 11]

#taking 20% of the data to set aside
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#Scaling the training and test data separately (Hence the use of dfo not dfoscaled)
sc_X = StandardScaler ()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X. transform (X_test)


print(X_train)


#Getting into the actual KNN algorithm




#Began using a K of 17 since rounding down from sprt(len(y_test)) is 17 (following some conventional wisdom on where to start guessing with K)
classifier = KNeighborsClassifier (n_neighbors=17, p=2, metric= "euclidean")
# Fit Model
classifier.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric= "euclidean",
                     metric_params = None, n_jobs=1, n_neighbors=17, p=2, 
                     weights='uniform')
 

# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred                    

#double-checking we got the correct nunber of neighbors guess
import math
math.sqrt(len(y_test))

  
# Evaluate Model
cm = confusion_matrix(y_pred, y_test)
print(cm)
print(f1_score (y_test, y_pred, average="micro"))
print(accuracy_score (y_pred, y_test))


dfo.head()


#Plotting the confusion matrix:

names = np.unique(y_pred)
# plot the matrix
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=False, xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')

# KNN Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn import svm
#clo = svm.SVC(kernel='linear', C=1, random_state=42)
accuracy1 = cross_val_score(classifier, X_train, y_train, cv=10)
accuracy1.mean()

precision1 = cross_val_score(classifier, X_train, y_train, cv=10, scoring = "precision")
precision1.mean()

accuracy = cross_val_score(classifier, X_train, y_train, cv=10)
accuracy.mean()

precision = cross_val_score(classifier, X_train, y_train, cv=10, scoring = "precision")
precision.mean()


#Metric of interest =1.4476085178716604
accuracy1.mean()+precision1.mean()

from sklearn import metrics
scores = cross_val_score(
classifier, X_train, y_train, cv=10, scoring='f1_macro')
scores







#Trying K=10
classifier1 = KNeighborsClassifier (n_neighbors=10, p=2, metric= "euclidean")
# Fit Model
classifier1.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric= "euclidean",
                     metric_params = None, n_jobs=1, n_neighbors=10, p=2, 
                     weights='uniform')
               


# Predict the test set results
y_pred1 = classifier1.predict(X_test)
y_pred1                    


 
# Evaluate Model
cm = confusion_matrix(y_test, y_pred1)
print (cm)
print(f1_score (y_test, y_pred1, average="micro"))
print(accuracy_score (y_test, y_pred1))

# KNN Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn import svm
#clo = svm.SVC(kernel='linear', C=1, random_state=42)
accuracy2 = cross_val_score(classifier1, X_train, y_train, cv=10)
accuracy2.mean()

precision2 = cross_val_score(classifier1, X_train, y_train, cv=10, scoring = "precision")
precision2.mean()

accuracy = cross_val_score(classifier1, X_train, y_train, cv=10)
accuracy.mean()

precision = cross_val_score(classifier1, X_train, y_train, cv=10, scoring = "precision")
precision.mean()


#Metric of interest =1.4921110503123298
accuracy2.mean()+precision2.mean()

from sklearn import metrics
scores = cross_val_score(
classifier1, X_train, y_train, cv=10, scoring='f1_macro')
scores








#Trying K=25
classifier2 = KNeighborsClassifier (n_neighbors=25, p=2, metric= "euclidean")
# Fit Model
classifier2.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric= "euclidean",
                     metric_params = None, n_jobs=1, n_neighbors=25, p=2, 
                     weights='uniform')
 

# Predict the test set results
y_pred2 = classifier2.predict(X_test)
y_pred2                    



  
# Evaluate Model
cm = confusion_matrix(y_test, y_pred2)
print (cm)
print(f1_score (y_test, y_pred2, average="micro"))
print(accuracy_score (y_test, y_pred2))




# KNN Cross-validation
from sklearn.model_selection import cross_val_score
from sklearn import svm
#clo = svm.SVC(kernel='linear', C=1, random_state=42)
accuracy3 = cross_val_score(classifier2, X_train, y_train, cv=10)
accuracy3.mean()

precision3 = cross_val_score(classifier2, X_train, y_train, cv=10, scoring = "precision")
precision3.mean()

accuracy = cross_val_score(classifier2, X_train, y_train, cv=10)
accuracy.mean()

precision = cross_val_score(classifier2, X_train, y_train, cv=10, scoring = "precision")
precision.mean()


#Metric of interest =1.4808053822817602
accuracy3.mean()+precision3.mean()

print("%0.2f accuracy with a standard deviation of %0.2f" % (accuracy.mean(), accuracy.std()))


from sklearn import metrics
scores = cross_val_score(
classifier2, X_train, y_train, cv=10, scoring='f1_macro')
scores


#The combined precision and model accuracy score shows that this model works pretty well. Additionally, we have been sure to minimize Type 1 errors as they are worse for business than type 2 errors. 
#Tried a bunch of different K-values (cross-validation) K=10 proved to be the best, even averaged across multiple runs. 









#Now we shall formulate a Naive Bayes model

#Loading a neccesary package
from sklearn.naive_bayes import BernoulliNB

#Defining and fitting the model (using a common training and test-set with the K nearest neighbor model)
model=BernoulliNB()
model.fit(X_train,y_train)
pred=model.predict(X_test)

#Evaluating our model
cm = confusion_matrix(pred, y_test)
print (cm)
print(f1_score (pred, y_test, average="micro"))
print(accuracy_score (y_test, y_pred))



# Naive-Bayes cross-validation

# Seeing where Naive-Bayes puts us
from sklearn.model_selection import cross_val_score
from sklearn import svm
#clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=10)
scores


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


from sklearn import metrics
#scores = cross_val_score(
#clf, X, y, cv=10, scoring='f1_macro')
scores

accuracy4 = cross_val_score(model, X_train, y_train, cv=10)
accuracy4.mean()

precision4 = cross_val_score(model, X_train, y_train, cv=10, scoring = "precision")
precision4.mean()

accuracy = cross_val_score(model, X_train, y_train, cv=10)
accuracy.mean()

precision = cross_val_score(model, X_train, y_train, cv=10, scoring = "precision")
precision.mean()


#Metric of interest =1.3476482531442149
accuracy4.mean()+precision4.mean()

#Looking at the "Metric of interest" (the best being 1.492 for the KNN Vs. 1.348 for the Naive-Bayes) we can see the Bernoulli Naive-Bayes is less than optimal, especially as it has significantly more Type 2 errors, but especially importantly, it has twice as many type 1 errors. A customer who pays a lot of money for a bottle of wine and gets lesser quality will be madder than someone who bought what they thought was cheap wine and it turned out to be high-quality wine. Thus we would go with the KNN model with K=10. 