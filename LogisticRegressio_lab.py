import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt


##  telecommunications dataset for predicting customer churn

dataset = " https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
data = pd.read_csv(dataset)   # read csv file and switch to dataframe format 
data.head()

# Let's select some features for the modeling. Also, we change the target data type to be an integer

data  =  data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
data['churn'] = data['churn'].astype('int') # the churn is the target data type turned from float into  integer 

#data.head()

## define X and y for the dataset 

X = np.asarray(data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
#X[0:5]

y = np.asarray(data['churn'])
## normalize the set 
X = preprocessing.StandardScaler().fit(X).transform(X)

## SPLIT THE DATASET INTO TRAIN AND TEST 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

## MODELING USING LOGISTIC REGRESSION 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

# prediction using the test set

y_hat = LR.predict(X_test)
y_hat

# predict_proba returns estimates for all classes, ordered by the label of classes. 
# So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X)

y_probability = LR.predict_proba(X_test)

# ACCURACY EVALUATION: WE CAN USE THE JACCARD INDEX 
# we can define jaccard as the size of the intersection divided by the size of the union of the two label sets. 
# If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

from sklearn.metrics import jaccard_score

jaccard_score(y_test, y_hat,pos_label=0)

# Another way of looking at the accuracy of the classifier is to look at confusion matrix.

from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
print(confusion_matrix(y_test, y_hat, labels=[1,0]))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


## Let's look at first row. The first row is for customers whose actual churn value in the test set is 1.
# As you can calculate, out of 40 customers, the churn value of 15 of them is 1. 
# Out of these 15 cases, the classifier correctly predicted 6 of them as 1, and 9 of them as 0. 

# This means, for 6 customers, the actual churn value was 1 in test set and classifier also correctly predicted those as 1. 
# However, while the actual label of 9 customers was 1, the classifier predicted those as 0, which is not very good. 
# We can consider it as the error of the model for first row.

# What about the customers with churn value 0? Lets look at the second row.
# It looks like  there were 25 customers whom their churn value were 0. 


# The classifier correctly predicted 24 of them as 0, and one of them wrongly as 1. 
# So, it has done a good job in predicting the customers with churn value 0. 
# A good thing about the confusion matrix is that it shows the model’s ability to correctly predict or separate the classes.  
# In a specific case of the binary classifier, such as this example,  we can interpret these numbers as the count of true positives, 
# false positives, true negatives, and false negatives. 


print (classification_report(y_test, y_hat))

# Precision is a measure of the accuracy provided that a class label has been predicted. 
# It is defined by: precision = TP / (TP + FP)

# Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)
# So, we can calculate the precision and recall of each class.

# F1 score:
# Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label. 

# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. 
# It is a good way to show that a classifer has a good value for both recall and precision.
# Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.

######################
# log loss__ for evaluation. In logistic regression, the output can be the probability of customer churn is yes (or equals to 1). 
# This probability is a value between 0 and 1.
# Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.

from sklearn.metrics import log_loss
log_loss(y_test, y_probability)

## second logistic regression with different solver ('sag')
LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))











