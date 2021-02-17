'''
Lab 3
'''

######### Part 1 ###########


'''
    1) Download the iris-data-1 from Canvas, use pandas.read_csv to load it.

'''
# YOUR CODE GOES HERE
import pandas as pd
with open('iris-data-1.csv') as csvfile:
    dff = pd.read_csv(csvfile, delimiter = ',')
    
X = dff[['sepal_length',  'sepal_width',  'petal_length',  'petal_width']]
y = (dff["species"])

'''
    2) Split your data into test set(%30) and train set(%70) randomly. (Hint: you can use scikit-learn package tools for doing this)
    
'''
# YOUR CODE GOES HERE 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
        
'''    
    3) Use KNeighborsClassifier from scikit-learn package. Train a KKN classifier using your training dataset  (K = 3, Euclidean distance).   
    
'''
# YOUR CODE GOES HERE  
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

'''   
    4) Test your classifier (Hint: use predict method) and report the performance (report accuracy, recall, precision, and F1score). (Hint: use classification_report from scikit learn)
'''

# YOUR CODE GOES HERE
from sklearn.metrics import classification_report

pred = knn.predict(X_test)
print (classification_report(y_test,pred))

'''   
    5) report micro-F1score, macro-F1score, and weighted F1-score.
'''

# YOUR CODE GOES HERE
from sklearn.metrics import f1_score
print (f1_score(y_test, pred, average='macro'))
print (f1_score(y_test, pred, average='micro'))
print (f1_score(y_test, pred, average='weighted'))

'''    
    6) Repeat Q3, Q4, and Q5 for "manhattan" distance function

'''
# YOUR CODE GOES HERE
knn2 = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn2.fit(X_train, y_train)

pred2 = knn2.predict(X_test)
print (classification_report(y_test,pred2))

print (f1_score(y_test, pred2, average='macro'))
print (f1_score(y_test, pred2, average='micro'))
print (f1_score(y_test, pred2, average='weighted'))

'''   
    7) Compare your results in Q5 and Q6.

'''
print ("euclidean distance function", classification_report(y_test,pred))
print ("manhattan distance function", classification_report(y_test,pred2))


'''
    8) Repeat Q3, Q4, Q5, Q6, and Q7 for K = 11.
'''
# YOUR CODE GOES HERE
knn3 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn3.fit(X_train, y_train)

pred3 = knn3.predict(X_test)
print (classification_report(y_test,pred3))

print (f1_score(y_test, pred3, average='macro'))
print (f1_score(y_test, pred3, average='micro'))
print (f1_score(y_test, pred3, average='weighted'))

knn4 = KNeighborsClassifier(n_neighbors=11, metric='manhattan')
knn4.fit(X_train, y_train)

pred4 = knn4.predict(X_test)
print (classification_report(y_test,pred4))

print (f1_score(y_test, pred4, average='macro'))
print (f1_score(y_test, pred4, average='micro'))
print (f1_score(y_test, pred4, average='weighted'))

######### Part 2 ###########
'''
    0)  Repeat Q1 and Q2 in part 1.

'''
# YOUR CODE GOES HERE


'''
    1) Train a KKN classifier using your training dataset  (K = 7, Euclidean distance). 
    
    1-1) Test your classifier using predict_proba method. What is the difference between predict_proba and predict method?
    
    1-2) report the performance based on your results in 1-1.
    
'''
# YOUR CODE GOES HERE
import numpy as np

knn5 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
knn5.fit(X_train, y_train)

pred_prob = knn5.predict_proba(X_test)  # it outputs the probability associated to each class
pred5 = np.argmax(pred_prob, axis =1)  # we have access to the indices not actual labels


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
transformed_y = le.fit_transform(y_test) # now we can compare pred5 with this, another solution is to map pred5 to classes names!

print (le.classes_)

print (classification_report(transformed_y,pred5))

######### Part 3 ###########

'''
    0) Repeat Q1 and Q2 in part 1.

'''
# YOUR CODE GOES HERE

'''
    1) Use DecisionTreeClassifier from scikit-learn package. Train a DT classifier using your training dataset  (criterion='entropy', splitter= 'best'). 

'''
# YOUR CODE GOES HERE
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(criterion='entropy', splitter= 'best')
DT.fit(X_train, y_train)

'''   
    2) Test your classifier (Hint: use predict method) and report the performance (report accuracy, recall, precision, and F1score). (Hint: use classification_report from scikit learn)
'''

# YOUR CODE GOES HERE
pred = DT.predict(X_test)
print (classification_report(y_test,pred))

'''   
    3) report micro-F1score, macro-F1score, and weighted F1-score
'''

# YOUR CODE GOES HERE
print (f1_score(y_test, pred, average='macro'))
print (f1_score(y_test, pred, average='micro'))
print (f1_score(y_test, pred, average='weighted'))

'''    
    4) Repeat Q1, Q2, and Q3 for "random" splitter.
'''
# YOUR CODE GOES HERE
DT1 = DecisionTreeClassifier(criterion='entropy', splitter= 'random')
DT1.fit(X_train, y_train)

pred1 = DT1.predict(X_test)
print (classification_report(y_test,pred1))

print (f1_score(y_test, pred1, average='macro'))
print (f1_score(y_test, pred1, average='micro'))
print (f1_score(y_test, pred1, average='weighted'))

'''   
    5) Compare your results in Q4 and Q3.

'''
# YOUR CODE GOES HERE
print ("best splitter", classification_report(y_test,pred))
print ("random splitter", classification_report(y_test,pred1))

'''   
    6) Repeat Q2, Q3, Q4, and Q5 for criterion = "gini".

'''
# YOUR CODE GOES HERE

DT2 = DecisionTreeClassifier(criterion='gini', splitter= 'best')
DT2.fit(X_train, y_train)

pred2 = DT2.predict(X_test)
print (classification_report(y_test,pred2))

print (f1_score(y_test, pred2, average='macro'))
print (f1_score(y_test, pred2, average='micro'))
print (f1_score(y_test, pred2, average='weighted'))

DT3 = DecisionTreeClassifier(criterion='gini', splitter= 'random')
DT3.fit(X_train, y_train)

pred2 = DT3.predict(X_test)
print (classification_report(y_test,pred3))

print (f1_score(y_test, pred3, average='macro'))
print (f1_score(y_test, pred3, average='micro'))
print (f1_score(y_test, pred3, average='weighted'))