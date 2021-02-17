'''
Lab 4
'''

######### Part 1 ###########


'''
    0) Download the iris-data-1 from Canvas, use pandas.read_csv to load it. Split your data into test set(%20) and train set(%80) randomly.
    
'''
# YOUR CODE GOES HERE   
    
        
'''    
    1) Train a KNN classifier with your training data. You need to use CV techniques to tune the following hyper-params:
        a) metric = {chebyshev, euclidean, manhattan}
        b) k = {1, 3, 5, 7, 9, 11, 13, 15}
    
    1-1) Use hold-out validation method to tune the hyper-params. (use 30% of your training data as a test set).  
    1-2) Use 10Fold-CV validation method to tune the hyper-params. (there are multiple ways for implementing it with sklearn).
    1-2-1) For each metric (e.g. chebyshev) plot the results of classifiers (e.g. F1-score or accuracy) vs k. 
'''
# YOUR CODE GOES HERE  

'''   
    2) Test your trained best classifiers in previous part.
'''
# YOUR CODE GOES HERE  


######### Part 2 ###########

'''   
    1) We want to see how normalization of the features affect the results in previous part. We will try two different normailizer fom sklearn:
    
    1-1) Use StandardScaler() to normalize your training data. Repeat Q1 and Q2 in part 1.
    1-2) Use  MinMaxScaler() to normalize your training data. Repeat Q1 and Q2 in part 1.
'''

# YOUR CODE GOES HERE  