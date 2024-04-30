def knn(url, k, rings, unnecessary_column):
    import pandas as pd
    import numpy as np
    abalone = pd.read_csv(url)
    try:
        abalone = abalone.drop(unnecessary_column, axis=1) #drop a column with strings in it
    except:
        abalone = abalone #if no column is supplied/a nonexistent column is supplied, don't do anything
    X = abalone.drop(rings, axis=1) #get the data points for all values minus the part that needs to be found out
    X = X.values #collect their values
    y = abalone[rings] #look at the requested values for all the supplied data points
    y = y.values #collect their values
    num_of_points = int(input("How many values?"))
    arraylist = []
    for i in range(1, num_of_points+1): #collect all the available data about the requested data point
        arraylist.append(float(input("What is the value of the data point?")))
    new_data_point = np.array(arraylist)
    distances = np.linalg.norm(X - new_data_point, axis=1) #calculate the distances between the new point and each existing point
    nearest_neighbor_ids = distances.argsort()[:k] #choose the k closest data points
    nearest_neighbor_rings = y[nearest_neighbor_ids] #check the requested value for each nearby data point
    prediction = nearest_neighbor_rings.mean() #calculate the mean of the requested value for each nearby data point
    return prediction #return the prediction

print(knn('https://raw.githubusercontent.com/rodolfo-mendes/abalone/master/data/abalone.csv', 3, "Rings", "Sex"))
