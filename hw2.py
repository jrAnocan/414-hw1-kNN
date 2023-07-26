
from math import sqrt
from sklearn.neighbors import KNeighborsRegressor
import re
import time


#get the min and max values of each column
def columns_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])-1):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

#normalize columns except first column and last column. 
#because we want to predict first column of dataset so we dont use it when calculating euclidian distance.
#last column is name of the cars which is irrelevant to mpg
def normalize(dataset, minmax):
	for row in dataset:
		for i in range(1,len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


#calculate euclidian distance of 2 rows. Excludes 1st and last column while calculating
def euclidean_distance(row1, row2):
	d = 0.0
	for i in range(1,len(row1)-1):
		d += ( row1[i]-row2[i] ) ** 2
	return sqrt(d)


#calculates k nearest neighbours of test_row in train set
#returns a list [[a,b],[],..k] where a is mpg of the neighbour and b is its distance(which is needed for weighted sum) 
def findNeighbours(test_row, train, k):
	dist = []
	for i in range(len(train)):
		
		dist.append( [ i,euclidean_distance(test_row,train[i]) ] )
	dist.sort(key = lambda x: x[1])
	
	res = []
	for i in range(k):
		#print(dist[i][0],train[dist[i][0]])
		res.append([train[dist[i][0]][0],dist[i][1]])
		
	return res

# calculates the weighted mean of neighbours(which is in lst).
# weighted sum = mpg * (1/distance)
# prediction = weighted sum / ( (1/distanceOfFirstElementInList) + .. + ((1/distanceOfkThElementInList))   )
# lst contains k nearest neighbours
def calculateWeightedMean(lst,k):
	res = 0
	weights = 0
	for i in range(k):
		res += lst[i][0]* (1/lst[i][1])
		weights += 1/lst[i][1]
	return res / weights

# predict all test rows for given i.
# i represents the cross validation step. It can take value 0,1 or 2. So 3 fold cross validation.
def predict(testList,trainList,k,i,result):
	for x in testList[i]:
		lst = findNeighbours(x,trainList[i],k)
		result.append(calculateWeightedMean(lst,k))
		

# this function is used for adapting the database to scikit kNeighboursRegressor.
# it fills necessary lists to perform certain functions of this library like fit() and predict()
# while filling the lists it ignores certain columns like last column which is irrelevant to mpg and first column because 
# 	it will be predicted
def fillListsForScikit(testList,trainList,testList2,trainList2,x_train,y_train):
	

	for i in range(3):
		temp = testList[i]
		for j in temp:
			testList2[i].append(j[1:8])
	for i in range(3):
		temp = trainList[i]
		for j in temp:
			x_train[i].append(j[1:8])
			y_train[i].append(j[0])
			trainList2[i].append(j[0:8])


# fills the dataList by reading the data file. Also converts every column to float. Except last one.
def fillDatalist(dataList,f,numOfRows):
	numOfRows = 0
	for line in f.readlines():
		dataList.append(line)
		numOfRows += 1
			
	delimiters = "  ", "\t"
	regexPattern = '|'.join(map(re.escape, delimiters))
	
	for i in range(numOfRows):
		dataList[i] = re.split(regexPattern, dataList[i])
	for i in range(numOfRows):
		for j in range(8):
			dataList[i][j] = float(dataList[i][j])
	return numOfRows


# my Knn implementation. At the end it prints the performance results for the respective k
# starts measuring time in the first line. Measuring time ends when performance is calculated.
# It doesn't measure time for printing the performance.
def myKnn():
	start_time = time.time()			   # <---- start measuring time.
	result = []
	for i in range(3):	
		predict(testList,trainList,k,i,result)		
	
	errorMape = 0
	errorMse = 0
	for i in range(numOfRows):
		errorMse += abs(dataList[i][0]-result[i]) ** 2
		errorMape += abs(dataList[i][0]-result[i]) / dataList[i][0]

	errorMape = errorMape*100/numOfRows
	errorMse =  errorMse/numOfRows
	errorRmse = sqrt(errorMse)
	timeSpent = time.time() - start_time    # <---- end measuring time here.
	print("MAPE from myKnn is ", errorMape)	
	print("MSE from myKnn is ", errorMse)	
	print("RMSE from myKnn is ", errorRmse)
	print("Time spent for calculations: %f" % timeSpent)
	print("--------------------------")
	return result

# scikit implementation. I only used kNeighboursRegressor. I didn't use decisionTreeRegressor.
def scikitPredict():
	start_time = time.time()						# <---- start measuring time.
	res = []
	res2 = []
	neigh = KNeighborsRegressor(n_neighbors=k)
	for i in range(3):

		neigh.fit(x_train[i], y_train[i])
		res.append(neigh.predict(testList2[i]))
	for x in res:
		for y in x:
			res2.append(y)

	errorMape2 = 0
	errorMse2 = 0
		
	for i in range(numOfRows):
		errorMse2 += abs(dataList[i][0]-res2[i]) ** 2
		errorMape2 += abs(dataList[i][0]-res2[i]) / dataList[i][0]

	errorMape2 = errorMape2*100/numOfRows
	errorMse2 =  errorMse2/numOfRows
	errorRmse2 = sqrt(errorMse2)	
	timeSpent = time.time() - start_time				# <---- end measuring time here.
	print("MAPE from Scikit is ", errorMape2)	
	print("MSE from Scikit is ", errorMse2)	
	print("RMSE from Scikit is ", errorRmse2)
	print("Time spent for calculations: %f" % timeSpent)
	print("------------------------------------------------------------")	
	return res2


# prints results of predicting mpg for both of the methods.
def printPredictionOfBothMethods():
	print("-----------Results for k = %d---------------------------" % k)	
	for i in range(numOfRows):
		print(dataList[i][0],resOfMyKnn[i],resOfScikit[i])
	print("-----------------------------------------------------")	

if __name__ == '__main__' :
	for k in range(2,11):
		print("------------------------------------------------------------")
		print("****** predicting with k = %d  ******" % k)
	
		f = open("auto-mpg.data", "r")
		dataList=[]
		
		numOfRows = fillDatalist(dataList,f,0)
		
		normalize(dataList,columns_minmax(dataList))

		firstDivide = numOfRows//3
		secondDivide = numOfRows*2//3
		testList = [dataList[0:firstDivide],dataList[firstDivide:secondDivide],dataList[secondDivide:numOfRows]]	
		trainList = [dataList[firstDivide:numOfRows],dataList[0:firstDivide]+dataList[secondDivide:numOfRows],dataList[0:secondDivide]]
		
		resOfMyKnn = myKnn()
		
		
		testList2 = [[],[],[]]
		trainList2 = [[],[],[]]
		x_train=[[],[],[]]
		y_train=[[],[],[]]

		fillListsForScikit(testList,trainList,testList2,trainList2,x_train,y_train)

		resOfScikit = scikitPredict()

		#printPredictionOfBothMethods()          #<--- you can print each row's prediction with both of the methods by removing #.
												 # order is actual mpg, my prediction, scikit prediction



