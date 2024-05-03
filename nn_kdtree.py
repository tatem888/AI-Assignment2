import numpy as np
import sys
import pandas as pd

#read command line and return file paths and the first dimension to be compared
def readCommandLine():

    trainingDataFile = sys.argv[1]
    testDataFile = sys.argv[2]
    initialDimension = sys.argv[3]

    return (trainingDataFile, testDataFile, initialDimension)


def createDataFrames(trainingDataFile, testDataFile):

    trainingDataFrame = pd.read_fwf(trainingDataFile)
    testDataFrame = pd.read_fwf(testDataFile)

    return (trainingDataFrame, testDataFrame)


class Node:
    def __init__(self, value, dimension):
        self.point = None
        self.value = value
        self.dimension = dimension
        self.left = None
        self.right = None


def buildKDTree(trainingDataFrame, currentDepth):

    #calculate dimension and value 
    dimension = currentDepth % 11
    value = trainingDataFrame.median().iloc[dimension]

    #convert dimension to column label string for pandas use
    dimensionString = trainingDataFrame.columns[dimension]
    

    #if frame is empty return null
    if trainingDataFrame.empty == True:
        return None
    
    #if only one point in frame (Bottom of tree, create leaf node), create node and return
    elif trainingDataFrame.shape[0] == 1:

        #create and return single new node
        point = trainingDataFrame.index[0]

        node = Node(value, dimension)
        node.point = point

        return node


    else:
    
        #create new node
        node = Node(value, dimension)

        #split training data frame and allocate to the left or right child node
        
        newDFleft = trainingDataFrame[trainingDataFrame[dimensionString] <= value]
        newDFRight = trainingDataFrame[trainingDataFrame[dimensionString] > value]

        node.left = buildKDTree(newDFleft, currentDepth+1)
        node.right = buildKDTree(newDFRight, currentDepth+1)

        return node
    
#Implement tree search

#for each line in testing input, run search alg to find nearest neighbour 


#pass in single line dataframe
def NNSearchKDTree(testDataFrame, currentNode, currentDepth):

    
    #naive implementation, drop to bottom of tree
    #get initial dimension and compare test data dimension to root node dimension

    dimension = currentDepth % 11
    rootValue = currentNode.value

    currentNode = currentNode

    testNumPy = testDataFrame.to_numpy()
    testCurrentValue = testNumPy[dimension]

    if currentNode.left is None and currentNode.right is None:

        return currentNode
        

    elif testCurrentValue > rootValue:

        return NNSearchKDTree(testDataFrame, currentNode.right, currentDepth+1)

    elif testCurrentValue <= rootValue:
        
        return NNSearchKDTree(testDataFrame, currentNode.left, currentDepth+1)

    
##Test



trainingDataFile = sys.argv[1]
testDataFile = sys.argv[2]

initialDimension = int(sys.argv[3])


dfs = createDataFrames(trainingDataFile, testDataFile)
trainingDataFrame = dfs[0]
testDataFrame = dfs[1]

tree_test = buildKDTree(trainingDataFrame, initialDimension)

numOfTests = testDataFrame.shape[0]

index = 0
while index < numOfTests:
    
    currentTestFrame = testDataFrame.iloc[index]

    nearestNeighbour = NNSearchKDTree(currentTestFrame, tree_test, initialDimension)
    

    print(trainingDataFrame.iloc[nearestNeighbour.point,11])

    index = index+1

