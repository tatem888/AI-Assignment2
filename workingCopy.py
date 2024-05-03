import numpy as np
import sys
import pandas as pd

#read command line and return file paths and the first dimension to be compared
def readCommandLine():

    trainingDataFile = sys.argv[1]
    testDataFile = sys.argv[2]
    initialDimension = int(sys.argv[3])

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

#euclidian distance calculation using numpy

def euclideanDistance(point1, point2):

    distance = np.linalg.norm(point1 - point2)
    return distance
    

def NNSearchKDTree(testDataFrame, currentNode, currentDepth):
    
    if currentNode is None:
        return None
    
    dimension = currentDepth % 11
    rootValue = currentNode.value
    testValue = testDataFrame[dimension]

    nextNode = None
    backTrackNode = None

    if currentNode.left is None and currentNode.right is None:

        return currentNode.point
        

    if testValue <= rootValue:

        nextNode = currentNode.left
        backTrackNode = currentNode.right

    else:

        nextNode = currentNode.right
        backTrackNode = currentNode.left
    
    best = NNSearchKDTree(testDataFrame, nextNode, currentDepth+1)


    # if distance between test point and current best is larger than the current node value
    # backtrack and check  backtrack node, and update if required, run recursion again
    if euclideanDistance(testDataFrame, best) > abs(rootValue-testValue):

        potentialBest = NNSearchKDTree(testDataFrame, backTrackNode,currentDepth+1)

        if euclideanDistance(testDataFrame, testDataFrame) < euclideanDistance(testDataFrame, best):
        
            best = potentialBest
    
    return best

    
##Test

trainingDataFile, testDataFile, initialDimension = readCommandLine()
dfs = createDataFrames(trainingDataFile, testDataFile)
trainingDataFrame = dfs[0]
testDataFrame = dfs[1]

KDTree = buildKDTree(trainingDataFrame, initialDimension)

numOfTests = testDataFrame.shape[0]

for index in range(numOfTests):
    
    currentTestFrame = testDataFrame.iloc[index]

    nearestNeighbour = NNSearchKDTree(currentTestFrame, KDTree, initialDimension)
    

    print(nearestNeighbour)

