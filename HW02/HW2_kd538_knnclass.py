from numpy import array, square
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from time import time
from collections import Counter
from heapq import heappush, heappushpop

# k-NN classification implementation as requested in Part-II
def knnclass(xtrain, xtest, ytrain):
    def MSE(ytest, ypred):  # same as computing square of L2 distance
        # ytest and ypred should be 1d numpy arrays of the same length
        return square(ytest - ypred).mean()

    def mode(arr):
        c = Counter(arr)
        return max(c, key=lambda x: c[x])
    
    def knn(xtrain, xtest, ytrain, k):  # perform k-NN classification when 'k' is set
        start_t = time()
        ytest = []
        for sample in xtest:
            pq = []  # emulate a max-heap, Nlog(k)
            for x, y in zip(xtrain, ytrain):
                dist = MSE(sample, x)
                if len(pq) < k:
                    heappush(pq, (- dist, y))
                elif dist < - pq[0][0]:
                    heappushpop(pq, (- dist, y))
            ytest.append(mode([y for _, y in pq]))
        print('Finished validating kNN with k =', k, 'using', time() - start_t, 's')
        return array(ytest)
    
    subtrain, cv = next(ShuffleSplit(test_size=.2, random_state=0).split(xtrain))
    # standardize the train & test data using only mean & deviation from train data
    scaler = StandardScaler().fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)
    # for k in range(1, 15):
    #     print(k, MSE(ytrain[cv], knn(xtrain[subtrain, :], xtrain[cv, :], ytrain[subtrain], k)))
    K = max(range(1, 10), key=lambda k: MSE(ytrain[cv], knn(xtrain[subtrain, :], xtrain[cv, :], ytrain[subtrain], k)))
    # print('Optimal K =', K)
    return knn(xtrain, xtest, ytrain, K)