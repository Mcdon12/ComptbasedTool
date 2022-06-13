from classes import ReadfilExport, Preprocess, TrainDataClass
from sklearn.preprocessing import MinMaxScaler

class Init:
    def __init__(self):

        readfile = ReadfilExport()
        
        url = '/Users/mcdons/Downloads/heart.csv'
        tesdf=readfile.readFile(url)

        tesdf

        preprocess = Preprocess(tesdf)
        onehotEcode = preprocess.onehot_encode()
        # onehotEcode

        X, y = preprocess.preprocess_inputs(tesdf, MinMaxScaler())

        train =  TrainDataClass(X,y)
        X_train, X_test, y_train, y_test = train.splitData() 

        svmAccuracy = train.modelSVC(X_train, y_train, X_test, y_test)

        print(svmAccuracy)

        
Init()
