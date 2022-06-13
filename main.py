from classes import ReadfilExport, Preprocess, TrainDataClass, Visualize
from sklearn.preprocessing import MinMaxScaler

class Init:
    def __init__(self):

        readfile = ReadfilExport()
        
        url = 'heart.csv'
        tesdf=readfile.readFile(url)

        tesdf

        visualize = Visualize(tesdf)
        visualize.visualize()
        preprocess = Preprocess(tesdf)

        onehotEcode = preprocess.onehot_encode()
        # onehotEcode

        X, y = preprocess.preprocess_inputs(tesdf, MinMaxScaler())

        train =  TrainDataClass(X,y)
        X_train, X_test, y_train, y_test = train.splitData() 

        svmAccuracy = train.modelSVC(X_train, y_train, X_test, y_test)

        print("Accuracy for SVM is:")
        print(svmAccuracy)


        
Init()

