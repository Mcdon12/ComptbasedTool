def onehot_encode(df, column_dict):
    df = df.copy() # makin a df copy so as not to manipulate the existing data set
    for column, prefix in column_dict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


#Functin to 
#Preprocessing the scaler function to fit and transform the dataSet subseqient

def preprocess_inputs(df, scaler):
    df = df.copy()
    
    # One-hot encode the nominal features
    nominal_features = ['cp', 'slp', 'thall']
    df = onehot_encode(df, dict(zip(nominal_features, ['CP', 'SL', 'TH'])))
    
    # Split df into X and y
    y = df['output'].copy()
    X = df.drop('output', axis=1).copy() # where we are dropping the target column to predict
    
    # Scale X
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X, y



# To store the newDF into the HD5 and CSV file
# h5File = "heart_problem.h5";
def exportCsv():
    X.to_csv("heart_problem.csv", sep='\t')
    X.to_hdf("heart_problem.h5", "/data")

    # df1 = pd.read_hdf("heart_problem.h5", "/data");
    # print("DataFrame read from the HDF5 file through pandas:");
    # print(df1);