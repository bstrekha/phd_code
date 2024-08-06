import numpy as np 

def hash_array(arr):
    '''
    Take an array (representing a device or a set of lags/violations, which are often very long arrays)
    and come up with a "unique" name for the .npy 
    '''
    return hash(arr.tobytes()) 

def convert_csv_hash(dataframe, fileName, colName, folderName):
    '''
    Take a data frame, for each row, grab the colName array and replace it with its hash. 
    Save the .npy of the array as folderName/{hash}.npy
    Save the data frame as fileName.csv. 
    '''
    pass 