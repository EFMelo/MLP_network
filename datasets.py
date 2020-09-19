from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from csv import reader
from numpy import array

class BreastCancer():

    @classmethod
    def load_data(cls, pp=None):
        
        """
        Description
        -----------
        Dataset with 519 training samples and 50 test samples.
        Classes:
            0 - No breast cancer
            1 - With breast cancer
    
        Parameters
        ----------
        pp : str or None
            Pre-processing type.
            The 'mms' or 'std' arguments can be entered, so the MinMaxScaler (mms) or StandardScaler (std) are calculated.
            The None argument doen nothing.
    
        Returns
        -------
        x_train : ndarray
        y_train : ndarray
        x_test : ndarray
        y_test : ndarray
        norm : sklearn.preprocessing
        """
        
        norm = None
        
        # reading cvs files
        x = read_csv('dataset/breast_cancer/input.csv')
        y = read_csv('dataset/breast_cancer/output.csv')
        
        # pre-processing
        if pp == 'mms':
            norm = MinMaxScaler()
        elif pp == 'std':
            norm = StandardScaler()
        
        if pp == None:
            x = x.iloc[0:569, 0:30].values
            y = y.iloc[0:569, 0:30].values
        else:
            x = norm.fit_transform(x.iloc[0:569, 0:30].values)
            y = norm.fit_transform(y.iloc[0:569, 0:30].values)

        # train data
        x_train = x[0:519]
        y_train = y[0:519]

        # test data
        x_test = x[519:569]
        y_test = y[519:569]

        return x_train, y_train, x_test, y_test, norm
    

class WineQuality:
    
    @classmethod
    def load_data(cls, pp=None):

        """
        Description
        -----------
        Dataset with 4500 training samples and 398 test samples.
        Output:
            Wine quality (score between 0 and 10)
    
        Parameters
        ----------
        pp : str or None
            Pre-processing type.
            The 'mms' or 'std' arguments can be entered, so the MinMaxScaler (mms) or StandardScaler (std) are calculated.
            The None argument doen nothing.
    
        Returns
        -------
        x_train : ndarray
        y_train : ndarray
        x_test : ndarray
        y_test : ndarray
        norm : sklearn.preprocessing
        """
        
        norm = None
        cls.__data = []

        # reading cv file
        cls.__read_data()
        
        # pre-processing
        if pp == 'mms':
            norm = MinMaxScaler()
        elif pp == 'std':
            norm = StandardScaler()
        
        if pp != None: 
            x = norm.fit_transform(cls.__data[:, 0:11])
            y = norm.fit_transform(cls.__data[:, 11].reshape(4898, 1))
        else:
            x = cls.__data[:, 0:11]
            y = cls.__data[:, 11].reshape(4898, 1)
        
        # train data
        x_train = x[0:4500]
        y_train = y[0:4500]

        # test data
        x_test = x[4500:4898]
        y_test = y[4500:4898]

        return x_train, y_train, x_test, y_test, norm
    
    
    @classmethod
    def __read_data(cls):
        
        with open('dataset/wine_quality/winequality_data.csv') as arch:
            all_data = reader(arch, delimiter=';')
            next(all_data)

            for value in all_data:
                cls.__data.append([float(v) for v in value])
            
            cls.__data = array(cls.__data)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            