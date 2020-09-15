from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataSet():

    @classmethod
    def breast_cancer(cls, pp=None):
       
        """
        Description
        -----------
        Dataset with 519 training samples and 50 test samples.
        Classes:
            0 - Don't have breast cancer
            1 - Have breast cancer

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
        
        # reading cvs files
        x = read_csv('dataset/breast_cancer/input.csv')
        y = read_csv('dataset/breast_cancer/output.csv')
        
        # pre-processing
        if pp == 'mms':
            norm = MinMaxScaler()
        elif pp == 'std':
            norm = StandardScaler()
            
        x = norm.fit_transform(x.iloc[0:569, 0:30].values)
        y = norm.fit_transform(y.iloc[0:569, 0:30].values)

        # train data
        x_train = x[0:519]
        y_train = y[0:519]

        # test data
        x_test = x[519:569]
        y_test = y[519:569]

        return x_train, y_train, x_test, y_test, norm