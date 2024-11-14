import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        medians = np.median(x, axis=0)
        abs_deviations = np.abs(x - medians)
        mean_abs_deviation = np.mean(abs_deviations, axis=0)
        return mean_abs_deviation

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        self.loc = np.median(features, axis=0)
        self.scale = self.mean_abs_deviation_from_median(features)


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        abs_diffs = np.abs(values - self.loc)
        logpdf_values = -np.log(2 * self.scale) - abs_diffs / self.scale
        return logpdf_values
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
