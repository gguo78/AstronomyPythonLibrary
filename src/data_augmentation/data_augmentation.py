import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
import ast 
import os


class DataAugmentation:
    """
    Class for performing spectral analysis on astronomical data.
    It computes the fractional derivatives of flux with respect to wavelength.
    """
    
    def __init__(self,datapath=None,dataframe=None):
        """
        Initializes the class with the data.
        Args:
            datapath (str, optional): Path to a csv file containing the data. Defaults to None.
            dataframe (pd.DataFrame, optional): The dataset as a dataframe. Defaults to None.
        """
        if datapath is None and dataframe is None:
            raise ValueError("One of data_folder or dataframe must be passed in")

        # Load the data
        if datapath is not None:
            print(f"checking for presaved csv at path: {datapath}")
            assert os.path.exists(datapath), "Data path not found."
            try:
                data = pd.read_csv(datapath)
            except:
                raise ValueError("Unknown error occured reading file.")
        else:
            data = dataframe
            assert isinstance(data, pd.DataFrame)

        self.data = data
    
    # def convert_list(self):
    #     """
    #     Convert lam and flux into correct list format.
    #     """
    #     self.data['lam'] = self.data['lam'].apply(ast.literal_eval)
    #     self.data['flux'] = self.data['flux'].apply(ast.literal_eval)


    def fractional_derivative(self, flux, log_wavelength, order=1):
        """
        Computes the fractional derivative of the flux with respect to the log of the wavelength.
        :param flux: A list or array of flux values.
        :param log_wavelength: A list or array of the logarithm of wavelength values.
        :param order: The order of the derivative. Defaults to 1 for the first derivative.
        :return: An array of the derivative of flux with respect to log wavelength.
        """
        flux_interpolated = interp1d(log_wavelength, flux, kind='cubic', fill_value="extrapolate")
        
        derivative_flux = np.array([derivative(flux_interpolated, lw, dx=1e-6, n=order, order=3)
                                    for lw in log_wavelength])
        
        return derivative_flux
    
    def process_data(self):
        """
        Main method to process the data and add a new column with the fractional derivative.
        """
        self.data['frac_dev'] = self.data.apply(
            lambda row: self.fractional_derivative(row['flux'], row['lam']), axis=1)


    

    
