import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
import ast
import os

class SpectralVisualizer:
    """
    SpectralVisualizer class for visualizing spectral data.
    
    Parameters:
    - datapath (str, optional): Path to a csv file containing the data. Defaults to None.
    - dataframe (pd.DataFrame, optional): The dataset as a dataframe. Defaults to None.
    - wavelength_column (str): Column name for Wavelength data. 
    - flux_column (str): Column name for flux data.

    Methods:
    - _prepare_data(): Prepare data for visualization.
    - _calculate_average_flux(): Calculate the average flux from the interpolated flux values.
    - _denoise_flux(): Apply Savitzky-Golay filter for denoising the average flux.
    - plot_spectral_visualization(): Plot the original and denoised flux for spectral visualization.
    """
    def __init__(self, datapath=None, dataframe=None, wavelength_column='lam', flux_column='flux'):
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
        self.wavelength_column = wavelength_column
        self.flux_column = flux_column      

    def _prepare_data(self):
        if isinstance(self.data[self.wavelength_column][0], str) or isinstance(self.data[self.flux_column][0], str):
            self.data[self.wavelength_column] = self.data[self.wavelength_column].apply(ast.literal_eval)
            self.data[self.flux_column] = self.data[self.flux_column].apply(ast.literal_eval)

        min_value = np.min(np.concatenate(self.data[self.wavelength_column].tolist()))
        max_value = np.max(np.concatenate(self.data[self.wavelength_column].tolist()))
        self.common_wavelength = np.linspace(min_value, max_value, 200)

        self.interpolated_flux = [np.interp(self.common_wavelength, lam_row, flux_row) 
                                  for lam_row, flux_row in zip(self.data[self.wavelength_column], 
                                                               self.data[self.flux_column])]

    def _calculate_average_flux(self):
        self.average_flux = np.mean(self.interpolated_flux, axis=0)

    def _denoise_flux(self):
        self.flux_denoised = savgol_filter(self.average_flux, window_length=5, polyorder=2)

    def plot_spectral_visualization(self):
        self._prepare_data()
        self._calculate_average_flux()
        self._denoise_flux()

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(self.common_wavelength, self.average_flux, label='Original', alpha=0.5)
        plt.plot(self.common_wavelength, self.flux_denoised, label='Denoised', color='red')
        plt.xlabel('Wavelength (lam)')
        plt.ylabel('Flux')
        plt.title('Spectral Visualization over Inferred Continuum')
        plt.legend()
        plt.show()

#visualizer = SpectralVisualizer(datapath='augmented_data.csv')
#visualizer = SpectralVisualizer(dataframe=augmented_data)
#visualizer.plot_spectral_visualization()
