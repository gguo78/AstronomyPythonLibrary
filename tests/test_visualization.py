import sys
import os
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.visualization import SpectralVisualizer


class TestSpectralVisualizer:
    @pytest.fixture(autouse=True)
    def initialize(self):
        data_file_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "data_augmentation", "augmented_data.csv"
        )
        self.visualizer = SpectralVisualizer(data_file_path)

    def test_init(self):
        assert isinstance(self.visualizer.data, pd.DataFrame)
        assert self.visualizer.wavelength_column in self.visualizer.data.columns
        assert self.visualizer.flux_column in self.visualizer.data.columns

    def test_prepare_data(self):
        self.visualizer._prepare_data()
        assert hasattr(self.visualizer, 'common_wavelength')
        assert hasattr(self.visualizer, 'interpolated_flux')
        assert len(self.visualizer.interpolated_flux) == len(self.visualizer.data)

    def test_calculate_average_flux(self):
        self.visualizer._prepare_data()
        self.visualizer._calculate_average_flux()
        assert hasattr(self.visualizer, 'average_flux')
        assert len(self.visualizer.average_flux) == 200 

    def test_denoise_flux(self):
        self.visualizer._prepare_data()
        self.visualizer._calculate_average_flux()
        original_average_flux = self.visualizer.average_flux.copy()
        self.visualizer._denoise_flux()
        assert hasattr(self.visualizer, 'flux_denoised')
        assert not np.array_equal(self.visualizer.flux_denoised, original_average_flux)

    def test_plot_spectral_visualization(self, monkeypatch):
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        self.visualizer.plot_spectral_visualization()
        assert plt.gcf() is not None
