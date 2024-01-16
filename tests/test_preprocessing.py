import os
import shutil
import numpy as np
import pytest
import pandas as pd
from src.data_retrieval import GetData
from src.preprocessing import CleanSpectralData


class Test_CleanSpectralData:
    @pytest.fixture(autouse=True)
    def initialize(self, request):
        # Create a temporary directory
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Get some data and also save it to a csv
        data_retriever = GetData()
        self.df = data_retriever.retrieve_sdss_data(
            sql_query="""
                        SELECT TOP 5 s.fiberid, s.plate, s.mjd, s.run2d, s.class
                        FROM PhotoObj AS p
                        JOIN SpecObj AS s ON s.bestobjid = p.objid
                        """
        )
        self.csv_path = os.path.join(temp_dir, "data.csv")
        self.df.to_csv(self.csv_path)

        # Initialze the cleaner module
        self.module = CleanSpectralData(dataframe=self.df)

        def teardown():
            # Remove the temporary directory after the test is done
            shutil.rmtree(temp_dir)

        request.addfinalizer(teardown)

    def test_init(self):
        # catch an error if no initialization method was given
        with pytest.raises(ValueError) as e:
            CleanSpectralData()

        # Check bad csv path input
        with pytest.raises(AssertionError) as e:
            CleanSpectralData(datapath="badpath")

        # Verify good initializations
        CleanSpectralData(datapath=self.csv_path)
        CleanSpectralData(dataframe=self.df)

        return

    def test_clean_data(self):
        cleaned_data = self.module.clean_data().copy()

        # after cleaning, flux and lam should not be a string literal but a list of floats
        # Check if 'flux' and 'lam' are lists of floats
        flux = cleaned_data["flux"].values
        assert len(flux) == 5, "Test expected to get a set of 5 fluxes."
        assert isinstance(flux[0], list), "expected flux to be a list of lists"
        assert all(
            isinstance(x, float) for x in flux[0]
        ), "Expected floats as the row's flux values"
        assert cleaned_data.isnull().sum().sum() == 0
        assert cleaned_data.columns.duplicated().sum() == 0

        return

    def test_interpolate_flux(self):
        # Prepare a known lambda_set for interpolation
        lambda_set = np.linspace(4000, 6000, 10)
        interpolated_fluxes = self.module.interpolate_flux(lambda_set)
        assert len(interpolated_fluxes) == len(self.module.data)
        return

    def test_get_normalize_flux(self):
        normalized_fluxes = self.module.get_normalize_flux()
        # Check if all values are between 0 and 1
        for flux_list in normalized_fluxes:
            assert all(0 <= x <= 1 for x in flux_list)
        return

    def test_remove_flux_outliers_iqr(self):
        # Create sample data with known outliers
        data = {
            "flux": [[2, 3, 100, 5, 6], [1, 2, 3, 4, 5], [10, 300, 20, 30, 40]],
            "lam": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
            "loglam": list(
                np.log10(np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]))
            ),
        }
        df = pd.DataFrame(data)
        module = CleanSpectralData(dataframe=df)
        cleaned_df = module.remove_flux_outliers_iqr()

        for flux in cleaned_df["flux"]:
            flux_array = np.array(flux)  # Each flux list is handled individually

            # Recalculate IQR on cleaned data
            Q1 = np.percentile(flux_array, 25)
            Q3 = np.percentile(flux_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Assert no values are outliers
            assert np.all((flux_array >= lower_bound) & (flux_array <= upper_bound))

    def test_get_inferred_continuum(self):
        # The basic idea of this test is that the denoised should always have less variance than the raw flux
        raw_flux = self.module.data["flux"].values
        inferred_continuum = self.module.get_inferred_continuum(sigma=5.0)
        assert len(raw_flux) == len(
            inferred_continuum
        ), "Should return same number of spectral curves"

        for i in range(len(raw_flux)):
            assert np.var(raw_flux[i]) >= np.var(
                inferred_continuum[i]
            ), "Inferred continuum cant have more variance than raw spectral curves."

        return

