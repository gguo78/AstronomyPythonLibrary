import pandas as pd
import os
import ast
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import numpy as np


class CleanSpectralData:
    """Module to clean the dataframe and reformat the flux and wavelength features for post-processing."""

    def __init__(self, datapath=None, dataframe=None):
        """Initialize the cleaner module.

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
        self.clean_data()

    def clean_data(self):
        """Cleans the stored (and returned) dataframe. Flux and lambda is changed from str to list of floats, NaN are removed, and the dataframe is checked for duplicate columns,rows.

        Returns:
            dataframe: cleaned dataframe (pointing to frame stored in the class).
        """
        for index, row in self.data.iterrows():
            lam = row["lam"]
            flux = row["flux"]

            # Check if 'lam' and 'flux' are strings and convert them to lists if they are
            if isinstance(lam, str):
                try:
                    lam = ast.literal_eval(lam)
                except ValueError as e:
                    print(f"Error parsing lam in row {index}: {e}")
                    continue  # Skip to the next iteration

            if isinstance(flux, str):
                try:
                    flux = ast.literal_eval(flux)
                except ValueError as e:
                    print(f"Error parsing flux in row {index}: {e}")
                    continue  # Skip to the next iteration

            self.data.at[index, "flux"] = flux
            self.data.at[index, "lam"] = lam
            self.data.at[index, "loglam"] = np.log10(lam)

        # Drop NaN
        self.data.dropna(inplace=True)

        # drop duplicate columns, if any
        duplicate_columns = set()
        for i in range(len(self.data.columns)):
            col = self.data.columns[i]
            for j in range(i + 1, len(self.data.columns)):
                other_col = self.data.columns[j]
                if col == other_col or (self.data[col].equals(self.data[other_col])):
                    duplicate_columns.add(other_col)
        self.data = self.data.drop(columns=list(duplicate_columns))

        print("cleaned...")
        return self.data

    def interpolate_flux(self, lambda_set):
        """Reinterpolates the flux data to align with the input wavelength vector. Internal dataframe is updated and interpolated fluxes are also returned.

        Args:
            lambda_set (list): list of floats corresponding to the new wavelength array in units 1/A (inverse angstroms).

        Returns:
            list: list of interpolated fluxes
        """

        loglam = np.log10(lambda_set)
        interpolated_fluxes = []
        for index, row in self.data.iterrows():
            # Convert strings to arrays
            lam = row["lam"]
            flux = row["flux"]
            if (
                isinstance(lam, list)
                and isinstance(flux, list)
                and len(lam) == len(flux)
            ):
                # Create an interpolation function
                interpolate_function = interp1d(
                    lam,
                    flux,
                    kind="linear",
                    bounds_error=False,  # Raises an error if extrapolation is attempted
                    fill_value="extrapolate",
                )

                # Interpolate flux values onto the new wavelength grid
                interpolated_flux = interpolate_function(lambda_set)
                interpolated_fluxes.append(interpolated_flux)

                # Replace the flux data in the dataframe
                self.data.at[index, "flux"] = interpolated_flux
                self.data.at[index, "lam"] = lambda_set
                self.data.at[index, "loglam"] = loglam
            else:
                # Handle error or inconsistent data
                print(f"Row {index} has invalid data. Skipping.")
                interpolated_fluxes.append(None)

        return interpolated_fluxes

    def align_wavelengths(self, num_wl):
        """Aligns the fluxes onto the same wavelength grid, linearly spaced.

        Args:
            num_wl (int): Number of wavelengths

        Returns:
            list: New wavelength grid
            list: Lists of interpolated fluxes aligned with the grid
        """
        lam = np.linspace(
            np.max(self.data.lam.min()), np.min(self.data.lam.max()), num_wl
        )
        return lam, self.interpolate_flux(lam)

    def remove_flux_outliers_iqr(self):
        """Remove outliers in the stored dataframe (same as returned) according to IQR criteria. Outliers are replaced with interpolated values.

        Returns:
            dataframe: dataframe with outlier flux removed.
        """
        for index, row in self.data.iterrows():
            flux = np.array(row["flux"])
            lam = np.array(row["lam"])

            if len(flux) > 2:  # Need at least 3 points to perform interpolation
                # Calculate IQR
                Q1 = np.percentile(flux, 25)
                Q3 = np.percentile(flux, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify outliers
                outlier_indices = np.where((flux < lower_bound) | (flux > upper_bound))[
                    0
                ]

                # Prepare data for interpolation
                non_outlier_indices = np.where(
                    (flux >= lower_bound) & (flux <= upper_bound)
                )[0]
                non_outlier_flux = flux[non_outlier_indices]
                non_outlier_lam = lam[non_outlier_indices]

                if (
                    len(non_outlier_flux) > 1
                ):  # Check if there are enough points for interpolation
                    # Create interpolation function
                    interpolate_function = interp1d(
                        non_outlier_lam,
                        non_outlier_flux,
                        kind="linear",
                        fill_value="extrapolate",
                    )

                    # Interpolate values for outliers
                    interpolated_flux = interpolate_function(lam[outlier_indices])

                    # Replace outlier values with interpolated values
                    flux[outlier_indices] = interpolated_flux
                    self.data.at[index, "flux"] = list(flux)

        return self.data

    def correct_redshift(self):
        # Correction is formula lam_em = lam_obs / (1 + z)
        # We need to go through each lam-flux entry and apply a correction to the axis wavelength
        for index, row in self.data.iterrows():
            lam = row["lam"]
            redshift = row["redshift"]
            if isinstance(lam, list) and isinstance(redshift, float):
                new_lam = np.array(lam) / (1 + redshift)
                self.data.at[index, "lam"] = list(new_lam)
                self.data.at[index, "loglam"] = list(np.log10(new_lam))
            else:
                # Handle error or inconsistent data
                print(f"Row {index} has invalid data. Skipping.")

        return self.data

    def get_normalize_flux(self, update_df=False):
        """Returns all flux data but normalized in the range [0, 1].

        Args:
            update_df (boolean, optional): If true, replaces flux values with the normalized ones.

        Returns:
            list: list of normalized fluxes for all rows in the dataframe
        """
        normalized_fluxes = []
        for index, row in self.data.iterrows():
            flux = np.array(row["flux"])

            # Perform min-max normalization
            min_flux = np.min(flux)
            max_flux = np.max(flux)

            # Avoid division by zero in case all flux values are the same
            if max_flux != min_flux:
                normalized_flux = (flux - min_flux) / (max_flux - min_flux)
            else:
                normalized_flux = np.zeros(
                    flux.shape
                )  # or another placeholder value, such as np.ones(flux.shape)

            self.data.at[index, "flux"] = normalized_flux
            normalized_fluxes.append(list(normalized_flux))
        return normalized_fluxes

    def get_inferred_continuum(self, sigma=5.0):
        """Returns the inferred continuum by denoising with gaussian convolution

        Args:
            sigma (float, optional): Std. dev of the gaussian kernel. Defaults to 5.

        Returns:
            list: List of denoised fluxes
        """
        # Correction is formula lam_em = lam_obs / (1 + z)
        smoothed_fluxes = []
        for _, row in self.data.iterrows():
            flux = np.array(row["flux"])
            smoothed_fluxes.append(list(gaussian_filter1d(flux, sigma)))

        return smoothed_fluxes
