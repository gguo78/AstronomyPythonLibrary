# 207 Project README.md
### by Skywalker

## Module 1: data_retrieval

### Class 1: GetData

#### Purpose
To retrieve data from the Sloan Digital Sky Survey (SDSS).

#### Attribute:
- data_release (int): The data release version of SDSS to be queried.

#### Functions
1. **retrieve_identifier_info**
   - Purpose: To use SQL query given by user (or the default) to retrieve fiberid, plate, mjd, run2d
2. **format_id**
   - Purpose: To zero-pad plate and fiberid to fit the format of retrieval link
3. **retrieve_sdss_data**
   - Purpose: To retrieve SDSS data and format it into a DataFrame.
4. **execute_custom_query**
   - Purpose: To execute user customized query input.


## Module 2: preprocessing

### Class 1: CleanSpectralData

#### Purpose
To clean and scale the data retrieved from the data_retrieval module.

#### Attribute: 
- data: DataFrame containing the input data.  

#### Functions
1. **clean_data**
   - Purpose: To clean the stored (and returned) dataframe. Flux and lambda is changed from str to list of floats, NaN are removed, and the dataframe is checked for duplicate columns,rows.
2. **interpolate_flux**
   - Purpose: To reinterpolate the flux data to align with the input wavelength vector. Internal dataframe is updated and interpolated fluxes are also returned.
3. **align_wavelengths**
   - Purpose: To align the fluxes onto the same linearly spaced wavelength grid.
4. **remove_flux_outliers_iqr**
   - Purpose: To remove outliers in the stored dataframe (same as returned) according to IQR criteria. Outliers are replaced with interpolated values.
5. **correct_redshift**
   - Purpose: To correct the redshift values using the formula lam_em = lam_obs / (1 + z). The function goes through each lam-flux entry and apply a correction to the axis wavelength.
6. **get_normalize_flux**
   - Purpose: To normalize all flux data within the range [0, 1].
6. **get_inferred_continuum**
   - Purpose: To return the inferred continuum by denoising with gaussian convolution.


---

## Module 3: data_augmentation 

### Class 1: DataAugmentation

#### Purpose
To compute derivatives as well as fractional derivatives and append them to each preprocessed spectra.

#### Attribute:
- data: DataFrame containing the input data. 

#### Functions
1. **fractional_derivative**
   - Purpose: To compute the fractional derivative of the flux with respect to the log of the wavelength.
2. **process_data** 
   - Purpose: To process the data and add a new column with the fractional derivative.

---

## Module 4: visualization

### Class 1: SpectralVisualizer

#### Purpose
To provide spectral visualization with an overlay of the inferred continuum.

#### Attribute: 
- data: DataFrame containing the input data.  
- wavelength_column (str): Column name for Wavelength data. Defaults to 'lam'.
- flux_column (str): Column name for flux data. Defaults to 'flux'.

#### Functions
1. **_prepare_data**
   - Purpose: To prepare data for visualization. Internal method.
2. **_calculate_average_flux**
   - Purpose: To calculate the average flux from the interpolated flux values. Internal method.
3. **_denoise_flux**
   - Purpose: To apply Savitzky-Golay filter for denoising the average flux. Internal method.
4. **plot_spectral_visualization**
   - Purpose: Plot the original and denoised flux for spectral visualization.

---

## Module 5: classification

### Class 1: SpectralClassifier

#### Purpose
Queries the SDSS spectral data to use as a training set and initializes an untrained MLP classifier.

#### Attribute: 
- lam: cleaned wavelength (lam) values.
- fluxes: cleaned and stacked flux values.
- label_to_int: dictionary mapping the classes to distinguish (Stars, Galaxies, and QSOs) to int.
- y_int: torch tensor object for the int class labels.
- train_accuracies: stores the train accuracies of the model.
- test_accuracies: stores the test accuracies of the model.
- model: a SimpleClassifier model.
- trained: denotes the state of training. Defaults to False.

#### Functions
1. **train**
   - Purpose: To train the feed-forward network classifier.
2. **plot_train_accuracy**
   - Purpose: To create a plot of the models training accuracy
3. **transform_predict**
   - Purpose: To transform the input and predict the object class.

### Class 2: SimpleClassifier(nn.Module)

#### Purpose
To create a neural network for classification tasks using PyTorch.

#### Attribute: 
- layers (nn.ModuleList): A container for holding the linear layers of the neural network.

#### Function:
1. **forward**
   - Purpose: To define the forward pass of the neural network. 
