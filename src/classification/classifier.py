import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F

from .nn import SimpleClassifier
from src.data_retrieval import GetData
from src.preprocessing import CleanSpectralData


class SpectralClassifier:
    def __init__(
        self, datapath=None, num_spectra=500, num_wl=500, classifier_layers=[32, 32]
    ):
        """Create a Spectral Classifier class. Queries the SDSS spectral data to use as a training set and initializes an untrained MLP classifier.

        Args:
            datapath (str, optional): Path to presaved dataframe from GetData. Defaults to None.
            num_spectra (int, optional): Otherwise, the number of spectra to query from SDSS. Defaults to 500.
            num_wl (int, optional): Number of wavelengths to use as features for classification. Defaults to 500.
            classifier_layers (list, optional): Number of neurons in each hidden layer. Defaults to [32, 32].
        """
        # Get the training data loader and the cleaner
        data_retriever = GetData()
        if datapath is not None:
            try:
                clean_module = CleanSpectralData(datapath=datapath)
            except Exception as e:
                print(f"An error occurred loading from datapath: {e}")
        else:
            df = data_retriever.retrieve_sdss_data(
                sql_query=f"""
                        SELECT TOP {num_spectra} s.fiberid, s.plate, s.mjd, s.run2d, s.class
                        FROM PhotoObj AS p
                        JOIN SpecObj AS s ON s.bestobjid = p.objid
                        """
            )
            clean_module = CleanSpectralData(dataframe=df)

        # Preprocess the training data
        _ = clean_module.remove_flux_outliers_iqr()
        self.lam, _ = clean_module.align_wavelengths(num_wl=num_wl)
        clean_module.get_normalize_flux(update_df=True)
        fluxes = clean_module.get_inferred_continuum()

        # Define training features and labels
        self.fluxes = np.stack(fluxes)
        labels = clean_module.data["class"].values
        self.label_to_int = {"GALAXY": 0, "QSO": 1, "STAR": 2}
        self.y_int = torch.tensor([self.label_to_int[label] for label in labels])

        # Hold fit info
        self.train_accuracies = []
        self.test_accuracies = []
        num_features = self.fluxes.shape[1]
        num_classes = len(self.label_to_int)
        self.model = SimpleClassifier(
            num_features, num_classes, layers=classifier_layers
        )
        self.trained = False

    def train(
        self,
        train_test_split=0.8,
        learning_rate=1e-3,
        epochs=500,
        verbose=False,
    ):
        """Call to train the feed-forward network classifier.

        Args:
            train_test_split (float, optional): Percent of train-test data split. Defaults to 0.8.
            learning_rate (float, optional): Learning rate for Adam. Defaults to 1e-3.
            epochs (int, optional): Number of training epochs. Defaults to 500.
            verbose (bool, optional): If true, print epoch losses. Defaults to False.
        """
        x = torch.tensor(self.fluxes, dtype=torch.float32)
        y = torch.tensor(self.y_int, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        dataset = TensorDataset(x, y)
        train_size = int(train_test_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create DataLoaders for training and testing
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Function to compute accuracy
        def get_accuracy(loader, model):
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / total

        # Train
        train_accuracies = []
        test_accuracies = []
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            self.model.eval()
            train_accuracy = get_accuracy(train_loader, self.model)
            test_accuracy = get_accuracy(test_loader, self.model)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            if verbose:
                print(
                    f"Epoch {epoch}, Loss: {loss.item()}, Train Acc: {train_accuracy*100:.2f}%, Test Acc: {test_accuracy*100:.2f}%"
                )
        self.train_accuracies = train_accuracies
        self.test_accuracies = test_accuracies

        self.trained = True
        return

    def plot_train_accuracy(self):
        """Create a plot of the models training accuracy"""
        # Plotting the accuracies
        plt.figure(figsize=(10, 6))
        num_epochs = len(self.train_accuracies)
        plt.plot(range(num_epochs), self.train_accuracies, label="Train Accuracy")
        plt.plot(range(num_epochs), self.test_accuracies, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Testing Accuracy vs Epoch")
        plt.legend()
        return

    def transform_predict(self, lam, flux):
        """Transform the input and predict the object class.

        Args:
            lam (np.array or list): array or list of lists containing wavelength vectors
            flux (np.array or list): array or list of lists containing fluxes

        Returns:
            list: class probabilities from the model and predicted labels.
        """
        if not self.trained:
            raise AssertionError("Train must be called first before model can predict.")

        if isinstance(lam, np.ndarray) and isinstance(flux, np.ndarray):
            lam = lam.tolist()
            flux = flux.tolist()

        assert all(
            isinstance(sublist, list) for sublist in lam
        ), "Lam should be a list of lists"
        assert all(
            isinstance(sublist, list) for sublist in flux
        ), "Flux should be a list of lists"
        assert len(lam) == len(flux), "lam and flux must have the same number of rows"
        assert all(
            len(sublist) == len(flux[i]) for i, sublist in enumerate(lam)
        ), "All rows in lam and flux must have the same length"
        loglam_lists = [list(map(np.log10, sublist)) for sublist in lam]

        df = pd.DataFrame({"lam": lam, "loglam": loglam_lists, "flux": flux})

        cleaner = CleanSpectralData(dataframe=df)
        cleaner.remove_flux_outliers_iqr()
        cleaner.interpolate_flux(self.lam)
        cleaner.get_normalize_flux(update_df=True)
        fluxes = cleaner.get_inferred_continuum()

        # predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(fluxes, dtype=torch.float32))
            probabilities = F.softmax(logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).tolist()

            int_to_label = {v: k for k, v in self.label_to_int.items()}
            # Convert the list of predicted class indices to labels
            predicted_labels = [int_to_label[idx] for idx in predicted_classes]

        return probabilities, predicted_labels
