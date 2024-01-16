import pytest
from src.classification import SpectralClassifier
from src.data_retrieval import GetData


class Test_SpectralClassifier:
    @pytest.fixture(autouse=True)
    def initialize(self):
        self.module = SpectralClassifier(num_spectra=5, num_wl=50)

    def test_train(self):
        # We don't want to unit test neural network performance since that can change with models etc
        # But let's assert that we have accuracies saved and reported
        self.module.train(epochs=10)
        train_accuracies = self.module.train_accuracies
        test_accuracies = self.module.test_accuracies
        assert (
            len(train_accuracies) == len(test_accuracies) == 10
        ), "Accuracies not correctly reported"
        assert all(
            0 <= num <= 1 for num in train_accuracies
        ), "Not all values are between 0 and 1"
        assert all(
            0 <= num <= 1 for num in test_accuracies
        ), "Not all values are between 0 and 1"

    def test_plot_train_accuracy(self):
        # Just verify that there is some plot function in the code
        # and that it runs without error
        self.module.plot_train_accuracy()

    def test_transform_predict(self):
        # generally speaking, one shouldn't write unit tests enforcing neural networks to pass
        # but we do want to ensure outputs are returned without error
        module = SpectralClassifier(num_spectra=5, num_wl=50)
        module.train(epochs=10)

        data_retriever = GetData()
        load_num_spec = 5
        df = data_retriever.retrieve_sdss_data(
            sql_query=f"""
                SELECT TOP {load_num_spec} s.fiberid, s.plate, s.mjd, s.run2d, s.class
                FROM PhotoObj AS p
                JOIN SpecObj AS s ON s.bestobjid = p.objid
                """
        )

        lam = df["lam"].values
        flux = df["flux"].values
        prob, predicted_label = module.transform_predict(lam, flux)
        assert prob.shape[0] == len(predicted_label), "Expected same number of rows"

        return
