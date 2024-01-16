import pytest
import os
import pandas as pd 
import numpy as np
import ast
from src.data_augmentation import DataAugmentation 

class TestAugmentation:

    def test_data_augmentation(self):
        np.random.seed(42)
        num_samples = 5
        flux_values = [np.random.rand(10).tolist() for _ in range(num_samples)]
        loglam_values = [np.random.rand(10).tolist() for _ in range(num_samples)]

        sample_data = pd.DataFrame({
            "flux": flux_values,
            "lam": loglam_values
        })

        a = DataAugmentation(dataframe=sample_data)
        a.process_data()
        expected_result = pd.DataFrame({"frac_dev":[[-100.542, 7.885, 38.591, -27.138, 1.45, 13.666, -2.575, 7.136, -24.224, -4.874]]})
        list_1 = a.data['frac_dev'][0]
        list_2 = expected_result['frac_dev'][0]
        rounded_list_1 = [round(x, 3) for x in list_1]
        rounded_list_2 = [round(x, 3) for x in list_2]

        assert rounded_list_1 == rounded_list_2

if __name__ == "__main__":
    pytest.main()