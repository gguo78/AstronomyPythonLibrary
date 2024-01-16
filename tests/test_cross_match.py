import pytest
import pandas as pd
from src.cross_match import CrossMatch_Gaia


class Test_CrossMatch_Gaia:
    @pytest.fixture(autouse=True)
    def initialize(self):
        self.module = CrossMatch_Gaia()

    def test_match_coords(self):
        target_ra = 83
        target_dec = -69
        radius = 1
        matches_df, source_ids = self.module.match_coords(target_ra, target_dec, radius)

        assert isinstance(source_ids, list), "Should return list of source ids."
        assert isinstance(
            matches_df, pd.DataFrame
        ), "Should return pandas dataframe of discovered ra and dec"
        assert "ra" in matches_df.columns, "Column 'ra' not found in the DataFrame"
        assert "dec" in matches_df.columns, "Column 'dec' not found in the DataFrame"

    def test_get_astrophysical_params(self):
        source_ids = [
            4657247478248863616,
            4657247516909872896,
            4657248032306595584,
            4657248096725101824,
            4657248238465061760,
            4657248818211810944,
            4657248822580494976,
            4657262944436063744,
        ]
        df_res = self.module.get_astrophysical_params(source_ids)
        assert not df_res.empty, "The output DataFrame is empty"
        assert len(df_res) == len(
            source_ids
        ), "The number of rows in the DataFrame does not match the length of source_ids"

    def test_get_cepheid_star_param(self):
        source_ids = [
            4657247478248863616,
            4657247516909872896,
            4657248032306595584,
            4657248096725101824,
            4657248238465061760,
            4657248818211810944,
            4657248822580494976,
            4657262944436063744,
        ]
        df_res = self.module.get_cepheid_star_param(source_ids)
        assert not df_res.empty, "The output DataFrame is empty"
        assert len(df_res) == len(
            source_ids
        ), "The number of rows in the DataFrame does not match the length of source_ids"

        # Assert that specific columns are in the DataFrame
        expected_columns = ["source_id", "pf", "teff_gspphot", "distance_gspphot"]
        for col in expected_columns:
            assert col in df_res.columns, f"Column '{col}' is missing in the DataFrame"
