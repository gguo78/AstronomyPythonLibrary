import pandas as pd
from pandas._testing import assert_frame_equal
from io import BytesIO
from src.data_retrieval import GetData
import requests
from astropy.io import fits
import pytest


class TestGetData:
    @classmethod
    def setup_class(cls):
        cls.data_retriever = GetData(data_release=17)

    def download_fake_fits(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            print(
                f"Failed to download FITS file from URL: {url} - Status code: {response.status_code}"
            )
            return None

    def test_retrieve_sdss_data(self):
        real_url = "http://dr17.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/6413/spec-6413-56336-0495.fits"
        result = self.data_retriever.retrieve_sdss_data(
            sql_query="""SELECT TOP 2 s.fiberid, s.plate, s.mjd, s.run2d, s.class FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid"""
        )
        result = result.drop(
            "lam", axis=1
        )  # remove lam because it is added after based on loaded loglam

        # This should be edited but for now, the real data download matches with second item of Top 2
        # This should not be assumed true as the foreign database we call can change anytime
        result = result.iloc[0:1]

        with fits.open(self.download_fake_fits(real_url)) as real_hdul:
            real_data = {
                "OBJID": real_hdul[2].data["OBJID"][0],
                "ra": real_hdul[2].data["RA"][0],
                "dec": real_hdul[2].data["DEC"][0],
                "flux": real_hdul[1].data["flux"].tolist(),
                "loglam": real_hdul[1].data["loglam"].tolist(),
                "ew": real_hdul[3].data["lineew"].tolist(),
                "redshift": real_hdul[2].data["Z"][0],
                "class": "GALAXY",
            }
        expected_result = pd.DataFrame([real_data])
        assert_frame_equal(result, expected_result)

    def test_retrieve_identifier_info(self):
        invalid_sql_query = """SELECT TOP 1 s.abc, s.plate, s.mjd, s.run2d, s.class FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid"""
        with pytest.raises(Exception):
            self.data_retriever.retrieve_identifier_info(sql_query=invalid_sql_query)

    def test_format_id(self):
        gd = GetData()
        assert gd.format_id(6) == "0006"
        assert gd.format_id(100) == "0100"
        assert gd.format_id(1000) == "1000"
