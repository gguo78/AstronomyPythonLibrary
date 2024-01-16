import pandas as pd
from astroquery.sdss import SDSS
from astropy.io import fits
import requests
from io import BytesIO


class GetData:
    """
    A class to retrieve data from the Sloan Digital Sky Survey (SDSS).
    Attributes:
        data_release (int): The data release version of SDSS to be queried.
    Methods:
        retrieve_identifier_info(sql_query): Retrieves information based on a SQL query.
        format_id(value): Formats an ID value to a standard representation.
        retrieve_sdss_data(sql_query): Retrieves SDSS data and formats it into a DataFrame.
        execute_custom_query: Execute user customized query input.
    """

    def __init__(self, data_release=17):
        self.data_release = data_release

    def retrieve_identifier_info(
        self,
        sql_query="""
                SELECT TOP 20 s.fiberid, s.plate, s.mjd, s.run2d, s.class
                FROM PhotoObj AS p
                JOIN SpecObj AS s ON s.bestobjid = p.objid
                """,
    ):
        try:
            result = SDSS.query_sql(sql_query, data_release=self.data_release)
            info = result.to_pandas()
            info["run2d"] = info["run2d"].str.decode("utf-8")
            info["class"] = info["class"].str.decode("utf-8")
            print("Identifier info retrieved successfully.")
            return info
        except Exception as e:
            raise (
                f"Incorrect input format. Package doesn't support this combination of inputs at this moment"
            )

    def format_id(self, value):
        return f"{value:04d}"

    def retrieve_sdss_data(
        self,
        sql_query="""
                SELECT TOP 20 s.fiberid, s.plate, s.mjd, s.run2d, s.class
                FROM PhotoObj AS p
                JOIN SpecObj AS s ON s.bestobjid = p.objid
                """,
    ):
        sdss_data = self.retrieve_identifier_info(sql_query)
        if sdss_data.empty:
            print("No data retrieved. Exiting.")
            return pd.DataFrame()

        gathered_data = []
        for index, row in sdss_data.iterrows():
            try:
                plateid_formatted = self.format_id(row["plate"])
                fiberid_formatted = self.format_id(row["fiberid"])
                run2d = row["run2d"]

                url = f"http://dr{self.data_release}.sdss.org/sas/dr{self.data_release}/eboss/spectro/redux/{run2d}/spectra/lite/{plateid_formatted}/spec-{plateid_formatted}-{row['mjd']}-{fiberid_formatted}.fits"
                response = requests.get(url)

                if response.status_code == 200:
                    with fits.open(BytesIO(response.content)) as hdul:
                        flux = hdul[1].data["flux"]
                        loglam = hdul[1].data["loglam"]
                        lam = 10**loglam  # 'Wavelength [\AA]'
                        objid = hdul[2].data["OBJID"]
                        ra = hdul[2].data["RA"]
                        dec = hdul[2].data["DEC"]
                        ew = hdul[3].data["lineew"]
                        z = hdul[2].data["Z"]
                        gathered_data.append(
                            {
                                "OBJID": objid[0],
                                "ra": ra[0],
                                "dec": dec[0],
                                "flux": flux.tolist(),
                                "lam": lam.tolist(),
                                "loglam": loglam.tolist(),
                                "ew": ew.tolist(),
                                "redshift": z[0],
                                "class": row["class"],
                            }
                        )
                else:
                    print(
                        f"Failed to download FITS file for URL: {url} - Status code: {response.status_code}"
                    )
            except Exception as e:
                print(f"Error processing row {index}: {e}")

        final_df = pd.DataFrame(gathered_data)
        return final_df

    def execute_custom_query(self, custom_sql_query):
        try:
            result = SDSS.query_sql(custom_sql_query, data_release=self.data_release)
            if result is not None:
                df = result.to_pandas()
                print("Custom query executed successfully.")
                return df
            else:
                print("No results found for the custom query.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error executing custom query: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Just so we have it in the future, lets run the module to load the entire spectral database and save it as a csv file
    # This might be useful for the machine learning module.
    # The module can also be used just for on the fly queries however

    data_retriever = GetData()
    final_df = data_retriever.retrieve_sdss_data(
        sql_query="""
                    SELECT s.fiberid, s.plate, s.mjd, s.run2d, s.class
                    FROM PhotoObj AS p
                    JOIN SpecObj AS s ON s.bestobjid = p.objid
                    """
    )
    final_df.to_csv("./data.csv")
