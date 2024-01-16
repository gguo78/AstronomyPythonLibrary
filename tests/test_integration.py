from src.data_retrieval import GetData
from src.preprocessing import CleanSpectralData
from src.data_augmentation import DataAugmentation 
from src.visualization import SpectralVisualizer
from src.cross_match import CrossMatch_Gaia
from src.classification import SpectralClassifier

import numpy as np


class Test_Integration:
    def test_pipeline(self):
        # GetData
        data_loader = GetData()
        df = data_loader.retrieve_sdss_data(
            sql_query="""
                    SELECT TOP 5 s.fiberid, s.plate, s.mjd, s.run2d, s.class
                    FROM PhotoObj AS p
                    JOIN SpecObj AS s ON s.bestobjid = p.objid
                    """
        )
        
        # CleanSpectralData
        cleaner = CleanSpectralData(dataframe=df)
        cleaner.clean_data()
        cleaner.remove_flux_outliers_iqr()
        cleaner.correct_redshift()
        lam = np.linspace(3100, 7000, 60)
        flux = cleaner.interpolate_flux(lam)
        normalized_flux = cleaner.get_normalize_flux()
        cleaned_data = cleaner.data
        
        # DataAugmentation
        augmentor = DataAugmentation(dataframe=cleaned_data)
        augmentor.process_data()
        augmented_data = augmentor.data 
        
        # Visualization
        visualizer = SpectralVisualizer(dataframe=augmented_data)
        fig = visualizer.plot_spectral_visualization()

        # Cross-match
        cross_matcher = CrossMatch_Gaia() 
        target_ra = 83
        target_dec = -69
        radius = 1
        cross_matcher.match_coords(target_ra, target_dec, radius)
        source_ids = [4657247478248863616,4657247516909872896,4657248032306595584,4657248096725101824]
        cross_matcher.get_astrophysical_params(source_ids) 
        cross_matcher.get_cepheid_star_param(source_ids)
        
        # Classification
        classifier = SpectralClassifier(datapath=None, num_spectra=200, num_wl=200, classifier_layers=[20, 20])
        classifier.train(epochs=100,verbose=True)

        return
