from tsfresh import extract_relevant_features 
from tsfresh import extract_features 
from typing import Tuple
import pandas as pd
import numpy as np
import pickle
from os.path import exists

class Tracks_preprocessing():
    def __init__(self):
        self.X = None
        self.y = None

    def _extract_features(self, tracks:pd.DataFrame, y:np.array):
        extracted_features = extract_relevant_features(tracks, y, column_id="order_id", column_sort="dt")        
        self.X = extracted_features
        self.y = y

    def preprocess(self, tracks: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
        if exists('./with_features_x.pkl'):
            with open('./with_features_x.pkl', 'rb') as X_file, open('./data/with_features_tracks.pkl', 'rb') as y_file:
                self.X = pickle.load(X_file)
                self.y = pickle.load(y_file)
            return self.X, self.y

        y = tracks[['order_id', 'is_aggressive']
                           ].drop_duplicates('order_id')
        y = y.set_index('order_id').squeeze()
        X_train = tracks.drop(
            ['Unnamed: 0.1', 'driver_id', 'lat_', 'lon_', 'is_aggressive'], axis=1)
        X_train.loc[0, 'speed'] = 0.0
        self._extract_features(X_train, y)
        with open('with_features_x.pkl', 'wb') as file, open('./data/with_features_tracks.pkl', 'wb') as y_file:
            pickle.dump(self.X, file)
            pickle.dump(self.y, y_file)

        return self.X, self.y