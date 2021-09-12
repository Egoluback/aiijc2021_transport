from tsfresh import extract_relevant_features
from tsfresh import extract_features
from typing import Tuple
import pandas as pd
import numpy as np
import pickle
from os.path import exists


class Tracks_preprocessing():
    def __init__(self, features_path='./data/speed_limits_features.csv'):
        self.X: pd.DataFrame = None
        self.y: pd.DataFrame = None
        self.relevant_features = None
        self.features_path: str = features_path
        self.road_info_path = './data/road_info.csv'

    def _extract_features(self, tracks: pd.DataFrame, y: np.array):
        extracted_features = extract_relevant_features(
            tracks, y, column_id="order_id", column_sort="dt")
        self.relevant_features=extracted_features.columns
        self.X = extracted_features
        self.y = y

    def speed_limits_features(self,data, features_path='./data/speed_limits_features.csv'):
        tracks = data.copy()
        if exists(features_path):
            features = pd.read_csv(features_path)
            return features

        road_info = pd.read_csv(self.road_info_path)

        road_info['lat'] = road_info['loc'].apply(
            lambda x: round(eval(x)[0], 3))
        road_info['lon'] = road_info['loc'].apply(
            lambda x: round(eval(x)[1], 3))
        tracks['lat'] = tracks['lat_'].apply(
            lambda x: round(x, 3))
        tracks['lon'] = tracks['lon_'].apply(
            lambda x: round(x, 3))

        df = road_info.merge(tracks, on=['lat', 'lon'], how='right')

        dic = {'ru:urban': 60, 'ru:living_street': 20, 'ru:rural': 90, 'ru:motorway':120}
        mxspeed = df.maxspeed.apply(lambda x: int(x) if type(x)==str and x.isdigit() else x)
        df.maxspeed =  mxspeed.apply(lambda x: dic[x.lower()] if type(x)==str else x)
        features=df.copy()
        features['is_violation'] = np.where(features.speed>features.maxspeed, True, False)
        features['near_violation'] = np.where((features.speed-features.maxspeed)<0.5, True, False)

        groupedby_order= features.groupby('order_id')

        count_violations = groupedby_order.apply(lambda x: x.is_violation.sum())
        count_near_violations = groupedby_order.apply(lambda x: x.near_violation.sum())

        features = pd.DataFrame({'violations':count_violations, 'near_violations': count_near_violations})
        return features

    def preprocess(self, tracks: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
        if exists('./data/with_features_x.pkl'):
            with open('./data/with_features_x.pkl', 'rb') as X_file, open('./data/with_features_y.pkl', 'rb') as y_file:
                self.X: pd.DataFrame = pickle.load(X_file)
                self.y: pd.DataFrame = pickle.load(y_file)

            if exists(self.features_path):
                features = pd.read_csv(self.features_path)
                self.X.merge(features, left_index=True,
                             how='left', right_index=True)
                print(self.X.head())
            return self.X, self.y

        y = tracks[['order_id', 'is_aggressive']
                   ].drop_duplicates('order_id')
        y = y.set_index('order_id').squeeze()

        features = self.speed_limits_features(tracks, self.features_path)

        X_train = tracks.drop(
            ['Unnamed: 0.1', 'driver_id', 'lat_', 'lon_', 'is_aggressive'], axis=1)
        X_train.loc[0, 'speed'] = 0.0

        self._extract_features(X_train, y)
        self.X.merge(features,left_index=True, how='left', right_index=True)

        if exists(self.features_path):
            features = pd.read_csv(self.features_path)
            self.X.merge(features, left_index=True, how='left')
            print(self.X.head())

        with open('./data/with_features_x.pkl', 'wb') as file, open('./data/with_features_y.pkl', 'wb') as y_file:
            pickle.dump(self.X, file)
            pickle.dump(self.y, y_file)

        return self.X, self.y


if __name__ == '__main__':
    preprocess = Tracks_preprocessing()
