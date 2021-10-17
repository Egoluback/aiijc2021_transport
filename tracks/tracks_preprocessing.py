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

    def extract_features(self, tracks: pd.DataFrame, y: np.array):
        extracted_features = extract_relevant_features(
            tracks, y, column_id="order_id", column_sort="dt")
        extracted_features = extracted_features
        self.relevant_features = extracted_features.columns
        print("TRACKS FEATURES")
        print(self.relevant_features)
        self.X = extracted_features
        self.y = y

    def _extract_features_unlabeled(self, X_train: pd.DataFrame, relevant_features):
        extracted_features = extract_features(X_train,column_id='order_id', column_sort="dt")
        extracted_features = extracted_features[relevant_features]
        self.X = extracted_features


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

        # if the difference bitween current speed and speed limit is more 20 km/h, it's a violation
        features['is_violation'] = np.where(features.speed-features.maxspeed>25, True, False)

        groupedby_order= features.groupby('order_id',sort=False)
        violation_sum = groupedby_order.apply(lambda x: x.is_violation.sum())
        violations_percentage = groupedby_order.apply(lambda x: x.is_violation.sum()/x.size)

        keys = features['order_id'].unique()
        features = pd.DataFrame({'violations':violation_sum, 'violations_percentage':violations_percentage})
        features.index=keys
        features.to_csv(self.features_path)

        return features

    def preprocess(self, tracks: pd.DataFrame) -> Tuple[pd.DataFrame, np.array, np.array]:
        if exists('./data/with_features_x.pkl'):
            with open('./data/with_features_x.pkl', 'rb') as X_file, open('./data/with_features_y.pkl', 'rb') as y_file, open('./data/relevant_features.pkl', 'rb') as c_file:
                self.X: pd.DataFrame = pickle.load(X_file)
                with open('./log.txt', 'a')as log:
                    log.write('\nTRACK PREPROCESSING\n')
                    log.write(' '.join([str(column) for column in self.X.columns]))
                self.y: pd.DataFrame = pickle.load(y_file)
                self.relevant_features = pickle.load(c_file)
            if exists(self.features_path):
                features = pd.read_csv(self.features_path)
                self.X = self.X.merge(features, left_index=True,
                             how='left', right_index=True)

                with open('./log.txt', 'a')as log:
                    log.write('TRACK PREPROCESSING FEATURES')
                    log.write(' '.join([str(column) for column in self.X.columns]))
            return self.X, self.y

        y = tracks[['order_id', 'is_aggressive']
                   ].drop_duplicates('order_id')
        y = y.set_index('order_id').squeeze()

        features = self.speed_limits_features(tracks, self.features_path)

        X_train = tracks.drop(
            ['Unnamed: 0.1', 'driver_id', 'lat_', 'lon_', 'is_aggressive'], axis=1)
        X_train.loc[0, 'speed'] = 0.0

        self.extract_features(X_train, y)
        self.X.merge(features,left_index = True, right_index = True)

        if exists(self.features_path):
            features = pd.read_csv(self.features_path)
            self.X.merge(features, left_index=True, right_index =True, how='left')

        with open('./data/with_features_x.pkl', 'wb') as file, open('./data/with_features_y.pkl', 'wb') as y_file, open('./data/relevant_features.pkl', 'wb') as c_file:
            pickle.dump(self.X, file)
            pickle.dump(self.y, y_file)
            pickle.dump(self.relevant_features.tolist(),c_file)

        return self.X, self.y
    
    def preprocess_unlabeled(self, tracks: pd.DataFrame) -> pd.DataFrame:
        with open('./data/relevant_features.pkl', 'rb') as c_file:
            relevant_features = np.array(pickle.load(c_file))
        features = self.speed_limits_features(tracks, self.features_path)
        X_train = tracks.drop(
            ['Unnamed: 0.1', 'driver_id', 'lat_', 'lon_'], axis=1)
        X_train.loc[0, 'speed'] = 0.0
        X_train = X_train.dropna()
        self._extract_features_unlabeled(X_train,relevant_features)
        self.X = self.X.merge(features,left_index=True, how='left', right_index=True)

        return self.X

        


if __name__ == '__main__':
    preprocess = Tracks_preprocessing()
    tracks = pd.read_csv('./data/labled_train_tracks_speed.csv')
    x,y= preprocess.preprocess(tracks)
