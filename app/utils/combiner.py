from sklearn.base import BaseEstimator, TransformerMixin
import geopandas as gpd
import pandas as pd

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs 
        self.add_bedrooms_per_room = add_bedrooms_per_room
        california_cities = pd.read_csv('utils/cal_cities_lat_long.csv')
        geometry_cal = gpd.points_from_xy(california_cities['Longitude'], california_cities['Latitude'])
        self.points_cal = gpd.GeoDataFrame(california_cities, geometry=geometry_cal)
    
    def add_nearest_cities(self, data):
        geometry = gpd.points_from_xy(data['lon'], data['lat'])
        points = gpd.GeoDataFrame(data, geometry=geometry)
        
        joined_data = gpd.sjoin_nearest(points, self.points_cal, how='left', 
                                        distance_col='distance_nearest_city')
        joined_data = joined_data.rename(columns={'Name': 'nearest_city'})

        return joined_data
    def add_ocean_distance(self, data_points):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        joined_data = self.add_nearest_cities(X)

        X = joined_data.drop(columns=['lon','lat','geometry', 'index_right','Latitude','Longitude'])

        # removing generated duplicated locations (when there are more than one nearest city)
        X = X[~X.index.duplicated(keep='first')]  

        X['rooms_per_household'] = X.total_rooms / X.households
        #X['population_per_household'] = X.population / X.households
        if self.add_bedrooms_per_room:
            X['bedrooms_per_room'] = X.total_bedrooms / X.total_rooms
            
        return X