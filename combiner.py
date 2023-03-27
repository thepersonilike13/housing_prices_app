from sklearn.base import BaseEstimator, TransformerMixin
import geopandas as gpd
import pandas as pd

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do 
    def transform(self, X, y=None):
        geometry_housing = gpd.points_from_xy(X['lon'], X['lat'])
        points_housing = gpd.GeoDataFrame(X, geometry=geometry_housing)

        california_cities = pd.read_csv('cal_cities_lat_long.csv')
        geometry_cal = gpd.points_from_xy(california_cities['Longitude'], california_cities['Latitude'])
        points_cal = gpd.GeoDataFrame(california_cities, geometry=geometry_cal)

        joined_data = gpd.sjoin_nearest(points_housing, points_cal, how='left', distance_col='distance_nearest_city')
        joined_data = joined_data.rename(columns={'Name': 'nearest_city'})

        X = joined_data.drop(columns=['lon','lat','geometry', 'index_right','Latitude','Longitude'])    

        X = X[~X.index.duplicated(keep='first')]
        

        X['rooms_per_household'] = X.total_rooms / X.households
        X['population_per_household'] = X.population / X.households

        if self.add_bedrooms_per_room:
            X['bedrooms_per_room'] = X.total_bedrooms / X.total_rooms
            
        return X
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False) 
#housing_extra_attribs = attr_adder.transform(housing.values)