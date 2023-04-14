from sklearn.base import BaseEstimator, TransformerMixin
import geopandas as gpd
import pandas as pd

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    DISTANCE_CRS = 3857
    CRS = 4326

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs 
        self.add_bedrooms_per_room = add_bedrooms_per_room

        # nearest cities
        california_cities = pd.read_csv('app/utils/cal_cities_lat_long.csv')
        geometry_cal = gpd.points_from_xy(california_cities['Longitude'], california_cities['Latitude'])
        self.points_cal = gpd.GeoDataFrame(california_cities, geometry=geometry_cal)
        self.points_cal.set_crs(self.CRS, inplace=True)


        # coastline
        self.coastline = gpd.read_file('data/coastline/US_Westcoast.shp')[['geometry']]
        self.coastline = self.coastline.to_crs(crs=self.CRS)

        self.points = None
    
    def add_nearest_cities(self):
        nearest_cities = gpd.sjoin_nearest(self.points, self.points_cal, how='left', 
                                        distance_col='distance_nearest_city')
        nearest_cities.drop(columns=['index_right', 'geometry'], inplace=True)
        nearest_cities = nearest_cities.rename(columns={'Name': 'nearest_city'})

        return nearest_cities
    
    def add_ocean_distance(self):
        nearest_coastline = gpd.sjoin_nearest(self.points.to_crs(crs=self.DISTANCE_CRS), self.coastline.to_crs(crs=self.DISTANCE_CRS), distance_col='distance_to_ocean')
        nearest_coastline.to_crs(self.CRS, inplace=True)
        nearest_coastline.drop(columns=['index_right', 'geometry'], inplace=True)

        return nearest_coastline

    def fit(self, X, y=None):
        geometry = gpd.points_from_xy(X['lon'], X['lat'])
        self.points = gpd.GeoDataFrame(X, geometry=geometry)
        self.points.set_crs(self.CRS, inplace=True)

        return self
    def transform(self, X, y=None):
        joined_data = self.add_ocean_distance()
        joined_data = self.add_nearest_cities()


        X = joined_data.drop(columns=['geometry','lon','lat','Latitude','Longitude'])

        # removing generated duplicated locations (when there are more than one nearest city)
        X = X[~X.index.duplicated(keep='first')]  

        X['rooms_per_household'] = X.total_rooms / X.households
        #X['population_per_household'] = X.population / X.households
        if self.add_bedrooms_per_room:
            X['bedrooms_per_room'] = X.total_bedrooms / X.total_rooms
            
        return X