import sqlite3
import pandas as pd

conn = sqlite3.connect('FPA_FOD_20170508.sqlite')

# Create and upload mapping table to SQlite database
from scipy.spatial import KDTree
city_data = pd.read_csv('uscities.csv')
city_tree = KDTree(city_data[['lat', 'lng']].values)
fire_data = pd.read_sql('SELECT DISTINCT latitude, longitude FROM Fires', conn)


def closest_city_and_county(lat, lon):
    distance, index = city_tree.query([lat, lon])
    city = city_data.loc[index, 'city']
    county = city_data.loc[index, 'county_name']
    state = city_data.loc[index, 'state_name']
    return city, county, state

fire_data['nearest_city'], fire_data['county_name'], fire_data['state_name'] = zip(*fire_data.apply(lambda x: closest_city_and_county(x['LATITUDE'], x['LONGITUDE']), axis=1))
fire_data.to_sql('LatLon_City_Mapping', conn, if_exists='replace', index=False)
