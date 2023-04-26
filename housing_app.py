import streamlit as st
st.set_page_config(page_title="Housing Prices Prediction", page_icon=":house:")
import pandas as pd
import numpy as np
import pickle
import folium
from geopy.geocoders import Nominatim
import geopy.distance
from streamlit_folium import st_folium
from utils.combiner import CombinedAttributesAdder

## -----------------------------------------------------------------------------------------##
## Functions

def _max_width_(prcnt_width:int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .appview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

rand_addresses = ['1219 Carleton Street, Berkeley CA 94702','24147 Clinton Court, Hayward CA 94545',
'560 Penstock Drive, Grass Valley CA 95945','1238 Roanwood Way, Concord CA 94521',
'2807 Huxley Place, Fremont CA 94555','441 Merritt Avenue, Oakland CA 94610',
'3377 Sandstone Court, Pleasanton CA 94588', '2443 Sierra Nevada Road, Mammoth Lakes CA 93546']
def get_rand_addr(addresses):
    return np.random.choice(addresses, replace=False)

def initialize_session_states():
    if 'markers' not in st.session_state:
        st.session_state['markers'] = []
    if 'lines' not in st.session_state:
        st.session_state['lines'] = []
    if 'fg' not in st.session_state:
        st.session_state['fg'] = None
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None
    if 'address' not in st.session_state:
        st.session_state['address'] = None
    if 'address_output' not in st.session_state:
        st.session_state['address_output'] = ""
    if 'location' not in st.session_state:
        st.session_state['location'] = None
    
    if 'random_values' not in st.session_state:
        random_rooms = np.random.randint(40, 600)
        random_bedrooms = np.random.randint(30, random_rooms)
        random_households = np.random.randint(0, 60) + random_bedrooms -  np.random.randint(0, 20)
        random_addr = get_rand_addr(rand_addresses)

        st.session_state['random_values'] = dict(
            random_rooms = random_rooms,
            random_bedrooms = random_bedrooms,
            random_households = random_households,
            random_address = random_addr
        )

# cached resources
@st.cache_resource
def initialize_nominatim(user_agent=f'housing_price_app_{np.random.randint(0,200)}'):
    with st.spinner('Initializing geolocator...'):
        return Nominatim(user_agent=user_agent)

@st.cache_resource
def load_model(filepath: str):
    with st.spinner('Loading model...'):
        return pickle.load(open(filepath, 'rb'))

@st.cache_resource
def load_combiner():
    with st.spinner('Loading components..'):
        return CombinedAttributesAdder()

geolocator = initialize_nominatim()
loaded_model = load_model('model/linear_reg_model.pkl')
combiner = load_combiner()

def get_location(address: str):
    return geolocator.geocode(address, addressdetails=True)

def transform_data(data: pd.DataFrame):
    return combiner.add_nearest_cities(data)

def get_nearest_city(location):
    lon, lat = location.longitude, location.latitude

    data = pd.DataFrame(dict(lon=lon, lat=lat), index=[0])
    transformed = transform_data(data)
    nearest_city = transformed['nearest_city'].values.squeeze()
    return nearest_city

def create_marker(m: folium.Map, location, icon_color='red', **kwargs):
    coords = [location.latitude, location.longitude]
    marker = folium.Marker(
        location=coords,  
        icon=folium.Icon(color=icon_color),
        **kwargs)

    return marker

def get_markers_addresses():
    return list(map(lambda marker: marker['address'], st.session_state['markers']))

def link_two_markers(marker1, marker2, **kwargs):
    return folium.PolyLine(locations=(marker1.location, marker2.location), **kwargs)

def clear_markers():
    st.session_state['markers'] = []
    st.session_state['lines'] = []

    return folium.FeatureGroup('objects')

def create_map():
    # Define the boundaries of California
    min_lat, min_lon = 32.5295, -124.4820
    max_lat, max_lon = 42.0095, -114.1315

    # Create a map centered on California
    map_ca = folium.Map(location=[37.7749, -122.4194], zoom_start=6, min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon, no_wrap=True, max_bounds=True)
    return map_ca

## -------------------------------------------------------------------------------------------------------##
## Webpage

_max_width_(70)
st.title("California Housing Prices Prediction")
st.markdown("""
##### A web application for predicting California Housing Prices.
 
This app uses machine learning to predict the price of the house. 
It loads a pre-trained linear regression model, which takes as input various features of the house, 
such as the number of rooms, the number of bedrooms, the population of the house's neighborhood, and the distance to the nearest city. 
The app preprocesses the input data by combining some of the features and adding new features, such as the distance to the nearest city.
""")
st.markdown("**:book: [GitHub repository](https://github.com/matheuscamposmt/housing_prices_app)** | :heart: **My profile:** [@matheuscamposmt](https://github.com/matheuscamposmt)")

data = pd.read_csv('data/housing.csv')
max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

map_ca = create_map()
initialize_session_states()
st.session_state['fg'] =  folium.FeatureGroup(name="objects", control=True)

rand_vals = st.session_state['random_values']

# layout and input data
col1, col2 = st.columns([1, 2], gap='large')
with col1:
    st.header("Enter the attributes of the housing.")
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        housing_median_age = np.nan

        total_rooms = st.number_input(
            "Total Rooms within a block",
            value=rand_vals['random_rooms'],
            min_value=int(min_values['total_rooms']),
            max_value=int(max_values['total_rooms']), 
            step=5)
        total_bedrooms = st.number_input(
            "Total Bedrooms within a block",
            value=rand_vals['random_bedrooms'], 
            min_value=int(min_values['total_bedrooms']), 
            max_value=int(max_values['total_bedrooms']), 
            step=5)
        
        ocean_proximity = st.selectbox(
        'Ocean Proximity:',
        ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))
        population = np.nan

    with subcol2:
        households = st.number_input(
            "Households for a block",
            value=rand_vals['random_households'], 
            min_value=int(min_values['households']), 
            max_value=int(max_values['households']), step=5)
        
        median_income = st.slider(
            "Median income within a block (in thousands of U.S. dollars)",
            value=2.0,
            min_value=float(min_values['median_income']), 
            max_value=float(max_values['median_income']), step=0.5)
        
        rand_addr_button = st.button("Random address", help='Generate a random address')
        
        if rand_addr_button:
            rand_vals['random_address'] = get_rand_addr(rand_addresses)
            st.session_state['address_output'] = ""
        
    address = st.text_input("Address", value=rand_vals['random_address'])

    st.caption("Press the button below to mark the address in the map.")
    locate_button = st.button("Locate")

    if address and locate_button and (not address in get_markers_addresses()):
        st.session_state['address'] = address

        location = get_location(address)
        st.session_state['location'] = location

        if location:
            state = location.raw['address']['state']

            if state == 'California':
                housing_coords = (location.latitude, location.longitude)
                housing_marker = create_marker(map_ca, location, popup=location)
                
                nearest_city = get_nearest_city(location)
                nearest_city_loc = get_location(nearest_city+", CA")
                nearest_city_coords = (nearest_city_loc.latitude, nearest_city_loc.longitude)

                distance_km = geopy.distance.distance(nearest_city_coords, housing_coords).km

                st.session_state['address_output'] = f'Nearest City: {nearest_city} | Distance {distance_km:.2f} km'
                nearest_city_marker = create_marker(
                    map_ca, nearest_city_loc, 
                    icon_color='green', popup=nearest_city)
                
                line_markers = link_two_markers(housing_marker, nearest_city_marker, tooltip=f'Distance {distance_km:.2f} km')
                
                st.session_state['markers'].append({'marker':housing_marker, 'address': address})
                st.session_state['markers'].append({'marker':nearest_city_marker, 'address': "n_city_"+address})
                st.session_state['lines'].append(line_markers)


            else:
                st.warning(
                    "The location should be in California. \
                    Inputting a place that doesn't belong to the state of California will lead to inconsistent results.")
        
        else:
            st.error("Address not found. Try again with another one.")
    st.write(st.session_state['address_output'])
    button = st.button("Predict", use_container_width=True)

    
    if button:
        if not address in get_markers_addresses():
            st.error("You didn't locate the address. Press the 'Locate' button to pinpoint it on the map.")
        elif total_bedrooms > total_rooms:
            st.error('Error: Total bedrooms cannot be bigger than total rooms.')
        
        else:
            location = st.session_state['location']
            input_data = {
            "lon": location.longitude,
            "lat": location.latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity
            }
            
            input_df = pd.DataFrame([input_data])

            prediction = loaded_model.predict(input_df).squeeze()
            st.session_state['prediction'] = prediction
            st.success("Done!")

    
    if st.session_state['prediction']:
        pred = st.session_state['prediction']
        st.markdown(
            """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 34px;
            color: green;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.metric(label='Median House Value', value=f"$ {pred:.2f}")

with col2:
    for marker_content in st.session_state["markers"]:
        st.session_state['fg'].add_child(marker_content['marker'])

    for line in st.session_state["lines"]:
        st.session_state['fg'].add_child(line)


    clean_button = st.button("Clear markers")
    if clean_button:
        st.session_state['fg'] = clear_markers()

    st_data = st_folium(map_ca, width=1200, height=800, feature_group_to_add=st.session_state['fg'])

    

