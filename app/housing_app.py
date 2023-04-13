import streamlit as st
st.set_page_config(page_title="Housing Prices Prediction", page_icon=":house:")
import pandas as pd
import numpy as np
import pickle
from utils.combiner import CombinedAttributesAdder
import folium
from geopy.geocoders import Nominatim, GeoNames
from streamlit_folium import st_folium
from utils.combiner import CombinedAttributesAdder

def _max_width_(prcnt_width:int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .appview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

def initialize_session_states():
    if 'markers' not in st.session_state:
        st.session_state['markers'] = []
    
    if 'lines' not in st.session_state:
        st.session_state['lines'] = []

    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None

    if 'address' not in st.session_state:
        st.session_state['address'] = None

    if 'address_output' not in st.session_state:
        st.session_state['address_output'] = ""

    if 'location' not in st.session_state:
        st.session_state['location'] = None

@st.cache_resource
def initialize_nominatim():
    return Nominatim(user_agent="geolocator_resource")

@st.cache_resource
def load_model(filepath: str):
    return pickle.load(open(filepath, 'rb'))

@st.cache_resource
def load_combiner():
    return CombinedAttributesAdder()

# cached resources
geolocator = initialize_nominatim()
loaded_model = load_model('app/model/linear_reg_model.pkl')
combiner = load_combiner()

def get_location(address: str):
    return geolocator.geocode(address, addressdetails=True)

def transform_data(data: pd.DataFrame):
    return combiner.add_nearest_cities(data)

def get_nearest_city(location):
    lon, lat = location.longitude, location.latitude

    data = pd.DataFrame(dict(lon=lon, lat=lat), index=[0])
    nearest_city = transform_data(data)['nearest_city'].values.squeeze()
    return nearest_city

def create_marker(m: folium.Map, location, icon_color='red', **kwargs):
    coords = [location.latitude, location.longitude]
    marker = folium.Marker(
        location=coords,  
        icon=folium.Icon(color=icon_color),
        **kwargs)

    return marker

def link_two_markers(m: folium.Map, marker1, marker2):
    line = folium.PolyLine(locations=(marker1.location, marker2.location))
    
    return line


def clear_markers():
    st.session_state['markers'] = []
    st.session_state['lines'] = []


def create_map():
    # Define the boundaries of California
    min_lat, min_lon = 32.5295, -124.4820
    max_lat, max_lon = 42.0095, -114.1315

    # Create a map centered on California
    map_ca = folium.Map(location=[37.7749, -122.4194], zoom_start=6, min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon, no_wrap=True, max_bounds=True)

    return map_ca

_max_width_(70)
st.title("California Housing Prices Prediction")
st.markdown("[GitHub repository](https://github.com/matheuscamposmt/housing_prices_app): @matheuscamposmt")

data = pd.read_csv('data/housing.csv')
max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

map_ca = create_map()
fg = folium.FeatureGroup(name="objects")

initialize_session_states()

# layout and input data
col1, col2 = st.columns([1, 2], gap='large')
with col1:
    st.header("Enter the attributes of the housing.")
    subcol1, subcol2 = st.columns(2)

    with subcol1:
        housing_median_age = np.nan
        total_rooms = st.number_input(
            "Total Rooms", 
            min_value=int(min_values['total_rooms']), 
            max_value=int(max_values['total_rooms']), 
            step=5)
        total_bedrooms = st.number_input(
            "Total Bedrooms", 
            min_value=int(min_values['total_bedrooms']), 
            max_value=int(max_values['total_bedrooms']), 
            step=5)
        
        population = st.number_input(
            "Population of the Locality", 
            min_value=int(min_values['population']), 
            max_value=int(max_values['population']), step=5)

    with subcol2:
        households = st.number_input(
            "Number of Households in the Locality", 
            min_value=int(min_values['households']), 
            max_value=int(max_values['households']), step=5)
        
        median_income = st.slider(
            "Median Income of the Locality", 
            min_value=float(min_values['median_income']), 
            max_value=float(max_values['median_income']), step=0.5)
        
        ocean_proximity = st.selectbox(
        'Ocean Proximity:',
        ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))

    address = st.text_input("Address")
    st.caption("Press enter to mark the address in the map.")
    st.write(st.session_state['address_output'])
    
    
    if address and (address != st.session_state['address'] or not st.session_state['markers']):

        st.session_state['address'] = address

        location = get_location(address)
        st.session_state['location'] = location

        if location:
            state = location.raw['address']['state']

            if state == 'California':
                
                housing_marker = create_marker(map_ca, location, popup=location)
                nearest_city = get_nearest_city(location)

                st.session_state['address_output'] = f'Nearest City: {nearest_city}'
                nearest_city_marker = create_marker(
                    map_ca, get_location(nearest_city), 
                    icon_color='green', popup=nearest_city)
                
                line_markers = link_two_markers(map_ca, housing_marker, nearest_city_marker)
                
                st.session_state['markers'].append(housing_marker)
                st.session_state['markers'].append(nearest_city_marker)
                st.session_state['lines'].append(line_markers)


            else:
                st.warning(
                    "The location should be in California. \
                    Inputting a place that doesn't belong \
                    to the state of California\ will lead to inconsistent results.")
        
        else:
            st.error("Address not found.")

    button = st.button("Predict", use_container_width=True)

    if button:
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
    
    if st.session_state['prediction']:
        st.success(f"Median House Value: ${st.session_state['prediction']:.2f}")

with col2:
    for marker in st.session_state["markers"]:
        fg.add_child(marker)

    for line in st.session_state["lines"]:
        fg.add_child(line)
    


    # Add the map to st_data
    st_data = st_folium(map_ca, width=1200, feature_group_to_add=fg)

    clean_button = st.button("Clear markers")
    if clean_button:
        map_ca = create_map()
        clear_markers()
