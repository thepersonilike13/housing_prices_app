import streamlit as st
st.set_page_config(page_title="Predição de Preços de Imóveis", page_icon=":house:")
import pandas as pd
import numpy as np
import pickle
from combiner import CombinedAttributesAdder
import folium
from geopy.geocoders import Nominatim, GeoNames
from streamlit_folium import st_folium


def _max_width_(prcnt_width:int = 70):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .appview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )
_max_width_(70)
st.title("Predição de Preços de Imóveis na Califórnia")
st.title("")
st.title("")


data = pd.read_csv('housing.csv')
max_values = data.select_dtypes(include=np.number).max()
min_values = data.select_dtypes(include=np.number).min()

@st.cache_resource
def load_model(filepath: str):
    return pickle.load(open(filepath, 'rb'))

loaded_model = load_model('reg.sav')

def get_location(address: str, geolocator):
    return geolocator.geocode(address, addressdetails=True)

def create_marker(m: folium.Map, location, address=None):
    coords = [location.latitude, location.longitude]
    marker = folium.Marker(location=coords, popup=address, icon=folium.Icon(color='red'))
    return marker

def clear_markers():
    st.session_state['markers'] = []

def create_map():
    # Define as coordenadas da Califórnia
    min_lat, min_lon = 32.5295, -124.4820
    max_lat, max_lon = 42.0095, -114.1315

    # Cria um mapa centralizado na Califórnia
    map_ca = folium.Map(location=[37.7749, -122.4194], zoom_start=6, min_lat=min_lat, min_lon=min_lon, max_lat=max_lat, max_lon=max_lon, no_wrap=True, max_bounds=True)

    return map_ca

@st.cache_resource
def initialize_nominatim():
    return Nominatim(user_agent="app")

if 'markers' not in st.session_state:
    st.session_state['markers'] = []

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'address' not in st.session_state:
    st.session_state['address'] = None

if 'location' not in st.session_state:
    st.session_state['location'] = None


map_ca = create_map()
fg = folium.FeatureGroup(name="markers")

geolocator = initialize_nominatim()

col1, col2 = st.columns([1, 2], gap='large')
with col1:
    st.header("Digite os atributos do imóvel.")
    subcol1, subcol2 = st.columns(2)

    with subcol1:
        housing_median_age = st.number_input(
            "Mediana de idade (em anos)",
            min_value=int(min_values['housing_median_age']), 
            max_value=int(max_values['housing_median_age']), 
            step=1)
        total_rooms = st.number_input(
            "Total de quartos", 
            min_value=int(min_values['total_rooms']), 
            max_value=int(max_values['total_rooms']), 
            step=5)
        total_bedrooms = st.number_input(
            "Total de quartos de dormir", 
            min_value=int(min_values['total_bedrooms']), 
            max_value=int(max_values['total_bedrooms']), 
            step=5)
        
        population = st.number_input(
            "População da localidade", 
            min_value=int(min_values['population']), 
            max_value=int(max_values['population']), step=5)

    with subcol2:
        households = st.number_input(
            "Número de lares na localidade", 
            min_value=int(min_values['households']), 
            max_value=int(max_values['households']), step=5)
        
        median_income = st.slider(
            "Renda mediana da localidade", 
            min_value=float(min_values['median_income']), 
            max_value=float(max_values['median_income']), step=0.5)
        
        ocean_proximity = st.selectbox(
        'Proximidade com o mar:',
        ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))

    address = st.text_input("Address")
    st.caption("Press enter to mark the address in the map.")
    
    if address and address != st.session_state['address']:
        st.session_state['address'] = address

        location = get_location(address, geolocator)
        st.session_state['location'] = location

        if location:
            state = location.raw['address']['state']

            if state == 'California':
                loc = np.array([location.longitude, location.latitude])

                st.text(f"Longitude: {location.longitude:.2f} \t\t  \t\t Latitude: {location.latitude:.2f}")
                
                marker = create_marker(map_ca, location, address=address)
                st.session_state['markers'].append(marker)

            else:
                st.warning("O local deve ser na Califórnia. Inserir um local que não pertença ao estado da Califórnia levará a resultados inconsistentes.")
        
        else:
            st.error("Endereço não encontrado.")



    button = st.button("Calcular a predição", use_container_width=True)

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
        st.metric(label="Valor mediano da casa", value=f"$ {st.session_state['prediction']:.2f}")


with col2:

    for marker in st.session_state["markers"]:
        fg.add_child(marker)

    # Add the map to st_data
    st_data = st_folium(map_ca, width=1200, feature_group_to_add=fg)

    clean_button = st.button("Clear markers")

    if clean_button:
        clear_markers()