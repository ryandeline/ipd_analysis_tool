# ----------------------------------------------------------------------
# Indicators of Potential Disadvantage (IPD) Analyzer - Streamlit App
# ----------------------------------------------------------------------
import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import numpy as np
import io
import folium
from streamlit_folium import st_folium
from functools import reduce

# --- Page Configuration ---
st.set_page_config(
    page_title="IPD Analyzer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Configuration ---
US_STATES_FIPS = {
    'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05', 'California': '06',
    'Colorado': '08', 'Connecticut': '09', 'Delaware': '10', 'District of Columbia': '11',
    'Florida': '12', 'Georgia': '13', 'Hawaii': '15', 'Idaho': '16', 'Illinois': '17',
    'Indiana': '18', 'Iowa': '19', 'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22',
    'Maine': '23', 'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26',
    'Minnesota': '27', 'Mississippi': '28', 'Missouri': '29', 'Montana': '30',
    'Nebraska': '31', 'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34',
    'New Mexico': '35', 'New York': '36', 'North Carolina': '37', 'North Dakota': '38',
    'Ohio': '39', 'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42',
    'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46', 'Tennessee': '47',
    'Texas': '48', 'Utah': '49', 'Vermont': '50', 'Virginia': '51', 'Washington': '53',
    'West Virginia': '54', 'Wisconsin': '55', 'Wyoming': '56'
}

ACS_VARIABLES = {
    'TOT_POP': 'B01003_001',
    'Y': {'count': 'B09001_001', 'universe': 'B01003_001'},
    'OA': {'count': 'S0101_C01_030', 'universe': 'S0101_C01_001'},
    'F': {'count': 'S0101_C05_001', 'universe': 'S0101_C01_001'},
    'RM': {'count': 'B02001_002', 'universe': 'B02001_001', 'calc_type': 'subtract'},
    'EM': {'count': 'B03002_012', 'universe': 'B03002_001'},
    'FB': {'count': 'B05012_003', 'universe': 'B05012_001'},
    'LEP': {'count': 'S1601_C05_001', 'universe': 'S1601_C01_001'},
    'D': {'count': 'S1810_C02_001', 'universe': 'S1810_C01_001'},
    'LI': {'count': 'S1701_C01_042', 'universe': 'S1701_C01_001'},
    'NC': {'count': 'B08201_002', 'universe': 'B08201_001'}
}

# --- Utility Functions ---

@st.cache_data
def get_all_counties(api_key, state_fips):
    """Fetches the list of all counties for a specific state from Census API."""
    if not api_key:
        return []
    url = f"https://api.census.gov/data/2022/acs/acs5?get=NAME&for=county:*&in=state:{state_fips}&key={api_key}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        counties = []
        for row in data[1:]:
            full_name = row[0]
            display_name = full_name.split(',')[0]
            counties.append({'name': display_name, 'fips': row[2]})
        return sorted(counties, key=lambda x: x['name'])
    except Exception:
        return []

@st.cache_data
def get_census_geometry(year, state_fips, geo_level):
    base_url = "https://www2.census.gov/geo/tiger"
    layer_name = "TRACT" if geo_level == 'tract' else "BG"
    file_code = "tract" if geo_level == 'tract' else "bg"
    url = f"{base_url}/TIGER{year}/{layer_name}/tl_{year}_{state_fips}_{file_code}.zip"
    try:
        gdf = gpd.read_file(url, engine='pyogrio')
        keep_cols = ['GEOID', 'geometry', 'ALAND', 'AWATER']
        return gdf[keep_cols]
    except Exception as e:
        st.error(f"Error fetching geometry: {e}")
        return gpd.GeoDataFrame()

def fetch_single_indicator(indicator_code, codes, year, state_fips, counties, geo_level, api_key):
    base_url = "https://api.census.gov/data"
    c_var = codes.get('count')
    u_var = codes.get('universe')
    vars_to_fetch = list(set([f"{c_var}E", f"{c_var}M", f"{u_var}E", f"{u_var}M"]))
    var_str = ",".join(vars_to_fetch)
    if counties:
        county_str = ",".join(counties)
        geo_clause = f"&for={geo_level}:*&in=state:{state_fips}&in=county:{county_str}"
    else:
        geo_clause = f"&for={geo_level}:*&in=state:{state_fips}"
    is_subject = any(v.startswith('S') or v.startswith('DP') for v in vars_to_fetch)
    endpoint = "acs/acs5/subject" if is_subject else "acs/acs5"
    call_url = f"{base_url}/{year}/{endpoint}?get={var_str},NAME{geo_clause}&key={api_key}"
    try:
        r = requests.get(call_url)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        for col in df.columns:
            if col not in ['NAME', 'state', 'county', 'tract', 'block group']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.warning(f"Failed to fetch {indicator_code}. Error: {e}")
        return pd.DataFrame()

def patch_missing_columns(df):
    rename_map = {
        'S0101_C01_030E': 'OA_est', 'S0101_C01_030M': 'OA_est_moe', 'S0101_C05_001E': 'F_est',
        'S0101_C05_001M': 'F_est_moe', 'S1601_C01_001E': 'LEP_uni', 'S1601_C05_001E': 'LEP_est',
        'S1810_C01_001E': 'D_uni', 'S1810_C02_001E': 'D_est', 'S1701_C01_001E': 'LI_uni', 
        'S1701_C01_042E': 'LI_est', 'B09001_001E': 'Y_est', 'B01003_001E': 'Y_uni',
        'B02001_002E': 'RM_est', 'B02001_001E': 'RM_uni', 'B03002_012E': 'EM_est',
        'B03002_001E': 'EM_uni', 'B05012_003E': 'FB_est', 'B05012_001E': 'FB_uni',
        'B08201_002E': 'NC_est', 'B08201_001E': 'NC_uni',
    }
    for raw_col, new_col in rename_map.items():
        if raw_col in df.columns:
            df.rename(columns={raw_col: new_col}, inplace=True)
    
    source_col = next((c for c in ['S0101_C01_001E', 'S0101_C01_001E_x'] if c in df.columns), None)
    if source_col:
        df['OA_uni'] = df[source_col]
        df['F_uni'] = df[source_col]
    return df

def process_indicators(df):
    if 'RM_uni' in df.columns and 'RM_est' in df.columns:
        df['RM_est'] = df['RM_uni'] - df['RM_est']
    
    indicators = ['Y', 'OA', 'F', 'RM', 'EM', 'FB', 'LEP', 'D', 'LI', 'NC']
    for ind in indicators:
        if f'{ind}_est' in df.columns and f'{ind}_uni' in df.columns:
            df[f'{ind}_pct'] = np.where(df[f'{ind}_uni'] > 0, (df[f'{ind}_est'] / df[f'{ind}_uni']) * 100, 0).round(1)
    return df

def calculate_sd_scores(df, indicators):
    df_scored = df.copy()
    df_scored['IPD_SCORE'] = 0
    stats_list = []
    
    for ind in indicators:
        pct_col = f'{ind}_pct'
        if pct_col not in df.columns: continue
        
        mean_val, sd_val = df_scored[pct_col].mean(), df_scored[pct_col].std()
        b1, b2, b3, b4 = max(0, mean_val - 1.5*sd_val), mean_val - 0.5*sd_val, mean_val + 0.5*sd_val, mean_val + 1.5*sd_val
        
        bins = [-np.inf, b1, b2, b3, b4, np.inf]
        df_scored[f'{ind}_score'] = pd.cut(df_scored[pct_col], bins=bins, labels=[0, 1, 2, 3, 4], right=False).astype(int)
        df_scored['IPD_SCORE'] += df_scored[f'{ind}_score']
        
        stats_list.append({'Indicator': ind, 'Mean': round(mean_val, 1), 'SD': round(sd_val, 1)})
    return df_scored, pd.DataFrame(stats_list)

def run_analysis(api_key, state_fips, state_name, counties, year, geo_level):
    status = st.status(f"Starting analysis for {state_name}...", expanded=True)
    indicator_dfs = []
    
    for i, (ind_code, codes) in enumerate(ACS_VARIABLES.items()):
        if ind_code == 'TOT_POP': continue
        status.write(f"Fetching {ind_code}...")
        df_ind = fetch_single_indicator(ind_code, codes, year, state_fips, counties, geo_level, api_key)
        if not df_ind.empty: indicator_dfs.append(df_ind)

    if not indicator_dfs:
        st.error("No data fetched. Please check your API key.")
        st.stop()
    
    merge_keys = ['state', 'county', 'tract', 'NAME']
    if geo_level == 'block group': merge_keys.append('block group')
    df_master = reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how='outer'), indicator_dfs)
    df_master['GEOID'] = df_master['state'] + df_master['county'] + df_master['tract']
    if geo_level == 'block group': df_master['GEOID'] += df_master['block group']

    df_master = patch_missing_columns(df_master)
    df_master = process_indicators(df_master)
    full_df_scored, summary_stats = calculate_sd_scores(df_master, ['Y', 'OA', 'F', 'RM', 'EM', 'FB', 'LEP', 'D', 'LI', 'NC'])

    gdf_geom = get_census_geometry(year, state_fips, geo_level)
    if counties: gdf_geom = gdf_geom[gdf_geom['GEOID'].str[2:5].isin(counties)]
    
    final_gdf = gdf_geom.merge(full_df_scored, on='GEOID', how='inner')
    status.update(label="Complete!", state="complete")
    return final_gdf, summary_stats

# --- UI and Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Secure API Key Handling
    # 1. Tries to find CENSUS_API_KEY in Streamlit Secrets
    # 2. Defaults to your test key if secrets aren't set up yet
    # REMINDER: Remove the hardcoded fallback 'dfb115...' before going live.
    api_key_placeholder = st.secrets.get("CENSUS_API_KEY", "dfb115d4ff6b35a8ccc01892add4258ba7b48eaf")
    api_key = st.text_input("Census API Key", value=api_key_placeholder, type="password")
    
    with st.expander("🔑 How to get an API Key"):
        st.markdown("""
        1. Visit the [Census API Key Signup page](https://api.census.gov/data/key_signup.html).
        2. Enter your Organization Name and Email.
        3. You will receive an email with your key.
        4. Copy and paste it here!
        """)
    
    st.divider()
    
    selected_state_name = st.selectbox("Select State", options=list(US_STATES_FIPS.keys()))
    selected_state_fips = US_STATES_FIPS[selected_state_name]

    available_counties = get_all_counties(api_key, selected_state_fips)
    
    selected_county_names = st.multiselect(
        "Select Counties (optional)",
        options=[c['name'] for c in available_counties],
        key=f"county_select_{selected_state_fips}",
        help="Leave blank to analyze all counties in the selected state."
    )
    selected_county_fips = [c['fips'] for c in available_counties if c['name'] in selected_county_names]

    year = st.selectbox("ACS 5-Year Data", [2022, 2021, 2020])
    
    # UPDATE: Restricted to 'tract' only because API fails for Block Groups on Subject Tables (S*)
    geo_level = st.selectbox(
        "Geography Level", 
        ['tract'], 
        help="Block Group analysis is unavailable because critical indicators (Disability, LEP, etc.) rely on Census Subject Tables (S-series), which are only published down to the Tract level."
    )
    
    st.divider()
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Execute Analysis
if run_btn:
    if not api_key: 
        st.error("A Census API Key is required to run this analysis.")
    else:
        results = run_analysis(api_key, selected_state_fips, selected_state_name, selected_county_fips, year, geo_level)
        st.session_state.analysis_results = results

# Display Persistent Results
if st.session_state.analysis_results:
    final_gdf, summary_stats = st.session_state.analysis_results
    st.title(f"IPD Analysis: {selected_state_name}")
    
    tabs = st.tabs(["🗺️ Visual Map", "📊 Summary Stats", "📋 Raw Data"])
    
    with tabs[0]:
        if not final_gdf.empty:
            avg_lat = final_gdf.geometry.centroid.y.mean()
            avg_lon = final_gdf.geometry.centroid.x.mean()
            
            m = folium.Map(location=[avg_lat, avg_lon], zoom_start=9)
            folium.Choropleth(
                geo_data=final_gdf.to_json(),
                data=final_gdf,
                columns=['GEOID', 'IPD_SCORE'],
                key_on='feature.properties.GEOID',
                fill_color='YlOrRd',
                legend_name='IPD Composite Score (Higher = More Disadvantaged)'
            ).add_to(m)
            st_folium(m, width=1100, height=600)
        else:
            st.warning("No geographic data matched your criteria. Check your county filters.")
            
    with tabs[1]:
        st.subheader("Indicator Statistics")
        st.dataframe(summary_stats, use_container_width=True)
        st.info("The IPD Score is calculated using standard deviation breaks across the 10 demographic indicators.")
        
    with tabs[2]:
        st.subheader("Export Data")
        st.dataframe(final_gdf.drop(columns='geometry'), use_container_width=True)
else:
    st.info("👋 Welcome! Use the sidebar to enter your API key, select a state, and click 'Run Analysis'.")