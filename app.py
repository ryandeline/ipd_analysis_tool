
# ----------------------------------------------------------------------
# Indicators of Potential Disadvantage (IPD) Analyzer - Streamlit App
# ----------------------------------------------------------------------
# This single file consolidates the logic from the 6 original Python scripts
# (run_ipd.py, ipd_config.py, etc.) into a web application using Streamlit.
# To deploy, you will need this file and a 'requirements.txt' file.
# ----------------------------------------------------------------------

import streamlit as st
import pandas as pd
import geopandas as gpd
import requests
import numpy as np
import io
import zipfile
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

# --- Constants & Configuration (from ipd_config.py and constants.ts) ---

# This data is derived from your constants.ts file
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

# A small subset of counties for demonstration; a full implementation would include all counties.
US_COUNTIES_FIPS = {
    '11': [{'name': 'District of Columbia', 'fips': '001'}],
    '42': [
        {'name': 'Philadelphia County', 'fips': '101'},
        {'name': 'Allegheny County', 'fips': '003'},
        {'name': 'Montgomery County', 'fips': '091'},
        {'name': 'Bucks County', 'fips': '017'},
        {'name': 'Delaware County', 'fips': '045'}
    ],
    '34': [
        {'name': 'Bergen County', 'fips': '003'},
        {'name': 'Middlesex County', 'fips': '023'},
        {'name': 'Essex County', 'fips': '013'},
        {'name': 'Hudson County', 'fips': '017'},
        {'name': 'Monmouth County', 'fips': '025'}
    ]
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

# --- Utility Functions (from original scripts, adapted for Streamlit) ---

# Caching is crucial for performance in Streamlit.
# It prevents re-downloading data on every interaction.
@st.cache_data
def get_census_geometry(year, state_fips, geo_level):
    """(from geo_utils.py) Fetches Census TIGER/Line shapefiles."""
    base_url = "https://www2.census.gov/geo/tiger"
    layer_name = "TRACT" if geo_level == 'tract' else "BG"
    file_code = "tract" if geo_level == 'tract' else "bg"
    url = f"{base_url}/TIGER{year}/{layer_name}/tl_{year}_{state_fips}_{file_code}.zip"
    try:
        gdf = gpd.read_file(url, engine='pyogrio')
        keep_cols = ['GEOID', 'geometry', 'ALAND', 'AWATER']
        return gdf[keep_cols]
    except Exception as e:
        st.error(f"Error fetching geometry from {url}: {e}")
        return gpd.GeoDataFrame()

def fetch_single_indicator(indicator_code, codes, year, state_fips, counties, geo_level, api_key, progress_bar):
    """(from run_ipd.py) Fetches data for one indicator."""
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
        progress_bar.text(f"Successfully fetched {indicator_code}.")
        return df
    except Exception as e:
        st.warning(f"Failed to fetch {indicator_code}. URL: {call_url}. Error: {e}")
        return pd.DataFrame()

def patch_missing_columns(df):
    """(from run_ipd.py) Renames raw Census columns."""
    rename_map = {
        'S0101_C01_030E': 'OA_est', 'S0101_C01_030M': 'OA_est_moe', 'S0101_C05_001E': 'F_est',
        'S0101_C05_001M': 'F_est_moe', 'S1601_C01_001E': 'LEP_uni', 'S1601_C01_001M': 'LEP_uni_moe',
        'S1601_C05_001E': 'LEP_est', 'S1601_C05_001M': 'LEP_est_moe', 'S1810_C01_001E': 'D_uni',
        'S1810_C01_001M': 'D_uni_moe', 'S1810_C02_001E': 'D_est', 'S1810_C02_001M': 'D_est_moe',
        'S1701_C01_001E': 'LI_uni', 'S1701_C01_001M': 'LI_uni_moe', 'S1701_C01_042E': 'LI_est',
        'S1701_C01_042M': 'LI_est_moe', 'B09001_001E': 'Y_est', 'B09001_001M': 'Y_est_moe',
        'B01003_001E': 'Y_uni', 'B01003_001M': 'Y_uni_moe', 'B02001_002E': 'RM_est',
        'B02001_002M': 'RM_est_moe', 'B02001_001E': 'RM_uni', 'B02001_001M': 'RM_uni_moe',
        'B03002_012E': 'EM_est', 'B03002_012M': 'EM_est_moe', 'B03002_001E': 'EM_uni',
        'B03002_001M': 'EM_uni_moe', 'B05012_003E': 'FB_est', 'B05012_003M': 'FB_est_moe',
        'B05012_001E': 'FB_uni', 'B05012_001M': 'FB_uni_moe', 'B08201_002E': 'NC_est',
        'B08201_002M': 'NC_est_moe', 'B08201_001E': 'NC_uni', 'B08201_001M': 'NC_uni_moe',
    }
    for raw_col, new_col in rename_map.items():
        if raw_col in df.columns:
            df.rename(columns={raw_col: new_col}, inplace=True)
        else:
            for suffix in ['_x', '_y']:
                if f"{raw_col}{suffix}" in df.columns and new_col not in df.columns:
                    df.rename(columns={f"{raw_col}{suffix}": new_col}, inplace=True)
    
    source_col = next((c for c in ['S0101_C01_001E', 'S0101_C01_001E_x', 'S0101_C01_001E_y'] if c in df.columns), None)
    source_col_moe = next((c for c in ['S0101_C01_001M', 'S0101_C01_001M_x', 'S0101_C01_001M_y'] if c in df.columns), None)
    if source_col:
        df['OA_uni'] = df[source_col]
        df['F_uni'] = df[source_col]
    if source_col_moe:
        df['OA_uni_moe'] = df[source_col_moe]
        df['F_uni_moe'] = df[source_col_moe]
    return df

def process_indicators(df):
    """(from analysis_utils.py) Calculates percentages and MOEs."""
    if 'RM_uni' in df.columns and 'RM_est' in df.columns:
        df['RM_est'] = df['RM_uni'] - df['RM_est']
        if 'RM_uni_moe' in df.columns and 'RM_est_moe' in df.columns:
            df['RM_est_moe_approx'] = np.sqrt(df['RM_uni_moe']**2 + df['RM_est_moe']**2)
    
    indicators = ['Y', 'OA', 'F', 'RM', 'EM', 'FB', 'LEP', 'D', 'LI', 'NC']
    for ind in indicators:
        if f'{ind}_est' not in df.columns or f'{ind}_uni' not in df.columns:
            continue
        df[f'{ind}_pct'] = np.where(df[f'{ind}_uni'] > 0, (df[f'{ind}_est'] / df[f'{ind}_uni']) * 100, 0).round(1)
        
        # MOE for proportion
        num, den, num_moe, den_moe = df[f'{ind}_est'], df[f'{ind}_uni'], df[f'{ind}_est_moe'], df[f'{ind}_uni_moe']
        prop_sq = (num / den)**2
        inner_val = np.maximum(0, num_moe**2 - prop_sq * den_moe**2) # Ensure non-negative
        df[f'{ind}_pct_moe'] = np.where(den > 0, (np.sqrt(inner_val) / den) * 100, 0).round(1)
        
    return df

def calculate_sd_scores(df, indicators):
    """(from analysis_utils.py) Applies Standard Deviation scoring."""
    df_scored = df.copy()
    df_scored['IPD_SCORE'] = 0
    stats_list = []
    
    for ind in indicators:
        pct_col = f'{ind}_pct'
        if pct_col not in df.columns: continue
        
        mean_val, sd_val = df_scored[pct_col].mean(), df_scored[pct_col].std()
        b1 = max(0.1, mean_val - (1.5 * sd_val))
        b2 = mean_val - (0.5 * sd_val)
        b3 = mean_val + (0.5 * sd_val)
        b4 = mean_val + (1.5 * sd_val)
        
        bins = [-np.inf, b1, b2, b3, b4, np.inf]
        labels = [0, 1, 2, 3, 4]
        score_col = f'{ind}_score'
        df_scored[score_col] = pd.cut(df_scored[pct_col], bins=bins, labels=labels, right=False)
        df_scored['IPD_SCORE'] += df_scored[score_col].astype(int)
        
        stats_list.append({
            'Indicator': ind, 'Mean': round(mean_val, 1), 'SD': round(sd_val, 1),
            'Break_Min_1.5SD': round(b1, 1), 'Break_Min_0.5SD': round(b2, 1),
            'Break_Plus_0.5SD': round(b3, 1), 'Break_Plus_1.5SD': round(b4, 1)
        })

    return df_scored, pd.DataFrame(stats_list)


# --- Main Analysis Function ---
def run_analysis(api_key, state_fips, state_name, counties, year, geo_level):
    """
    (from run_ipd.py) This is the main orchestrator function, refactored
    to take UI inputs and return DataFrames instead of saving files.
    """
    status = st.status(f"Starting analysis for {state_name}...", expanded=True)
    
    indicator_dfs = []
    total_indicators = len([i for i in ACS_VARIABLES if i != 'TOT_POP'])
    progress_bar = status.progress(0)
    
    for i, (ind_code, codes) in enumerate(ACS_VARIABLES.items()):
        if ind_code == 'TOT_POP': continue
        status.write(f"Fetching indicator {i+1}/{total_indicators}: {ind_code}...")
        df_ind = fetch_single_indicator(ind_code, codes, year, state_fips, counties, geo_level, api_key, status)
        if not df_ind.empty:
            indicator_dfs.append(df_ind)
        progress_bar.progress((i + 1) / total_indicators)

    if not indicator_dfs:
        status.error("No data could be fetched. Check API key and selections.")
        st.stop()
    
    status.write("Merging indicator data...")
    merge_keys = ['state', 'county', 'tract', 'NAME']
    if geo_level == 'block group': merge_keys.append('block group')
    df_master = reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how='outer'), indicator_dfs)
    
    if geo_level == 'tract':
        df_master['GEOID'] = df_master['state'] + df_master['county'] + df_master['tract']
    else:
        df_master['GEOID'] = df_master['state'] + df_master['county'] + df_master['tract'] + df_master['block group']

    status.write("Cleaning and processing data...")
    df_master = patch_missing_columns(df_master)
    df_master = process_indicators(df_master)
    
    status.write("Calculating final scores...")
    indicators = ['Y', 'OA', 'F', 'RM', 'EM', 'FB', 'LEP', 'D', 'LI', 'NC']
    full_df_scored, summary_stats = calculate_sd_scores(df_master, indicators)

    status.write("Fetching geographic data...")
    gdf_geom = get_census_geometry(year, state_fips, geo_level)
    if counties:
        gdf_geom = gdf_geom[gdf_geom['GEOID'].str[2:5].isin(counties)]
    
    status.write("Merging geographic and attribute data...")
    final_gdf = gdf_geom.merge(full_df_scored, on='GEOID', how='inner')
    
    status.update(label="Analysis Complete!", state="complete", expanded=False)
    
    return final_gdf, summary_stats

# --- Streamlit UI ---
st.title("🗺️ Indicators of Potential Disadvantage (IPD) Analyzer")
st.markdown("This tool fetches and analyzes US Census data to identify areas of potential disadvantage based on 10 demographic indicators. Configure the analysis in the sidebar and run.")

# Sidebar Controls
st.sidebar.header("⚙️ Analysis Controls")
api_key = st.sidebar.text_input("Census API Key", type="password", help="Your API key is required to fetch data from the Census Bureau.")
with st.sidebar.expander("How to get a Census API Key"):
    st.markdown("""
    1. Visit the [Census API Key Signup page](https://api.census.gov/data/key_signup.html).
    2. Fill out the short form.
    3. You will receive an email with your key. Copy and paste it above.
    """)

selected_state_name = st.sidebar.selectbox("Select a State", options=list(US_STATES_FIPS.keys()))
selected_state_fips = US_STATES_FIPS.get(selected_state_name)

available_counties = [{'name': 'All Counties', 'fips': 'ALL'}] + US_COUNTIES_FIPS.get(selected_state_fips, [])
selected_county_names = st.sidebar.multoselect(
    "Select Counties (optional)",
    options=[c['name'] for c in available_counties],
    help="Leave blank to analyze all counties in the selected state."
)

selected_county_fips = [c['fips'] for c in available_counties if c['name'] in selected_county_names]
if "ALL" in selected_county_fips or not selected_county_fips:
    selected_county_fips = [] # API expects empty list for all counties

col1, col2 = st.sidebar.columns(2)
year = col1.selectbox("ACS Year", [2022, 2021, 2020], help="Latest available ACS 5-Year data is usually from 2 years prior.")
geo_level = col2.selectbox("Geo Level", ['tract', 'block group'], help="Tract level is generally more reliable.")

run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)


# --- Main Panel: Execution and Results ---
if 'results' not in st.session_state:
    st.session_state.results = None

if run_button:
    if not api_key:
        st.error("Please enter a Census API Key in the sidebar.")
    elif not selected_state_fips:
        st.error("Please select a state.")
    else:
        with st.spinner("Running Analysis... This may take a few minutes."):
            final_gdf, summary_stats = run_analysis(api_key, selected_state_fips, selected_state_name, selected_county_fips, year, geo_level)
            st.session_state.results = (final_gdf, summary_stats)
            
if st.session_state.results:
    final_gdf, summary_stats = st.session_state.results
    
    st.success("Analysis complete!")
    
    # Display Results
    st.subheader("📊 Summary Statistics")
    st.dataframe(summary_stats)
    
    st.subheader("🗺️ Interactive Disadvantage Score Map")
    
    # Create Folium Map
    if not final_gdf.empty:
        # Center the map on the analyzed area
        map_center = [final_gdf.geometry.centroid.y.mean(), final_gdf.geometry.centroid.x.mean()]
        m = folium.Map(location=map_center, zoom_start=9)
        
        # Add Choropleth layer
        folium.Choropleth(
            geo_data=final_gdf.to_json(),
            name='IPD Score',
            data=final_gdf,
            columns=['GEOID', 'IPD_SCORE'],
            key_on='feature.properties.GEOID',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='IPD Composite Score'
        ).add_to(m)
        
        # Add tooltips
        tooltip = folium.GeoJsonTooltip(fields=['GEOID', 'NAME', 'IPD_SCORE'], aliases=['GEOID:', 'Name:', 'IPD Score:'])
        folium.GeoJson(final_gdf.to_json(), tooltip=tooltip).add_to(m)

        st_folium(m, width=1200, height=600)
    else:
        st.warning("No geographic data to display.")

    st.subheader("📋 Attribute Data Preview")
    st.dataframe(final_gdf.drop(columns='geometry'))
    
    st.subheader("💾 Download Results")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # Convert dataframes to CSV in memory for download
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    dl_col1.download_button(
        label="Download Summary (CSV)",
        data=convert_df_to_csv(summary_stats),
        file_name=f"IPD_Summary_{selected_state_name}_{year}.csv",
        mime='text/csv',
    )
    
    dl_col2.download_button(
        label="Download Full Attributes (CSV)",
        data=convert_df_to_csv(final_gdf.drop(columns='geometry')),
        file_name=f"IPD_Attributes_{selected_state_name}_{year}.csv",
        mime='text/csv',
    )

    # Convert GeoDataFrame to GeoPackage in memory for download
    @st.cache_data
    def convert_gdf_to_gpkg(_gdf):
        with io.BytesIO() as buffer:
            _gdf.to_file(buffer, driver='GPKG')
            return buffer.getvalue()
    
    dl_col3.download_button(
        label="Download Geospatial (GPKG)",
        data=convert_gdf_to_gpkg(final_gdf),
        file_name=f"IPD_Results_{selected_state_name}_{year}.gpkg",
        mime='application/octet-stream'
    )
else:
    st.info("Configure your analysis in the sidebar and click 'Run Analysis' to begin.")
    st.markdown("""
        ### How it Works
        1.  **Configure**: Use the sidebar to input your Census API key, select a state, and optionally narrow down by county.
        2.  **Fetch**: The app calls the US Census Bureau's API to get the latest ACS 5-Year data for 10 demographic indicators.
        3.  **Process**: It calculates percentages, margins of error, and a composite "IPD Score" for each census tract based on a standard deviation methodology.
        4.  **Visualize**: Results are displayed as an interactive map and data tables.
        5.  **Download**: You can download the summary stats, full attribute table, and a GeoPackage file for use in GIS software.
    """)