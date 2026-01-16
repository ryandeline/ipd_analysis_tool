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

# --- TRACT LEVEL VARIABLES (Original S-Tables) ---
ACS_VARS_TRACT = {
    'TOT_POP': {'count': 'B01003_001', 'universe': None},
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

# --- BLOCK GROUP LEVEL VARIABLES (Substituted B/C-Tables) ---
# Note: Disability (D) is unavailable at Block Group level.
ACS_VARS_BG = {
    'TOT_POP': {'count': 'B01003_001', 'universe': None},
    'Y': {'count': 'B09001_001', 'universe': 'B01003_001'},
    'OA': {
        # Summing Males 65+ (20-25) and Females 65+ (44-49)
        'count': [f'B01001_0{i:03d}' for i in range(20, 26)] + [f'B01001_0{i:03d}' for i in range(44, 50)],
        'universe': 'B01001_001'
    },
    'F': {'count': 'B01001_026', 'universe': 'B01001_001'},
    'RM': {'count': 'B02001_002', 'universe': 'B02001_001', 'calc_type': 'subtract'},
    'EM': {'count': 'B03002_012', 'universe': 'B03002_001'},
    'FB': {'count': 'B05012_003', 'universe': 'B05012_001'},
    # LEP Proxy: Limited English Speaking Households
    'LEP': {
        'count': ['C16002_004', 'C16002_007', 'C16002_010', 'C16002_013'],
        'universe': 'C16002_001'
    },
    # Disability omitted (Not available)
    # Low Income Proxy: Ratio of Income to Poverty < 2.00
    'LI': {
        'count': ['C17002_002', 'C17002_003', 'C17002_004', 'C17002_005', 'C17002_006', 'C17002_007'],
        'universe': 'C17002_001'
    },
    'NC': {'count': 'B08201_002', 'universe': 'B08201_001'}
}

# --- Utility Functions ---

@st.cache_data
def get_all_counties(api_key, state_fips):
    if not api_key: return []
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
    
    # Handle 'count' being a list (for summation) or a string
    raw_counts = codes.get('count')
    if isinstance(raw_counts, str):
        count_vars = [raw_counts]
    else:
        count_vars = raw_counts # It's already a list
        
    u_var = codes.get('universe')
    
    # Flatten vars to fetch
    vars_to_fetch = []
    for c in count_vars:
        vars_to_fetch.extend([f"{c}E", f"{c}M"])
    if u_var:
        vars_to_fetch.extend([f"{u_var}E", f"{u_var}M"])
    
    vars_to_fetch = list(set(vars_to_fetch))
    var_str = ",".join(vars_to_fetch)
    
    if counties:
        county_str = ",".join(counties)
        geo_clause = f"&for={geo_level}:*&in=state:{state_fips}&in=county:{county_str}"
    else:
        geo_clause = f"&for={geo_level}:*&in=state:{state_fips}"
        
    # Check if we are using S-tables (Subject) or B/C-tables (Detail)
    is_subject = any(v.startswith('S') or v.startswith('DP') for v in vars_to_fetch)
    endpoint = "acs/acs5/subject" if is_subject else "acs/acs5"
    
    call_url = f"{base_url}/{year}/{endpoint}?get={var_str},NAME{geo_clause}&key={api_key}"
    
    try:
        r = requests.get(call_url)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Convert all data columns to numeric
        cols_to_numeric = [c for c in df.columns if c not in ['NAME', 'state', 'county', 'tract', 'block group']]
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
        # --- SUMMATION LOGIC ---
        # 1. Sum Estimates
        est_cols = [f"{c}E" for c in count_vars]
        df[f"{indicator_code}_est"] = df[est_cols].sum(axis=1)
        
        # 2. Sum MOEs (sqrt(sum(moe^2)))
        moe_cols = [f"{c}M" for c in count_vars]
        df[f"{indicator_code}_est_moe"] = np.sqrt((df[moe_cols] ** 2).sum(axis=1))
        
        # 3. Handle Universe
        if u_var:
            df[f"{indicator_code}_uni"] = df[f"{u_var}E"]
            df[f"{indicator_code}_uni_moe"] = df[f"{u_var}M"]
            
        return df
    except Exception as e:
        st.warning(f"Failed to fetch {indicator_code}. Error: {e}")
        return pd.DataFrame()

def patch_missing_columns(df):
    # This function is largely redundant with the new fetch logic 
    # but kept for safety with legacy column names if any slip through
    return df

def process_indicators(df, active_indicators):
    # Racial Minority Calculation: Universe - White Alone
    if 'RM' in active_indicators and 'RM_uni' in df.columns and 'RM_est' in df.columns:
        # Currently RM_est is "White Alone". We want "Non-White".
        # Non-White = Total - White Alone
        df['RM_est_original'] = df['RM_est'] # store white alone for verification if needed
        df['RM_est'] = df['RM_uni'] - df['RM_est_original']
        
        # MOE for subtraction: sqrt(moe1^2 + moe2^2)
        if 'RM_uni_moe' in df.columns and 'RM_est_moe' in df.columns:
            df['RM_est_moe'] = np.sqrt(df['RM_uni_moe']**2 + df['RM_est_moe']**2)
    
    for ind in active_indicators:
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
    
    # Select variable set based on geography
    if geo_level == 'block group':
        VARIABLES = ACS_VARS_BG
        # Filter out 'TOT_POP' and keys that don't exist in BG (like 'D')
        active_inds = [k for k in VARIABLES.keys() if k != 'TOT_POP']
    else:
        VARIABLES = ACS_VARS_TRACT
        active_inds = [k for k in VARIABLES.keys() if k != 'TOT_POP']

    indicator_dfs = []
    
    for i, ind_code in enumerate(active_inds):
        codes = VARIABLES[ind_code]
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

    df_master = process_indicators(df_master, active_inds)
    full_df_scored, summary_stats = calculate_sd_scores(df_master, active_inds)

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
    api_key_placeholder = st.secrets.get("CENSUS_API_KEY", "dfb115d4ff6b35a8ccc01892add4258ba7b48eaf")
    api_key = st.text_input("Census API Key", value=api_key_placeholder, type="password")
    
    with st.expander("🔑 How to get an API Key"):
        st.markdown("[Get Key Here](https://api.census.gov/data/key_signup.html)")
    
    st.divider()
    
    selected_state_name = st.selectbox("Select State", options=list(US_STATES_FIPS.keys()))
    selected_state_fips = US_STATES_FIPS[selected_state_name]

    available_counties = get_all_counties(api_key, selected_state_fips)
    
    selected_county_names = st.multiselect(
        "Select Counties (optional)",
        options=[c['name'] for c in available_counties],
        key=f"county_select_{selected_state_fips}",
        help="Leave blank to analyze all counties."
    )
    selected_county_fips = [c['fips'] for c in available_counties if c['name'] in selected_county_names]

    year = st.selectbox("ACS 5-Year Data", [2022, 2021, 2020])
    
    # Updated: Now supports Block Group
    geo_level = st.selectbox("Geography Level", ['block group', 'tract'])
    
    st.divider()
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_btn:
    if not api_key: 
        st.error("Census API Key required.")
    else:
        results = run_analysis(api_key, selected_state_fips, selected_state_name, selected_county_fips, year, geo_level)
        st.session_state.analysis_results = results

if st.session_state.analysis_results:
    final_gdf, summary_stats = st.session_state.analysis_results
    st.title(f"IPD Analysis: {selected_state_name}")
    
    if geo_level == 'block group':
        st.warning("⚠️ Note: Disability (D) data is excluded from Block Group analysis as it is not available from the Census at this level.")

    tabs = st.tabs(["🗺️ Map", "📊 Stats", "📋 Data"])
    
    with tabs[0]:
        if not final_gdf.empty:
            m = folium.Map(location=[final_gdf.geometry.centroid.y.mean(), final_gdf.geometry.centroid.x.mean()], zoom_start=9)
            folium.Choropleth(
                geo_data=final_gdf.to_json(),
                data=final_gdf,
                columns=['GEOID', 'IPD_SCORE'],
                key_on='feature.properties.GEOID',
                fill_color='YlOrRd',
                legend_name='IPD Score'
            ).add_to(m)
            st_folium(m, width=1100, height=600)
    
    with tabs[1]:
        st.dataframe(summary_stats, use_container_width=True)
        
    with tabs[2]:
        st.dataframe(final_gdf.drop(columns='geometry'), use_container_width=True)