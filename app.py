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
ACS_VARS_BG = {
    'TOT_POP': {'count': 'B01003_001', 'universe': None},
    'Y': {'count': 'B09001_001', 'universe': 'B01003_001'},
    'OA': {
        'count': [f'B01001_0{i:03d}' for i in range(20, 26)] + [f'B01001_0{i:03d}' for i in range(44, 50)],
        'universe': 'B01001_001'
    },
    'F': {'count': 'B01001_026', 'universe': 'B01001_001'},
    'RM': {'count': 'B02001_002', 'universe': 'B02001_001', 'calc_type': 'subtract'},
    'EM': {'count': 'B03002_012', 'universe': 'B03002_001'},
    'FB': {'count': 'B05012_003', 'universe': 'B05012_001'},
    'LEP': {
        'count': ['C16002_004', 'C16002_007', 'C16002_010', 'C16002_013'],
        'universe': 'C16002_001'
    },
    'D': {'count': 'S1810_C02_001', 'universe': 'S1810_C01_001', 'interpolate': True},
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
    interpolate = codes.get('interpolate', False)
    fetch_geo_level = 'tract' if interpolate and geo_level == 'block group' else geo_level

    raw_counts = codes.get('count')
    count_vars = [raw_counts] if isinstance(raw_counts, str) else raw_counts
    u_var = codes.get('universe')
    
    vars_to_fetch = []
    for c in count_vars:
        vars_to_fetch.extend([f"{c}E", f"{c}M"])
    if u_var:
        vars_to_fetch.extend([f"{u_var}E", f"{u_var}M"])
    
    vars_to_fetch = list(set(vars_to_fetch))
    
    # Chunking variables to prevent URL length errors
    chunk_size = 40
    chunks = [vars_to_fetch[i:i + chunk_size] for i in range(0, len(vars_to_fetch), chunk_size)]
    
    partial_dfs = []
    
    for i, chunk in enumerate(chunks):
        var_str = ",".join(chunk)
        if counties:
            county_str = ",".join(counties)
            geo_clause = f"&for={fetch_geo_level}:*&in=state:{state_fips}&in=county:{county_str}"
        else:
            geo_clause = f"&for={fetch_geo_level}:*&in=state:{state_fips}"
            
        is_subject = any(v.startswith('S') or v.startswith('DP') for v in chunk)
        endpoint = "acs/acs5/subject" if is_subject else "acs/acs5"
        call_url = f"{base_url}/{year}/{endpoint}?get={var_str},NAME{geo_clause}&key={api_key}"
        
        try:
            r = requests.get(call_url)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            partial_dfs.append(df)
        except Exception as e:
            print(f"Failed to fetch chunk {i} for {indicator_code}. Error: {e}")
            return pd.DataFrame()

    if not partial_dfs: return pd.DataFrame()
    
    if len(partial_dfs) > 1:
        merge_keys = ['NAME', 'state', 'county', 'tract']
        if fetch_geo_level == 'block group': merge_keys.append('block group')
        df_final = reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how='outer'), partial_dfs)
    else:
        df_final = partial_dfs[0]
        
    cols_to_numeric = [c for c in df_final.columns if c not in ['NAME', 'state', 'county', 'tract', 'block group']]
    for col in cols_to_numeric:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
        
    # Summation Logic
    available_est_cols = [f"{c}E" for c in count_vars if f"{c}E" in df_final.columns]
    df_final[f"{indicator_code}_est"] = df_final[available_est_cols].sum(axis=1) if available_est_cols else 0
    
    available_moe_cols = [f"{c}M" for c in count_vars if f"{c}M" in df_final.columns]
    df_final[f"{indicator_code}_est_moe"] = np.sqrt((df_final[available_moe_cols] ** 2).sum(axis=1)) if available_moe_cols else 0
    
    if u_var:
        # Use simple get with default or check if column exists to avoid AttributeError on some GeoDataFrame versions
        if f"{u_var}E" in df_final.columns:
            df_final[f"{indicator_code}_uni"] = df_final[f"{u_var}E"]
        else:
            df_final[f"{indicator_code}_uni"] = 0
            
        if f"{u_var}M" in df_final.columns:
            df_final[f"{indicator_code}_uni_moe"] = df_final[f"{u_var}M"]
        else:
            df_final[f"{indicator_code}_uni_moe"] = 0
    
    return df_final

def process_indicators(df, active_indicators):
    if 'RM' in active_indicators and 'RM_uni' in df.columns and 'RM_est' in df.columns:
        df['RM_est'] = df['RM_uni'] - df['RM_est']
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
        
        if pd.isna(mean_val) or pd.isna(sd_val) or sd_val == 0:
            stats_list.append({'Indicator': ind, 'Mean': 0, 'SD': 0})
            df_scored[f'{ind}_score'] = 0
            continue

        b1, b2, b3, b4 = max(0, mean_val - 1.5*sd_val), mean_val - 0.5*sd_val, mean_val + 0.5*sd_val, mean_val + 1.5*sd_val
        bins = sorted(list(set([-np.inf, b1, b2, b3, b4, np.inf])))
        
        if len(bins) < 6:
            try:
                df_scored[f'{ind}_score'] = pd.cut(df_scored[pct_col], bins=5, labels=False).fillna(0).astype(int)
            except:
                df_scored[f'{ind}_score'] = 0
        else:
             df_scored[f'{ind}_score'] = pd.cut(df_scored[pct_col], bins=bins, labels=[0, 1, 2, 3, 4], right=False).fillna(0).astype(int)
             
        df_scored['IPD_SCORE'] += df_scored[f'{ind}_score']
        stats_list.append({'Indicator': ind, 'Mean': round(mean_val, 1), 'SD': round(sd_val, 1)})
        
    return df_scored, pd.DataFrame(stats_list)

def run_analysis(api_key, state_fips, state_name, counties, year, geo_level):
    status = st.status(f"Analysing {state_name} ({geo_level})...", expanded=True)
    
    VARIABLES = ACS_VARS_BG if geo_level == 'block group' else ACS_VARS_TRACT
    active_inds = [k for k in VARIABLES.keys() if k != 'TOT_POP']

    indicator_dfs = []
    total_steps = len(active_inds) + 2
    
    for i, ind_code in enumerate(active_inds):
        status.update(label=f"Fetching Indicator: {ind_code} ({i+1}/{total_steps})")
        df_ind = fetch_single_indicator(ind_code, VARIABLES[ind_code], year, state_fips, counties, geo_level, api_key)
        if not df_ind.empty: indicator_dfs.append(df_ind)

    if not indicator_dfs:
        st.error("No data fetched. Check API key.")
        st.stop()
    
    status.update(label="Merging datasets...")
    bg_dfs = [df for df in indicator_dfs if 'block group' in df.columns]
    tract_dfs = [df for df in indicator_dfs if 'block group' not in df.columns]
    
    if geo_level == 'block group' and not bg_dfs:
        st.error("Critical Error: No Block Group data fetched.")
        st.stop()
        
    common_keys = list(set.intersection(*(set(df.columns) for df in (bg_dfs if bg_dfs else tract_dfs))))
    base_keys = [k for k in ['state', 'county', 'tract', 'block group'] if k in common_keys]
    
    # Deduplicate NAME column
    for df in (bg_dfs if bg_dfs else tract_dfs):
        if 'NAME' in df.columns and 'NAME' not in base_keys: base_keys.append('NAME')

    try:
        if bg_dfs:
            df_master = reduce(lambda left, right: pd.merge(left, right, on=base_keys, how='outer'), bg_dfs)
        else:
            df_master = reduce(lambda left, right: pd.merge(left, right, on=base_keys, how='outer'), tract_dfs)
    except Exception as e:
        st.error(f"Merge error: {e}")
        st.stop()
    
    if tract_dfs and geo_level == 'block group':
        tract_keys = ['state', 'county', 'tract']
        for t_df in tract_dfs:
            if 'NAME' in t_df.columns: t_df = t_df.drop(columns=['NAME'])
            df_master = pd.merge(df_master, t_df, on=tract_keys, how='left')

    if geo_level == 'tract':
        df_master['GEOID'] = df_master['state'] + df_master['county'] + df_master['tract']
    else:
        df_master['GEOID'] = df_master['state'] + df_master['county'] + df_master['tract'] + df_master['block group']

    status.update(label="Calculating scores & geometry...")
    df_master = process_indicators(df_master, active_inds)
    full_df_scored, summary_stats = calculate_sd_scores(df_master, active_inds)

    gdf_geom = get_census_geometry(year, state_fips, geo_level)
    if counties: gdf_geom = gdf_geom[gdf_geom['GEOID'].str[2:5].isin(counties)]
    
    final_gdf = gdf_geom.merge(full_df_scored, on='GEOID', how='inner')
    status.update(label="Analysis Complete!", state="complete", expanded=False)
    return final_gdf, summary_stats

# --- UI Layout ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

with st.sidebar:
    st.title("🗺️ IPD Settings")
    
    with st.expander("🔑 API Key Configuration", expanded=False):
        api_key_placeholder = st.secrets.get("CENSUS_API_KEY", "dfb115d4ff6b35a8ccc01892add4258ba7b48eaf")
        api_key = st.text_input("Census API Key", value=api_key_placeholder, type="password")
        st.markdown("[Get a Key](https://api.census.gov/data/key_signup.html)")

    st.subheader("📍 Geography")
    selected_state_name = st.selectbox("State", options=list(US_STATES_FIPS.keys()))
    selected_state_fips = US_STATES_FIPS[selected_state_name]

    available_counties = get_all_counties(api_key, selected_state_fips)
    selected_county_names = st.multiselect(
        "Counties (Optional)",
        options=[c['name'] for c in available_counties],
        key=f"county_{selected_state_fips}",
        placeholder="All Counties"
    )
    selected_county_fips = [c['fips'] for c in available_counties if c['name'] in selected_county_names]

    st.subheader("📅 Parameters")
    col_yr, col_geo = st.columns(2)
    year = col_yr.selectbox("Year", [2022, 2021, 2020])
    geo_level = col_geo.selectbox("Level", ['tract', 'block group'])
    
    st.divider()
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# --- Main Dashboard ---
if run_btn:
    if not api_key: st.error("API Key Required")
    else:
        st.session_state.analysis_results = run_analysis(api_key, selected_state_fips, selected_state_name, selected_county_fips, year, geo_level)

if st.session_state.analysis_results:
    final_gdf, summary_stats = st.session_state.analysis_results
    
    # 1. Header Metrics
    st.markdown(f"### 📊 Results for {selected_state_name}")
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("Geographic Level", geo_level.title())
    met2.metric("Total Units Analyzed", f"{len(final_gdf):,}")
    
    # Safe access to TOT_POP_est using bracket notation with fillna, avoiding .get() on gdf
    total_pop = final_gdf['TOT_POP_est'].sum() if 'TOT_POP_est' in final_gdf.columns else 0
    met3.metric("Total Population", f"{int(total_pop):,}")
    met4.metric("Avg IPD Score", f"{final_gdf['IPD_SCORE'].mean():.1f}")
    
    st.divider()

    # 2. Main Map Visualization
    m = folium.Map(location=[final_gdf.geometry.centroid.y.mean(), final_gdf.geometry.centroid.x.mean()])
    # Calculate bounds for auto-zoom
    if not final_gdf.empty:
        if final_gdf.crs != 'EPSG:4326': final_gdf = final_gdf.to_crs('EPSG:4326')
        min_lon, min_lat, max_lon, max_lat = final_gdf.total_bounds
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    folium.Choropleth(
        geo_data=final_gdf.to_json(),
        data=final_gdf,
        columns=['GEOID', 'IPD_SCORE'],
        key_on='feature.properties.GEOID',
        fill_color='YlOrRd',
        legend_name='IPD Score'
    ).add_to(m)
    st_folium(m, width="100%", height=500)

    # 3. Data & Downloads
    tab_data, tab_stats = st.tabs(["📋 Detailed Data Table", "📈 Summary Statistics"])
    
    # Helpers
    @st.cache_data
    def convert_df(df): return df.to_csv(index=False).encode('utf-8')
    
    @st.cache_data
    def convert_gdf_to_geojson(_gdf): return _gdf.to_json()

    with tab_data:
        # Display Data with Progress Bar for IPD Score
        display_cols = ['GEOID', 'IPD_SCORE', 'TOT_POP_est'] + [c for c in final_gdf.columns if '_pct' in c]
        # Ensure cols exist
        display_cols = [c for c in display_cols if c in final_gdf.columns]
        display_df = final_gdf[display_cols].copy()
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "IPD_SCORE": st.column_config.ProgressColumn(
                    "IPD Score",
                    help="Composite Disadvantage Score",
                    format="%d",
                    min_value=0,
                    max_value=int(final_gdf['IPD_SCORE'].max()),
                ),
                "TOT_POP_est": st.column_config.NumberColumn("Population", format="%d")
            }
        )
        
        c1, c2 = st.columns(2)
        c1.download_button("📥 Download GeoJSON (Map)", convert_gdf_to_geojson(final_gdf), f"IPD_{selected_state_name}_Map.geojson", "application/json", use_container_width=True)
        c2.download_button("📥 Download CSV (Data)", convert_df(final_gdf.drop(columns='geometry')), f"IPD_{selected_state_name}_Data.csv", "text/csv", use_container_width=True)

    with tab_stats:
        st.dataframe(summary_stats, use_container_width=True)
        st.download_button("📥 Download Stats CSV", convert_df(summary_stats), f"IPD_{selected_state_name}_Stats.csv", "text/csv")

else:
    st.info("👈 Use the sidebar to configure and run the analysis.")