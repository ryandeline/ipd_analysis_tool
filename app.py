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
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="IPD Analyzer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for layout control
st.markdown("""
<style>
    /* Adjust main container padding to position header correctly */
    .block-container {
        padding-top: 3rem; 
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Compact metrics */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    /* Lift bottom section closer to header */
    h3 {
        margin-top: -1rem;
        padding-top: 0rem;
        font-size: 1.3rem;
    }
    hr {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    /* Center the processing status widget */
    div[data-testid="stStatusWidget"] {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        width: 400px;
    }
</style>
""", unsafe_allow_html=True)

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

# --- VARIABLES MAPS ---
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
    
    # Chunking logic for long URLs
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
        
    available_est_cols = [f"{c}E" for c in count_vars if f"{c}E" in df_final.columns]
    df_final[f"{indicator_code}_est"] = df_final[available_est_cols].sum(axis=1) if available_est_cols else 0
    
    available_moe_cols = [f"{c}M" for c in count_vars if f"{c}M" in df_final.columns]
    df_final[f"{indicator_code}_est_moe"] = np.sqrt((df_final[available_moe_cols] ** 2).sum(axis=1)) if available_moe_cols else 0
    
    if u_var:
        if f"{u_var}E" in df_final.columns: df_final[f"{indicator_code}_uni"] = df_final[f"{u_var}E"]
        else: df_final[f"{indicator_code}_uni"] = 0
        if f"{u_var}M" in df_final.columns: df_final[f"{indicator_code}_uni_moe"] = df_final[f"{u_var}M"]
        else: df_final[f"{indicator_code}_uni_moe"] = 0
    
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
    score_cols = []
    
    for ind in indicators:
        pct_col = f'{ind}_pct'
        if pct_col not in df.columns: continue
        
        mean_val = df_scored[pct_col].mean()
        sd_val = df_scored[pct_col].std()
        
        if pd.isna(mean_val) or pd.isna(sd_val) or sd_val == 0:
            stats_list.append({'Indicator': ind, 'Mean': 0, 'SD': 0})
            df_scored[f'{ind}_score'] = 0
            df_scored[f'{ind}_class'] = "Insufficient Data"
            continue

        b1 = max(0, mean_val - (1.5 * sd_val))
        b2 = mean_val - (0.5 * sd_val)
        b3 = mean_val + (0.5 * sd_val)
        b4 = mean_val + (1.5 * sd_val)
        
        def get_score_and_class(val, breaks):
            b1, b2, b3, b4 = breaks
            if val < b1: return 0, "Well Below Average"
            if val < b2: return 1, "Below Average"
            if val < b3: return 2, "Average"
            if val < b4: return 3, "Above Average"
            return 4, "Well Above Average"
            
        results = df_scored[pct_col].apply(lambda x: get_score_and_class(x, (b1, b2, b3, b4)))
        score_col = f'{ind}_score'
        class_col = f'{ind}_class'
        df_scored[score_col] = results.apply(lambda x: x[0]).astype(int)
        df_scored[class_col] = results.apply(lambda x: x[1])
        df_scored['IPD_SCORE'] += df_scored[score_col]
        score_cols.append(score_col)
        
        stats_list.append({
            'Indicator': ind, 'Mean': round(mean_val, 1), 'SD': round(sd_val, 1),
            'Break_Min_1.5SD': round(b1, 1), 'Break_Min_0.5SD': round(b2, 1),
            'Break_Plus_0.5SD': round(b3, 1), 'Break_Plus_1.5SD': round(b4, 1)
        })

    ipd_col = 'IPD_SCORE'
    if ipd_col in df_scored.columns:
        mean_ipd = df_scored[ipd_col].mean()
        sd_ipd = df_scored[ipd_col].std()
        
        if not pd.isna(mean_ipd) and not pd.isna(sd_ipd) and sd_ipd > 0:
            ib1 = max(0, mean_ipd - (1.5 * sd_ipd))
            ib2 = mean_ipd - (0.5 * sd_ipd)
            ib3 = mean_ipd + (0.5 * sd_ipd)
            ib4 = mean_ipd + (1.5 * sd_ipd)
            
            comp_results = df_scored[ipd_col].apply(lambda x: get_score_and_class(x, (ib1, ib2, ib3, ib4)))
            df_scored['IPD_SCORE_score'] = comp_results.apply(lambda x: x[0]).astype(int)
            df_scored['IPD_SCORE_class'] = comp_results.apply(lambda x: x[1])
            
            stats_list.append({
                'Indicator': 'IPD_SCORE_COMPOSITE', 'Mean': round(mean_ipd, 1), 'SD': round(sd_ipd, 1),
                'Break_Min_1.5SD': round(ib1, 1), 'Break_Min_0.5SD': round(ib2, 1),
                'Break_Plus_0.5SD': round(ib3, 1), 'Break_Plus_1.5SD': round(ib4, 1)
            })
        
        if score_cols:
            df_scored['indicators_mean'] = df_scored[score_cols].mean(axis=1)
            df_scored['indicators_std'] = df_scored[score_cols].std(axis=1)
            def get_confidence(row):
                if row['indicators_mean'] == 0: return 'High'
                cv = row['indicators_std'] / row['indicators_mean']
                if cv < 0.5: return 'High'
                elif cv < 1.0: return 'Medium'
                else: return 'Low'
            df_scored['IPD_CONFIDENCE'] = df_scored.apply(get_confidence, axis=1)
            stats_list.append({
                'Indicator': 'ROW_WISE_CONFIDENCE',
                'Mean': round(df_scored['indicators_mean'].mean(), 1),
                'SD': round(df_scored['indicators_mean'].std(), 1)
            })
        
    return df_scored, pd.DataFrame(stats_list)

def run_analysis(api_key, state_fips, state_name, counties, year, geo_level):
    status = st.status(f"Analysing {state_name} ({geo_level})...", expanded=True)
    
    VARIABLES = ACS_VARS_BG if geo_level == 'block group' else ACS_VARS_TRACT
    active_inds = [k for k in VARIABLES.keys() if k != 'TOT_POP']

    indicator_dfs = []
    total_steps = len(active_inds) + 3 
    
    status.write(f"Fetching Indicator: TOT_POP (1/{total_steps})")
    df_pop = fetch_single_indicator('TOT_POP', VARIABLES['TOT_POP'], year, state_fips, counties, geo_level, api_key)
    if not df_pop.empty: indicator_dfs.append(df_pop)
    
    for i, ind_code in enumerate(active_inds):
        codes = VARIABLES[ind_code]
        desc = "Fetching"
        if isinstance(codes.get('count'), list): desc = "Summing tables for"
        elif codes.get('interpolate'): desc = "Interpolating tract data for"
        status.write(f"{desc} Indicator: {ind_code} ({i+2}/{total_steps})")
        df_ind = fetch_single_indicator(ind_code, VARIABLES[ind_code], year, state_fips, counties, geo_level, api_key)
        if not df_ind.empty: indicator_dfs.append(df_ind)

    if not indicator_dfs:
        st.error("No data fetched. Check API key.")
        st.stop()
    
    status.write("Merging datasets...")
    bg_dfs = [df for df in indicator_dfs if 'block group' in df.columns]
    tract_dfs = [df for df in indicator_dfs if 'block group' not in df.columns]
    
    if geo_level == 'block group' and not bg_dfs:
        st.error("Critical Error: No Block Group data fetched.")
        st.stop()
        
    common_keys = list(set.intersection(*(set(df.columns) for df in (bg_dfs if bg_dfs else tract_dfs))))
    base_keys = [k for k in ['state', 'county', 'tract', 'block group'] if k in common_keys]
    
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

    status.write("Calculating standard deviation scores (Mean ± 1.5 SD)...")
    df_master = process_indicators(df_master, active_inds)
    full_df_scored, summary_stats = calculate_sd_scores(df_master, active_inds)

    status.write("Applying geometry...")
    gdf_geom = get_census_geometry(year, state_fips, geo_level)
    if counties: gdf_geom = gdf_geom[gdf_geom['GEOID'].str[2:5].isin(counties)]
    
    final_gdf = gdf_geom.merge(full_df_scored, on='GEOID', how='inner')
    
    # Remove status widget to free up space
    status.empty()
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
    
    # 1. Header Metrics Placeholder
    metrics_container = st.container() 
    st.divider()

    # Layout Columns
    col_data, col_map = st.columns([1, 1])

    # 2. Data Table (Left Column)
    with col_data:
        tab_data, tab_stats = st.tabs(["📋 Detailed Data Table", "📈 Summary Statistics"])
        
        with tab_data:
            # Prepare display data
            base_cols = ['GEOID', 'IPD_SCORE_score', 'IPD_SCORE', 'IPD_SCORE_class', 'IPD_CONFIDENCE', 'TOT_POP_est']
            pct_cols = [c for c in final_gdf.columns if c.endswith('_pct')]
            score_cols = [c for c in final_gdf.columns if c.endswith('_score') and c != 'IPD_SCORE_score']
            display_cols = base_cols + sorted(pct_cols + score_cols)
            display_cols = [c for c in display_cols if c in final_gdf.columns]
            
            display_df = final_gdf[display_cols].copy()
            
            # --- Check Map Click ---
            map_click_geoid = None
            if st.session_state.get("map_last_click"):
                click_data = st.session_state["map_last_click"]
                if click_data:
                    map_click_geoid = click_data.get("properties", {}).get("GEOID")

            # Filter table if map clicked
            if map_click_geoid:
                table_data = display_df[display_df['GEOID'] == map_click_geoid].copy()
                st.caption(f"Filtered by Map Click: {map_click_geoid}")
            else:
                table_data = display_df.copy()

            col_config = {
                "IPD_SCORE_score": st.column_config.ProgressColumn(
                    "Normalized IPD Score (0-4)", help="Composite Disadvantage Score (0-4 Scale)",
                    format="%d", min_value=0, max_value=4,
                ),
                "TOT_POP_est": st.column_config.NumberColumn("Population", format="%d")
            }

            selection = st.dataframe(
                table_data,
                use_container_width=True,
                column_config=col_config,
                height=450, 
                on_select="rerun",
                selection_mode="multi-row",
                key="table_selection"
            )
    
    # 3. Map (Right Column)
    with col_map:
        # Determine 'active' data based on table selection OR map click
        if selection.selection.rows:
            selected_indices = selection.selection.rows
            selected_geoids = table_data.iloc[selected_indices]['GEOID'].tolist()
            map_data = final_gdf[final_gdf['GEOID'].isin(selected_geoids)].copy()
            active_df_stats = map_data
        elif map_click_geoid:
             map_data = final_gdf[final_gdf['GEOID'] == map_click_geoid].copy()
             active_df_stats = map_data
        else:
            map_data = final_gdf.copy()
            active_df_stats = final_gdf

        # Helpers
        @st.cache_data
        def convert_df(df): return df.to_csv(index=False).encode('utf-8')
        @st.cache_data
        def convert_gdf_to_geojson(_gdf): return _gdf.to_json()

        # Download Buttons (Above Map)
        c1, c2 = st.columns(2)
        c1.download_button("📥 Download GeoJSON", convert_gdf_to_geojson(final_gdf), f"IPD_{selected_state_name}.geojson", "application/json", use_container_width=True, key="btn_geo")
        c2.download_button("📥 Download CSV", convert_df(final_gdf.drop(columns='geometry')), f"IPD_{selected_state_name}.csv", "text/csv", use_container_width=True, key="btn_csv")

        # Render Map
        if not map_data.empty:
            center_lat = map_data.geometry.centroid.y.mean()
            center_lon = map_data.geometry.centroid.x.mean()
            m = folium.Map(location=[center_lat, center_lon])
            
            if map_data.crs != 'EPSG:4326': map_data = map_data.to_crs('EPSG:4326')
            min_lon, min_lat, max_lon, max_lat = map_data.total_bounds
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        else:
            m = folium.Map(location=[39.8, -98.5], zoom_start=4)

        map_col = 'IPD_SCORE_score' if 'IPD_SCORE_score' in final_gdf.columns else 'IPD_SCORE'
        
        choropleth = folium.Choropleth(
            geo_data=map_data.to_json(),
            data=map_data,
            columns=['GEOID', map_col],
            key_on='feature.properties.GEOID',
            fill_color='YlOrRd',
            legend_name='IPD Score (0-4)',
            name='IPD Scores'
        )
        choropleth.add_to(m)
        choropleth.geojson.add_child(folium.features.GeoJsonTooltip(fields=['GEOID', map_col], labels=True))
        
        st_folium(m, width="100%", height=450, key="map_last_click", returned_objects=["last_object_clicked"])

    # 4. Update Stats Tab (Based on Active Selection)
    with tab_stats:
        if not active_df_stats.empty:
            subset_stats = []
            indicators = [c.replace('_pct', '') for c in active_df_stats.columns if c.endswith('_pct')]
            
            for ind in indicators:
                pct_col = f'{ind}_pct'
                subset_stats.append({
                    'Indicator': ind,
                    'Mean': round(active_df_stats[pct_col].mean(), 1),
                    'SD': round(active_df_stats[pct_col].std(), 1)
                })
            
            if 'IPD_SCORE' in active_df_stats.columns:
                 subset_stats.append({
                    'Indicator': 'Composite',
                    'Mean': round(active_df_stats['IPD_SCORE'].mean(), 1),
                    'SD': round(active_df_stats['IPD_SCORE'].std(), 1)
                })
            
            stats_df = pd.DataFrame(subset_stats)
            st.dataframe(stats_df, use_container_width=True, height=450)
            st.download_button("📥 Download Stats", convert_df(stats_df), "IPD_Stats_Subset.csv", "text/csv", key="btn_stat_sub")
        else:
            st.write("No data selected.")

    # 5. Update Header Metrics
    with metrics_container:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Geographic Level", geo_level.title())
        m2.metric("Selected Units", f"{len(active_df_stats):,}")
        
        pop = active_df_stats['TOT_POP_est'].sum() if 'TOT_POP_est' in active_df_stats.columns else 0
        m3.metric("Population (Selected)", f"{int(pop):,}")
        
        if 'IPD_SCORE_score' in active_df_stats.columns:
            avg = active_df_stats['IPD_SCORE_score'].mean()
            m4.metric("Avg Score (Selected)", f"{avg:.2f}" if not pd.isna(avg) else "0.0")
        else:
            m4.metric("Avg Score", f"{active_df_stats['IPD_SCORE'].mean():.1f}")

else:
    st.info("👈 Use the sidebar to configure and run the analysis.")