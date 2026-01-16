import os
import pandas as pd
import geopandas as gpd
from functools import reduce
from ipd_config import *
from census_utils import get_acs_data, calculate_racial_minority_replicates
from geo_utils import get_census_geometry
from analysis_utils import process_indicators, calculate_sd_scores
from export_utils import export_results

def patch_missing_columns(df):
    """
    Renames raw Census columns to internal IPD names (OA_uni, etc.).
    Handles cases where merge operations might have suffixed columns (e.g., _x, _y).
    """
    # 1. Strict mappings for unique variables
    rename_map = {
        'S0101_C01_030E': 'OA_est', 'S0101_C01_030M': 'OA_est_moe',
        'S0101_C05_001E': 'F_est',  'S0101_C05_001M': 'F_est_moe',
        'S1601_C01_001E': 'LEP_uni', 'S1601_C01_001M': 'LEP_uni_moe',
        'S1601_C05_001E': 'LEP_est', 'S1601_C05_001M': 'LEP_est_moe',
        'S1810_C01_001E': 'D_uni',   'S1810_C01_001M': 'D_uni_moe',
        'S1810_C02_001E': 'D_est',   'S1810_C02_001M': 'D_est_moe',
        'S1701_C01_001E': 'LI_uni',  'S1701_C01_001M': 'LI_uni_moe',
        'S1701_C01_042E': 'LI_est',  'S1701_C01_042M': 'LI_est_moe',
        'B09001_001E': 'Y_est', 'B09001_001M': 'Y_est_moe',
        'B01003_001E': 'Y_uni', 'B01003_001M': 'Y_uni_moe', 
        'B02001_002E': 'RM_est', 'B02001_002M': 'RM_est_moe',
        'B02001_001E': 'RM_uni', 'B02001_001M': 'RM_uni_moe',
        'B03002_012E': 'EM_est', 'B03002_012M': 'EM_est_moe',
        'B03002_001E': 'EM_uni', 'B03002_001M': 'EM_uni_moe',
        'B05012_003E': 'FB_est', 'B05012_003M': 'FB_est_moe',
        'B05012_001E': 'FB_uni', 'B05012_001M': 'FB_uni_moe',
        'B08201_002E': 'NC_est', 'B08201_002M': 'NC_est_moe',
        'B08201_001E': 'NC_uni', 'B08201_001M': 'NC_uni_moe',
    }

    # Iterate and rename. If exact match missing, check for suffixes (_x, _y)
    for raw_col, new_col in rename_map.items():
        if raw_col in df.columns:
            df.rename(columns={raw_col: new_col}, inplace=True)
        else:
            for suffix in ['_x', '_y', '_z']:
                suffixed_col = f"{raw_col}{suffix}"
                if suffixed_col in df.columns:
                    if new_col not in df.columns:
                        df.rename(columns={suffixed_col: new_col}, inplace=True)

    # 3. SPECIAL HANDLING FOR SHARED UNIVERSE (S0101_C01_001)
    # The columns may appear as S0101_C01_001E_x or S0101_C01_001E_y in the raw DF.
    
    # We want to fill 'OA_uni' and 'F_uni' with this data.
    possible_cols = ['S0101_C01_001E', 'S0101_C01_001E_x', 'S0101_C01_001E_y']
    possible_cols_moe = ['S0101_C01_001M', 'S0101_C01_001M_x', 'S0101_C01_001M_y']
    
    # Find the first available column that matches
    source_col = next((c for c in possible_cols if c in df.columns), None)
    source_col_moe = next((c for c in possible_cols_moe if c in df.columns), None)
    
    if source_col:
        df['OA_uni'] = df[source_col]
        df['F_uni'] = df[source_col]
    
    if source_col_moe:
        df['OA_uni_moe'] = df[source_col_moe]
        df['F_uni_moe'] = df[source_col_moe]
        
    return df

def fetch_single_indicator_debug(indicator_code, codes, year, state_fips, counties, geo_level):
    import requests
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
    call_url = f"{base_url}/{year}/{endpoint}?get={var_str},NAME{geo_clause}&key={API_KEY}"
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
        print(f"    Failed to fetch {indicator_code}: {e}")
        return pd.DataFrame()

def main():
    print(f"Starting IPD Analysis for Year {YEAR}...")
    
    all_data_frames = []
    all_geoms = []
    census_tables_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "census_tables")
    if not os.path.exists(census_tables_dir):
        os.makedirs(census_tables_dir)

    for area in TARGET_AREAS:
        state = area['state']
        state_fips = area['state_fips']
        counties = area['counties'] 
        print(f"Processing {state} (Counties: {counties if counties else 'All'})...")
        
        indicator_dfs = []
        print("  Fetching indicators individually...")
        for ind_code, codes in ACS_VARIABLES.items():
            if ind_code == 'TOT_POP': continue 
            print(f"    Fetching {ind_code}...")
            df_ind = fetch_single_indicator_debug(ind_code, codes, YEAR, state_fips, counties, GEO_LEVEL)
            if not df_ind.empty:
                fname = f"raw_{state}_{YEAR}_{ind_code}.csv"
                df_ind.to_csv(os.path.join(census_tables_dir, fname), index=False)
                indicator_dfs.append(df_ind)
        
        if not indicator_dfs:
            print("CRITICAL ERROR: No indicators could be fetched. Skipping area.")
            continue

        print("  Merging indicators into master dataframe...")
        merge_keys = ['state', 'county', 'tract', 'NAME']
        if GEO_LEVEL == 'block group':
            merge_keys.append('block group')
            
        df_master = reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how='outer'), indicator_dfs)
        
        if GEO_LEVEL == 'tract':
            df_master['GEOID'] = df_master['state'] + df_master['county'] + df_master['tract']
        elif GEO_LEVEL == 'block group':
            df_master['GEOID'] = df_master['state'] + df_master['county'] + df_master['tract'] + df_master['block group']

        master_raw_path = os.path.join(census_tables_dir, f"acs_raw_{state}_{YEAR}_CONSTRUCTED.csv")
        df_master.to_csv(master_raw_path, index=False)
        
        df_master = patch_missing_columns(df_master)
        
        df_reps = calculate_racial_minority_replicates(YEAR, state_fips, counties)
        if not df_reps.empty:
            print("Merging Variance Replicate MOEs...")
            df_master = df_master.merge(df_reps, on='GEOID', how='left')
            if 'RM_est_moe_replicate' in df_master.columns and 'RM_est_moe' in df_master.columns:
                df_master['RM_est_moe'] = df_master['RM_est_moe_replicate'].fillna(df_master['RM_est_moe'])
        
        all_data_frames.append(df_master)
        
        gdf_state = get_census_geometry(YEAR, state_fips, GEO_LEVEL)
        if counties:
            gdf_state['county_fips'] = gdf_state['GEOID'].str[2:5]
            gdf_state = gdf_state[gdf_state['county_fips'].isin(counties)]
        all_geoms.append(gdf_state)
        
    print("Combining regions...")
    if not all_data_frames:
        return

    full_df = pd.concat(all_data_frames, ignore_index=True)
    full_gdf = pd.concat(all_geoms, ignore_index=True)
    
    print("Calculating Scores...")
    full_df = process_indicators(full_df)
    
    indicators = ['Y', 'OA', 'F', 'RM', 'EM', 'FB', 'LEP', 'D', 'LI', 'NC']
    full_df_scored, summary_stats = calculate_sd_scores(full_df, indicators)
    
    print("Merging Spatial Data...")
    final_gdf = full_gdf.merge(full_df_scored, on='GEOID', how='inner')
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Exporting Summary Statistics to {OUTPUT_DIR}...")
    summary_stats.to_csv(os.path.join(OUTPUT_DIR, f"IPD_Summary_Stats_{YEAR}.csv"), index=False)

    print(f"Exporting Geospatial Results to {OUTPUT_DIR}...")
    full_df_scored.to_csv(os.path.join(OUTPUT_DIR, f"IPD_Results_Attributes_{YEAR}.csv"), index=False)
    
    export_results(
        final_gdf, 
        OUTPUT_DIR, 
        f"IPD_Results_{YEAR}", 
        formats=['gpkg', 'parquet', 'geojson'] 
    )
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()