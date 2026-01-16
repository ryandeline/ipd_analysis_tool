import requests
import pandas as pd
import numpy as np
import io
import zipfile
from ipd_config import ACS_VARIABLES, API_KEY

def get_acs_data(year, state_fips, county_fips, geo_level):
    """
    Fetches ACS data for all variables defined in config.
    Returns a clean DataFrame with Estimates (E) and Margins of Error (M).
    """
    base_url = "https://api.census.gov/data"
    
    # Construct list of variables to fetch
    vars_to_fetch = []
    col_map = {}
    
    for indicator, codes in ACS_VARIABLES.items():
        if indicator == 'TOT_POP': continue 
        
        # Add Count Variables
        c_var = codes['count']
        vars_to_fetch.extend([f"{c_var}E", f"{c_var}M"])
        col_map[f"{c_var}E"] = f"{indicator}_est"
        col_map[f"{c_var}M"] = f"{indicator}_est_moe"
        
        # Add Universe Variables
        u_var = codes['universe']
        vars_to_fetch.extend([f"{u_var}E", f"{u_var}M"])
        col_map[f"{u_var}E"] = f"{indicator}_uni"
        col_map[f"{u_var}M"] = f"{indicator}_uni_moe"

    vars_to_fetch = list(set(vars_to_fetch))
    
    # Split Subject (S*) vs Detail (B*) variables
    subject_vars = [v for v in vars_to_fetch if v.startswith('S') or v.startswith('DP')]
    detail_vars = [v for v in vars_to_fetch if v not in subject_vars]
    
    # Create Batches (Limit 45 vars per call)
    chunk_size = 45
    batches = []
    
    for i in range(0, len(subject_vars), chunk_size):
        batches.append({'vars': subject_vars[i:i+chunk_size], 'type': 'subject'})
    
    for i in range(0, len(detail_vars), chunk_size):
        batches.append({'vars': detail_vars[i:i+chunk_size], 'type': 'detail'})
        
    dfs = []
    
    for batch in batches:
        chunk = batch['vars']
        if not chunk: continue
        
        var_str = ",".join(chunk)
        
        # Geography clause
        if county_fips:
            county_str = ",".join(county_fips)
            geo_clause = f"&for={geo_level}:*&in=state:{state_fips}&in=county:{county_str}"
        else:
            geo_clause = f"&for={geo_level}:*&in=state:{state_fips}"
            
        # Select Endpoint
        if batch['type'] == 'subject':
            call_url = f"{base_url}/{year}/acs/acs5/subject?get={var_str},NAME{geo_clause}&key={API_KEY}"
        else:
            call_url = f"{base_url}/{year}/acs/acs5?get={var_str},NAME{geo_clause}&key={API_KEY}"
            
        try:
            r = requests.get(call_url)
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            dfs.append(df)
        except Exception as e:
            print(f"Error fetching batch ({batch['type']}): {e}")
            print(f"URL: {call_url}")

    if not dfs:
        return pd.DataFrame()
        
    # Merge all frames
    final_df = dfs[0]
    for df in dfs[1:]:
        # Identify common columns to merge on (Geographic identifiers)
        merge_cols = ['state', 'county', 'tract', 'NAME']
        if geo_level == 'block group':
            merge_cols.append('block group')
        
        # Filter strictly for merge columns + new data columns
        cols_to_use = merge_cols + [c for c in df.columns if c not in final_df.columns]
        
        final_df = final_df.merge(df[cols_to_use], on=merge_cols, how='outer')

    # Construct GEOID
    if geo_level == 'tract':
        final_df['GEOID'] = final_df['state'] + final_df['county'] + final_df['tract']
    elif geo_level == 'block group':
        final_df['GEOID'] = final_df['state'] + final_df['county'] + final_df['tract'] + final_df['block group']

    # Rename columns
    final_df.rename(columns=col_map, inplace=True)
    
    # Convert to numeric
    for col in final_df.columns:
        if col.endswith('_est') or col.endswith('_moe') or col.endswith('_uni'):
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)

    return final_df

def calculate_racial_minority_replicates(year, state_fips, target_counties):
    """
    Downloads Variance Replicate Tables for B02001 to calculate precise MOE for 
    Racial Minority. Handles GEOID string/float issues.
    """
    print(f"Downloading Variance Replicates for State FIPS {state_fips}...")
    
    url = f"https://www2.census.gov/programs-surveys/acs/replicate_estimates/{year}/data/5-year/140/B02001_{state_fips}.csv.zip"
    
    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"Variance Replicate table not found for {state_fips} (Status {r.status_code})")
            return pd.DataFrame()

        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # Robust CSV finding
        csv_name = None
        for name in z.namelist():
            if name.endswith('.csv'):
                csv_name = name
                break
        
        if not csv_name:
            print("No CSV found in archive.")
            return pd.DataFrame()

        # Read CSV with explicit dtype for GEOID to prevent float conversion
        df = pd.read_csv(z.open(csv_name), dtype={'GEOID': str})
        
        # Ensure GEOID is treated as string, removing any potential float artifacts
        df['GEOID'] = df['GEOID'].astype(str)
        
        # Handle GEOID parsing
        if df['GEOID'].iloc[0].startswith('1400000US'):
             df['geoid_short'] = df['GEOID'].str.split('US').str[1]
        else:
             # Fallback for bare FIPS
             df['geoid_short'] = df['GEOID']
             
        df['county_fips'] = df['geoid_short'].str[2:5]
        
        if target_counties:
            df = df[df['county_fips'].isin(target_counties)]
        
        target_tables = [
            "Black or African American alone", 
            "American Indian and Alaska Native alone", 
            "Asian alone", 
            "Native Hawaiian and Other Pacific Islander alone", 
            "Some other race alone", 
            "Two or more races:"
        ]
        
        df_target = df[df['TITLE'].isin(target_tables)]
        
        # Sum Estimates and Replicates
        rep_cols = [c for c in df.columns if c.startswith('Var_Rep')]
        grouped = df_target.groupby('geoid_short')[['ESTIMATE'] + rep_cols].sum()
        
        # Calculate MOE: 1.645 * sqrt(0.05 * sum((Rep - Est)^2))
        def calc_variance(row):
            estimate = row['ESTIMATE']
            reps = row[rep_cols]
            sq_diff = (reps - estimate) ** 2
            return np.sqrt(0.05 * sq_diff.sum()) * 1.645

        grouped['RM_est_moe_replicate'] = grouped.apply(calc_variance, axis=1)
        
        return grouped[['RM_est_moe_replicate']].reset_index().rename(columns={'geoid_short': 'GEOID'})

    except Exception as e:
        print(f"Failed to process variance replicates: {e}")
        return pd.DataFrame()