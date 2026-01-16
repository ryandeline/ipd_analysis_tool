import geopandas as gpd
import pandas as pd
from shapely.geometry import box

def get_census_geometry(year, state_fips, geo_level):
    """
    Fetches Census TIGER/Line shapefiles for the given state and level.
    Uses Geopandas to read directly from the Census Bureau FTP/Web interface.
    """
    base_url = "https://www2.census.gov/geo/tiger"
    
    # TIGER URL structure changes occasionally. 
    # For 2023: TIGER2023/TRACT/tl_2023_{state}_tract.zip
    
    layer_name = "TRACT" if geo_level == 'tract' else "BG"
    file_code = "tract" if geo_level == 'tract' else "bg"
    
    url = f"{base_url}/TIGER{year}/{layer_name}/tl_{year}_{state_fips}_{file_code}.zip"
    
    print(f"Fetching geometry from: {url}")
    
    try:
        gdf = gpd.read_file(url)
        
        # Standardize GEOID column
        gdf['GEOID'] = gdf['GEOID']
        
        # Ensure CRS is projected for accurate mapping later (e.g., Albers USA or State Plane)
        # Using EPSG:4269 (NAD83) comes default. 
        # For analysis we might keep it lat/lon, but for area calcs we project.
        # We will return original CRS but ensure column naming is clean.
        
        keep_cols = ['GEOID', 'geometry', 'ALAND', 'AWATER']
        return gdf[keep_cols]
        
    except Exception as e:
        print(f"Error fetching geometry: {e}")
        return gpd.GeoDataFrame()