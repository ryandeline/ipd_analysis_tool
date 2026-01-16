import os
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

def export_results(gdf, output_dir, base_name, formats=['gpkg']):
    """
    Exports the analyzed GeoDataFrame to multiple robust storage formats.
    
    Args:
        gdf (GeoDataFrame): The data to save.
        output_dir (str): Folder to save to.
        base_name (str): Filename without extension (e.g., 'IPD_Results_2023').
        formats (list): List of formats to save: ['gpkg', 'parquet', 'geojson', 'postgis']
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path_base = os.path.join(output_dir, base_name)
    
    # ---------------------------------------------------------
    # 1. GeoPackage (Recommended Default)
    # ---------------------------------------------------------
    if 'gpkg' in formats:
        print(f"Saving GeoPackage: {path_base}.gpkg...")
        # layer name is typically the base name
        gdf.to_file(f"{path_base}.gpkg", driver="GPKG", layer=base_name)
        
    # ---------------------------------------------------------
    # 2. GeoParquet (Recommended for Analytics/Cloud)
    # ---------------------------------------------------------
    if 'parquet' in formats:
        print(f"Saving GeoParquet: {path_base}.parquet...")
        # Requires pyarrow: pip install pyarrow
        try:
            gdf.to_parquet(f"{path_base}.parquet")
        except ImportError:
            print("Error: 'pyarrow' library required for Parquet export.")
            print("Run: pip install pyarrow")

    # ---------------------------------------------------------
    # 3. GeoJSON (Web/Legacy)
    # ---------------------------------------------------------
    if 'geojson' in formats:
        print(f"Saving GeoJSON: {path_base}.geojson...")
        gdf.to_file(f"{path_base}.geojson", driver="GeoJSON")

    # ---------------------------------------------------------
    # 4. PostGIS (Enterprise/Database)
    # ---------------------------------------------------------
    if 'postgis' in formats:
        print("Uploading to PostGIS...")
        # Requires environment variables or config for DB connection
        # DB_CONN_STR = "postgresql://user:password@localhost:5432/mydb"
        db_url = os.getenv("DATABASE_URL")
        
        if db_url:
            try:
                engine = create_engine(db_url)
                # Save to database
                gdf.to_postgis(name=base_name.lower(), con=engine, if_exists='replace')
                print("Upload success.")
            except Exception as e:
                print(f"PostGIS upload failed: {e}")
        else:
            print("Skipping PostGIS: DATABASE_URL environment variable not set.")

    print("Export Complete.")