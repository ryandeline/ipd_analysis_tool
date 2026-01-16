# **otbx\_ipd: Indicators of Potential Disadvantage Analysis**

**otbx\_ipd** is a Python-based analytical engine designed to identify and locate populations of interest under federal equity mandates (Title VI, Environmental Justice) and strategic market analysis.

It automates the extraction of **American Community Survey (ACS) 5-Year Estimates**, calculates statistical concentrations using a **Standard Deviation (SD) Scoring** methodology, and exports robust geospatial datasets for planning and intelligence.

## **🚀 Key Features**

* **Automated Census Retrieval:** Fetches variables for 10 key demographic indicators (Youth, Seniors, Low-Income, Carless, etc.) directly from the Census API.  
* **Statistical Precision:** Uses **Variance Replicate Tables** to calculate accurate Margins of Error (MOE) for complex derived variables (e.g., Racial Minority), ensuring defensible data.  
* **Relative Scoring:** Scores every Census Tract from 0 (Well Below Average) to 4 (Well Above Average) based on regional standard deviations.  
* **Modern Data Engineering:** Exports results to enterprise-grade formats including **GeoPackage** (.gpkg) and **GeoParquet** (.parquet) to overcome legacy Shapefile limitations.

## **📦 Installation**

### **1\. Prerequisites**

* Python 3.9 or higher  
* A U.S. Census Bureau API Key ([Get one here](http://api.census.gov/data/key_signup.html))

### **2\. Install Dependencies**

This project relies on modern geospatial and data analysis libraries. Install the required packages using the provided requirements.txt:

pip install \-r requirements.txt

**Core Dependencies:**

* pandas & numpy: Data manipulation and statistical calculations.  
* geopandas & shapely: Spatial data handling and geometric operations.  
* requests: HTTP library for interacting with the Census API.  
* pyarrow: High-performance engine for reading/writing Parquet files.  
* sqlalchemy: SQL toolkit for database interactions (PostGIS support).

## **⚙️ Configuration**

All project settings are managed in ipd\_config.py.

### **1\. Set API Key**

You can set your key as an environment variable (recommended) or paste it directly into the config file.

\# In ipd\_config.py  
API\_KEY \= "YOUR\_CENSUS\_API\_KEY\_HERE"

### **2\. Define Geography**

Edit the TARGET\_AREAS list in ipd\_config.py to define which regions to analyze. You can mix multiple states and specific counties.

**Example for Indiana (Elkhart, Kosciusko, Marshall, and St. Joseph counties):**

TARGET\_AREAS \= \[  
    {  
        "state": "IN",  
        "counties": \["039", "085", "099", "141"\], \# Elkhart, Kosciusko, Marshall, St. Joseph  
        "state\_fips": "18"  
    }  
\]

### **3\. Adjust Analysis Year**

Set the YEAR variable to the desired ACS 5-Year data vintage (e.g., 2023).

## **🏃 Usage**

To run the full analysis workflow:

python run\_ipd.py

**What happens when you run this:**

1. **Fetch:** Downloads raw demographic counts and geometry for defined areas.  
2. **Process:** Calculates Percentages, Margins of Error, and Variance Replicates.  
3. **Score:** Calculates Regional Means and Standard Deviations; assigns SD scores (0-4).  
4. **Export:** Saves final datasets to the output/ directory.

## **📂 Output**

The script generates the following files in the output/ folder:

| File | Description |
| :---- | :---- |
| **IPD\_Results\_YYYY.gpkg** | **Primary Output.** A single GeoPackage containing all geometries and attributes. No column truncation. Ideal for QGIS/ArcGIS. |
| **IPD\_Results\_YYYY.parquet** | **Analytics Output.** A highly compressed GeoParquet file. Ideal for Python/R analysis or cloud storage. |
| **IPD\_Results\_YYYY.geojson** | **Web Output.** A standard GeoJSON file for web mapping applications. |
| **IPD\_Summary\_Stats\_YYYY.csv** | **Report.** A CSV listing the regional Mean and SD breaks for every indicator. |

## **📊 Indicators**

The tool analyzes the following 10 populations:

1. **Y:** Youth (Under 18\)  
2. **OA:** Older Adults (65+)  
3. **F:** Female  
4. **RM:** Racial Minority (Non-White)  
5. **EM:** Ethnic Minority (Hispanic)  
6. **FB:** Foreign Born  
7. **LEP:** Limited English Proficiency  
8. **D:** Disabled  
9. **LI:** Low Income (\<200% Poverty)  
10. **NC:** Carless (No Vehicle Available)

## **📁 Project Structure**

otbx\_ipd/  
├── ipd\_config.py      \# Configuration (Key, Geography, Year)  
├── census\_utils.py    \# API Fetching & Variance Replicates  
├── geo\_utils.py       \# TIGER/Line Shapefile Downloader  
├── analysis\_utils.py  \# Scoring & Math Logic  
├── export\_utils.py    \# GeoPackage/Parquet Export Logic  
├── run\_ipd.py         \# Main Execution Script  
└── output/            \# Generated Results  
