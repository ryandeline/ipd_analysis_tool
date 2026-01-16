"""
Configuration settings for the Indicators of Potential Disadvantage (IPD) Analysis.
"""
import os

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS
# ---------------------------------------------------------------------------
# Census API Key (Get one at http://api.census.gov/data/key_signup.html)
# Leave empty to rely on environment variable 'CENSUS_API_KEY'
API_KEY = os.getenv("CENSUS_API_KEY", "dfb115d4ff6b35a8ccc01892add4258ba7b48eaf")

# Analysis Year (Use 2022 or 2023 for latest available ACS 5-Year)
YEAR = 2023

# Geography Level: 'tract' or 'block group'
# Note: ACS 5-Year data is most reliable at the Tract level. 
# Block Group data has higher margins of error.
GEO_LEVEL = 'tract' 

# ---------------------------------------------------------------------------
# GEOGRAPHY SELECTION
# ---------------------------------------------------------------------------
# List of states (abbreviations) and specific counties (FIPS codes) to analyze.
# If counties list is empty [], it will fetch all counties for the state.
TARGET_AREAS = [
    {
        "state": "DC",
        "counties": ["001"], # District of Columbia
        "state_fips": "11"
    },
    # Example of adding PA/NJ counties dynamically:
    # {
    #     "state": "IN",
    #     "counties": ["039", "085", "099", "141"], # Elkhart, Kosciusko, Marshall, St. Joseph
    #     "state_fips": "18"
    # }
]

# Output Directory
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# ACS VARIABLES MAP
# ---------------------------------------------------------------------------
# Mapping IPD Indicators to ACS Table IDs.
# Format: 'Internal_Name': {'E': 'Estimate_Variable', 'U': 'Universe_Variable'}

ACS_VARIABLES = {
    # Total Population (Universe for many)
    'TOT_POP': 'B01003_001', 
    
    # 1. Youth (Under 18)
    # Count: B09001_001 (Total Pop Under 18)
    # Universe: B01003_001 (Total Population)
    'Y': {'count': 'B09001_001', 'universe': 'B01003_001'},

    # 2. Older Adults (65+)
    # Count: S0101_C01_030 (Total 65+)
    # Universe: S0101_C01_001 (Total Population)
    'OA': {'count': 'S0101_C01_030', 'universe': 'S0101_C01_001'},

    # 3. Female
    # Count: S0101_C05_001 (Total Female)
    # Universe: S0101_C01_001 (Total Population)
    'F': {'count': 'S0101_C05_001', 'universe': 'S0101_C01_001'},

    # 4. Racial Minority (Non-White)
    # Calculation: Universe (Total) - White Alone
    # Count: B02001_002 (White Alone) -> Used to subtract
    # Universe: B02001_001 (Total Population)
    'RM': {'count': 'B02001_002', 'universe': 'B02001_001', 'calc_type': 'subtract'},

    # 5. Ethnic Minority (Hispanic)
    # Count: B03002_012 (Hispanic or Latino)
    # Universe: B03002_001 (Total Population)
    'EM': {'count': 'B03002_012', 'universe': 'B03002_001'},

    # 6. Foreign Born
    # Count: B05012_003 (Foreign Born)
    # Universe: B05012_001 (Total Population)
    'FB': {'count': 'B05012_003', 'universe': 'B05012_001'},

    # 7. Limited English Proficiency (LEP)
    # Count: S1601_C05_001 (Speak English less than "very well")
    # Universe: S1601_C01_001 (Population 5 years and over)
    'LEP': {'count': 'S1601_C05_001', 'universe': 'S1601_C01_001'},

    # 8. Disabled
    # Count: S1810_C02_001 (With a disability)
    # Universe: S1810_C01_001 (Total civilian noninstitutionalized pop)
    'D': {'count': 'S1810_C02_001', 'universe': 'S1810_C01_001'},

    # 9. Low Income
    # Count: S1701_C01_042 (Population below 200% poverty level)
    # Universe: S1701_C01_001 (Population for whom poverty status is determined)
    'LI': {'count': 'S1701_C01_042', 'universe': 'S1701_C01_001'},

    # 10. Carless Populations (No Vehicle Available)
    # Count: B08201_002 (No vehicle available)
    # Universe: B08201_001 (Occupied housing units)
    'NC': {'count': 'B08201_002', 'universe': 'B08201_001'}
}