# Collect stats
        stats_list.append({
            'Indicator': ind,
            'Mean': round(mean_val, 1),
            'SD': round(sd_val, 1),
            'Break_Min_1.5SD': round(b1, 1),
            'Break_Min_0.5SD': round(b2, 1),
            'Break_Plus_0.5SD': round(b3, 1),
            'Break_Plus_1.5SD': round(b4, 1)
        })

    # 2. Score the Composite IPD_SCORE (Comparison Logic)
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
                'Indicator': 'IPD_SCORE_COMPOSITE',
                'Mean': round(mean_ipd, 1),
                'SD': round(sd_ipd, 1),
                'Break_Min_1.5SD': round(ib1, 1),
                'Break_Min_0.5SD': round(ib2, 1),
                'Break_Plus_0.5SD': round(ib3, 1),
                'Break_Plus_1.5SD': round(ib4, 1)
            })
        
        # 3. Confidence Check (Row-Level Consistency)
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
    # Fetch TOT_POP separately to ensure it exists
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

    # --- INTERACTION STATE MANAGEMENT ---
    # Check map click state
    map_click_geoid = None
    if st.session_state.get("map_last_click"):
        click_payload = st.session_state["map_last_click"]
        if click_payload and "last_object_clicked" in click_payload:
             obj = click_payload["last_object_clicked"]
             if obj and "properties" in obj:
                 map_click_geoid = obj["properties"].get("GEOID")

    # 2. Data Table (Left Column)
    with col_data:
        tab_data, tab_stats = st.tabs(["📋 Detailed Data Table", "📈 Summary Statistics"])
        
        with tab_data:
            base_cols = ['GEOID', 'IPD_SCORE_score', 'IPD_SCORE', 'IPD_SCORE_class', 'IPD_CONFIDENCE', 'TOT_POP_est']
            pct_cols = [c for c in final_gdf.columns if c.endswith('_pct')]
            score_cols = [c for c in final_gdf.columns if c.endswith('_score') and c != 'IPD_SCORE_score']
            display_cols = base_cols + sorted(pct_cols + score_cols)
            display_cols = [c for c in display_cols if c in final_gdf.columns]
            
            table_source_df = final_gdf[display_cols].copy()
            
            # ...UNLESS Map was clicked.
            if map_click_geoid:
                table_display_df = table_source_df[table_source_df['GEOID'] == map_click_geoid]
                st.caption(f"📍 Filtered by Map Selection: **{map_click_geoid}**")
                if st.button("Clear Map Selection", key="clear_map"):
                    st.session_state["map_reset_token"] = st.session_state.get("map_reset_token", 0) + 1
                    st.rerun()
            else:
                table_display_df = table_source_df

            col_config = {
                "IPD_SCORE_score": st.column_config.ProgressColumn(
                    "Normalized IPD Score (0-4)", help="Composite Disadvantage Score (0-4 Scale)",
                    format="%d", min_value=0, max_value=4,
                ),
                "TOT_POP_est": st.column_config.NumberColumn("Population", format="%d")
            }

            selection = st.dataframe(
                table_display_df,
                use_container_width=True,
                column_config=col_config,
                height=450, 
                on_select="rerun",
                selection_mode="multi-row",
                key="table_selection"
            )
    
    # --- DETERMINE FINAL ACTIVE DATASET ---
    if selection.selection.rows:
        selected_indices = selection.selection.rows
        # Map indices back to GEOIDs from displayed dataframe
        selected_geoids = table_display_df.iloc[selected_indices]['GEOID'].tolist()
        active_df_stats = final_gdf[final_gdf['GEOID'].isin(selected_geoids)].copy()
    elif map_click_geoid:
        active_df_stats = final_gdf[final_gdf['GEOID'] == map_click_geoid].copy()
    else:
        active_df_stats = final_gdf.copy()

    # 3. Map (Right Column)
    with col_map:
        # Helpers
        @st.cache_data
        def convert_df(df): return df.to_csv(index=False).encode('utf-8')
        @st.cache_data
        def convert_gdf_to_geojson(_gdf): return _gdf.to_json()

        c1, c2 = st.columns(2)
        c1.download_button("📥 Download GeoJSON", convert_gdf_to_geojson(final_gdf), f"IPD_{selected_state_name}.geojson", "application/json", use_container_width=True, key="btn_geo")
        c2.download_button("📥 Download CSV", convert_df(final_gdf.drop(columns='geometry')), f"IPD_{selected_state_name}.csv", "text/csv", use_container_width=True, key="btn_csv")

        # Map bounds based on ACTIVE selection
        if not active_df_stats.empty:
            center_lat = active_df_stats.geometry.centroid.y.mean()
            center_lon = active_df_stats.geometry.centroid.x.mean()
            m = folium.Map(location=[center_lat, center_lon])
            
            if active_df_stats.crs != 'EPSG:4326': active_df_stats = active_df_stats.to_crs('EPSG:4326')
            min_lon, min_lat, max_lon, max_lat = active_df_stats.total_bounds
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        else:
            m = folium.Map(location=[39.8, -98.5], zoom_start=4)

        map_col = 'IPD_SCORE_score' if 'IPD_SCORE_score' in final_gdf.columns else 'IPD_SCORE'
        map_legend = 'IPD Score (0-4)' if 'IPD_SCORE_score' in final_gdf.columns else 'IPD Score'
        
        map_data_to_render = active_df_stats if not active_df_stats.empty else final_gdf

        choropleth = folium.Choropleth(
            geo_data=map_data_to_render.to_json(),
            data=map_data_to_render,
            columns=['GEOID', map_col],
            key_on='feature.properties.GEOID',
            fill_color='YlOrRd',
            legend_name=map_legend,
            name='IPD Scores'
        )
        choropleth.add_to(m)
        choropleth.geojson.add_child(folium.features.GeoJsonTooltip(fields=['GEOID', map_col], labels=True))
        
        map_key = f"map_last_click_{st.session_state.get('map_reset_token', 0)}"
        st_folium(m, width="100%", height=450, key=map_key, returned_objects=["last_object_clicked"])

    # 4. Update Stats Tab
    with tab_stats:
        if not active_df_stats.empty:
            subset_stats = []
            indicators = [c.replace('_pct', '') for c in active_df_stats.columns if c.endswith('_pct')]
            
            for ind in indicators:
                pct_col = f'{ind}_pct'
                mean_val = active_df_stats[pct_col].mean()
                sd_val = active_df_stats[pct_col].std()
                subset_stats.append({
                    'Indicator': ind,
                    'Mean': round(mean_val, 1) if not pd.isna(mean_val) else 0,
                    'SD': round(sd_val, 1) if not pd.isna(sd_val) else 0
                })
            
            if 'IPD_SCORE' in active_df_stats.columns:
                 mean_ipd = active_df_stats['IPD_SCORE'].mean()
                 sd_ipd = active_df_stats['IPD_SCORE'].std()
                 subset_stats.append({
                    'Indicator': 'Composite',
                    'Mean': round(mean_ipd, 1) if not pd.isna(mean_ipd) else 0,
                    'SD': round(sd_ipd, 1) if not pd.isna(sd_ipd) else 0
                })
            
            stats_df = pd.DataFrame(subset_stats)
            st.dataframe(stats_df, use_container_width=True, height=400)
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