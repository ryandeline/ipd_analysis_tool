import numpy as np
import pandas as pd

def calculate_moe_prop(num, den, num_moe, den_moe):
    """
    Calculates Margin of Error for a proportion/percentage.
    Formula: (1/den) * sqrt( num_moe^2 - (prop^2 * den_moe^2) )
    """
    if den == 0:
        return 0
    
    prop = num / den
    
    # Check for negative value under sqrt (can happen if estimates are small)
    # Census standard: If negative, add instead of subtract (conservative estimate)
    inner = (num_moe**2) - ((prop**2) * (den_moe**2))
    
    if inner < 0:
        inner = (num_moe**2) + ((prop**2) * (den_moe**2))
        
    return (np.sqrt(inner) / den) * 100

def process_indicators(df):
    """
    Performs all variable-specific calculations (e.g. Racial Minority subtraction).
    Calculates Percentages and MOEs.
    """
    # 1. Racial Minority Calculation: Total Pop (Universe) - White Alone (Count)
    # We loaded 'RM_count' as White Alone in config, so we invert it here.
    if 'RM_uni' in df.columns and 'RM_est' in df.columns:
        df['RM_est'] = df['RM_uni'] - df['RM_est']
        # Approximate MOE for derived count (sqrt(moe1^2 + moe2^2))
        if 'RM_uni_moe' in df.columns and 'RM_est_moe' in df.columns:
            df['RM_est_moe_approx'] = np.sqrt(df['RM_uni_moe']**2 + df['RM_est_moe']**2)
    
    # 2. Loop through all indicators to calc Pct and Pct_MOE
    indicators = ['Y', 'OA', 'F', 'RM', 'EM', 'FB', 'LEP', 'D', 'LI', 'NC']
    
    for ind in indicators:
        # Skip if indicator columns are missing (prevent KeyError)
        if f'{ind}_est' not in df.columns or f'{ind}_uni' not in df.columns:
            print(f"Warning: Missing columns for indicator {ind}. Skipping.")
            continue
            
        # Calculate Percentage
        # Handle division by zero
        df[f'{ind}_pct'] = np.where(df[f'{ind}_uni'] > 0, 
                                    (df[f'{ind}_est'] / df[f'{ind}_uni']) * 100, 
                                    0).round(1)
        
        # Calculate Percentage MOE
        df[f'{ind}_pct_moe'] = df.apply(
            lambda x: calculate_moe_prop(
                x[f'{ind}_est'], x[f'{ind}_uni'], 
                x[f'{ind}_est_moe'], x[f'{ind}_uni_moe']
            ), axis=1
        ).round(1)
        
    return df

def calculate_sd_scores(df, indicators):
    """
    Applies the Standard Deviation scoring methodology.
    Includes Row-wise Confidence check and Column-wise stats comparison.
    """
    df_scored = df.copy()
    
    # Initialize Composite Score
    df_scored['IPD_SCORE'] = 0
    
    stats_list = []
    score_cols = []
    
    # 1. Score Individual Indicators
    for ind in indicators:
        pct_col = f'{ind}_pct'
        if pct_col not in df.columns: continue
        
        # Calculate Stats (Mean and SD)
        mean_val = df_scored[pct_col].mean()
        sd_val = df_scored[pct_col].std()
        
        # Define Breaks
        # Break 1: Mean - 1.5 SD
        b1 = mean_val - (1.5 * sd_val)
        # Handle negative break (coerce to 0.1 so 0.0 is 'Well Below')
        if b1 < 0: b1 = 0.1
        
        b2 = mean_val - (0.5 * sd_val)
        b3 = mean_val + (0.5 * sd_val)
        b4 = mean_val + (1.5 * sd_val)
        
        # Assign Scores
        def get_score(val, breaks):
            b1, b2, b3, b4 = breaks
            if val < b1: return 0
            if val < b2: return 1
            if val < b3: return 2
            if val < b4: return 3
            return 4
            
        def get_class(score):
            mapping = {
                0: "Well Below Average",
                1: "Below Average",
                2: "Average",
                3: "Above Average",
                4: "Well Above Average"
            }
            return mapping.get(score, "Unknown")
            
        score_col = f'{ind}_score'
        df_scored[score_col] = df_scored[pct_col].apply(lambda x: get_score(x, (b1, b2, b3, b4)))
        df_scored[f'{ind}_class'] = df_scored[score_col].apply(get_class)
        
        # Add to composite
        df_scored['IPD_SCORE'] += df_scored[score_col]
        score_cols.append(score_col)
        
        # Collect stats for summary report
        stats_list.append({
            'Indicator': ind,
            'Mean': round(mean_val, 1),
            'SD': round(sd_val, 1),
            'Break_Min_1.5SD': round(b1, 1),
            'Break_Min_0.5SD': round(b2, 1),
            'Break_Plus_0.5SD': round(b3, 1),
            'Break_Plus_1.5SD': round(b4, 1)
        })

    # 2. Score the Composite IPD_SCORE itself (Comparison Logic)
    # We apply the same logic (Mean +/- SD) to the final summed score
    ipd_col = 'IPD_SCORE'
    if ipd_col in df_scored.columns:
        mean_ipd = df_scored[ipd_col].mean()
        sd_ipd = df_scored[ipd_col].std()
        
        # Breaks for IPD Score
        ib1 = mean_ipd - (1.5 * sd_ipd)
        if ib1 < 0: ib1 = 0.1
        ib2 = mean_ipd - (0.5 * sd_ipd)
        ib3 = mean_ipd + (0.5 * sd_ipd)
        ib4 = mean_ipd + (1.5 * sd_ipd)
        
        # Reuse get_score logic
        df_scored['IPD_SCORE_score'] = df_scored[ipd_col].apply(lambda x: get_score(x, (ib1, ib2, ib3, ib4)))
        df_scored['IPD_SCORE_class'] = df_scored['IPD_SCORE_score'].apply(get_class)
        
        # Add stats for the composite score to the summary
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
        # Calculate row-wise stats for indicator scores
        if score_cols:
            df_scored['indicators_mean'] = df_scored[score_cols].mean(axis=1)
            df_scored['indicators_std'] = df_scored[score_cols].std(axis=1)
            
            # Logic: If a tract has a high score but high variance, it means it scores high 
            # on just a few indicators, not all.
            # Low Variance = Consistent disadvantage across all measures.
            # High Variance = Disadvantage driven by outliers.
            
            # We can create a "Confidence" flag based on the Coefficient of Variation (CV)
            # CV = SD / Mean. High CV means inconsistent scores.
            
            def get_confidence(row):
                # Avoid division by zero
                if row['indicators_mean'] == 0:
                    return 'High' # Consistent 0s
                
                cv = row['indicators_std'] / row['indicators_mean']
                
                # Heuristic thresholds
                if cv < 0.5:
                    return 'High'
                elif cv < 1.0:
                    return 'Medium'
                else:
                    return 'Low'

            df_scored['IPD_CONFIDENCE'] = df_scored.apply(get_confidence, axis=1)
            
            # Add Row-wise stats to summary
            stats_list.append({
                'Indicator': 'ROW_WISE_INDICATOR_SCORES',
                'Mean': round(df_scored['indicators_mean'].mean(), 1),
                'SD': round(df_scored['indicators_mean'].std(), 1),
                'Break_Min_1.5SD': np.nan, 
                'Break_Min_0.5SD': np.nan,
                'Break_Plus_0.5SD': np.nan,
                'Break_Plus_1.5SD': np.nan
            })
        
    return df_scored, pd.DataFrame(stats_list)