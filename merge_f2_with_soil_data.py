import pandas as pd
import numpy as np


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        'Temparature': 'Temperature',
        'Phosphorous': 'Phosphorus',
        'Nitrogen Value': 'Nitrogen',
        'Phosphorous value': 'Phosphorus',
        'Potassium value': 'Potassium',
        'pH': 'pH',
        'District': 'District',
    }
    return df.rename(columns=column_map)


def round_numeric(series: pd.Series) -> pd.Series:
    # Round to nearest integer for matching
    return pd.to_numeric(series, errors='coerce').round().astype('Int64')


def main():
    df_f2 = pd.read_csv('f2.csv')
    df_soil = pd.read_csv('Soil data.csv')

    df_f2 = normalize_columns(df_f2)
    df_soil = normalize_columns(df_soil)

    # Prepare keys for joining on N/P/K (rounded to nearest int for compatibility)
    for col in ['Nitrogen', 'Phosphorus', 'Potassium']:
        if col in df_f2.columns:
            df_f2[f'{col}_key'] = round_numeric(df_f2[col])
        if col in df_soil.columns:
            df_soil[f'{col}_key'] = round_numeric(df_soil[col])

    # Select soil metadata to merge
    soil_meta = df_soil[[
        'Nitrogen_key', 'Phosphorus_key', 'Potassium_key', 'District', 'pH'
    ]].drop_duplicates()

    # Merge to enrich f2 with District and pH
    merged = df_f2.merge(
        soil_meta,
        on=['Nitrogen_key', 'Phosphorus_key', 'Potassium_key'],
        how='left',
        suffixes=('', '_soil')
    )

    # If pH still missing, try a looser join (within ±1 tolerance)
    # Ensure canonical columns exist
    if 'pH' not in merged.columns:
        merged['pH'] = np.nan
    if 'District' not in merged.columns:
        merged['District'] = np.nan

    # Coalesce possible variants created by previous runs or merges
    if 'District_soil' in merged.columns:
        merged['District'] = merged['District'].combine_first(merged['District_soil'])
    if 'pH_soil' in merged.columns:
        merged['pH'] = merged['pH'].combine_first(merged['pH_soil'])
    # Handle any _x/_y leftovers
    if 'District_x' in merged.columns:
        merged['District'] = merged['District'].combine_first(merged['District_x'])
    if 'District_y' in merged.columns:
        merged['District'] = merged['District'].combine_first(merged['District_y'])
    if 'pH_x' in merged.columns:
        merged['pH'] = merged['pH'].combine_first(merged['pH_x'])
    if 'pH_y' in merged.columns:
        merged['pH'] = merged['pH'].combine_first(merged['pH_y'])
    missing_mask = merged['pH'].isna()
    if missing_mask.any():
        f2_missing = merged.loc[missing_mask, ['Nitrogen_key', 'Phosphorus_key', 'Potassium_key']].copy()
        # Build expanded keys for tolerance join
        expanded = []
        for delta_n in [-1, 0, 1]:
            for delta_p in [-1, 0, 1]:
                for delta_k in [-1, 0, 1]:
                    tmp = f2_missing.copy()
                    tmp['Nitrogen_key'] = tmp['Nitrogen_key'] + delta_n
                    tmp['Phosphorus_key'] = tmp['Phosphorus_key'] + delta_p
                    tmp['Potassium_key'] = tmp['Potassium_key'] + delta_k
                    expanded.append(tmp)
        f2_expanded = pd.concat(expanded, ignore_index=True).drop_duplicates()

        # Join expanded keys to soil meta and then map back
        tolerant = f2_expanded.merge(soil_meta, on=['Nitrogen_key', 'Phosphorus_key', 'Potassium_key'], how='left')
        tolerant = tolerant.dropna(subset=['pH'])
        tolerant = tolerant.drop_duplicates(subset=['Nitrogen_key', 'Phosphorus_key', 'Potassium_key'])

        # Map back where missing
        merged_idx = merged.index[missing_mask]
        key_cols = ['Nitrogen_key', 'Phosphorus_key', 'Potassium_key']
        key_to_meta = tolerant.set_index(key_cols)[['District', 'pH']]
        for idx in merged_idx:
            key = tuple(merged.loc[idx, key_cols])
            if key in key_to_meta.index:
                merged.at[idx, 'District'] = key_to_meta.loc[key, 'District']
                merged.at[idx, 'pH'] = key_to_meta.loc[key, 'pH']

    # Drop helper keys
    drop_cols = ['Nitrogen_key', 'Phosphorus_key', 'Potassium_key', 'District_soil', 'pH_soil', 'District_x', 'District_y', 'pH_x', 'pH_y']
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # Reorder columns: add District and pH at the end if not already present
    cols = list(merged.columns)
    for c in ['District', 'pH']:
        if c in cols:
            cols = [x for x in cols if x != c] + [c]
    merged = merged[cols]

    # If less than 1000 rows, append mapped rows from soil data to reach 1000
    target_rows = 1000
    if len(merged) < target_rows:
        deficit = target_rows - len(merged)
        # Prepare appendable rows from soil data mapped to f2 schema
        df_append = df_soil.copy()
        # Ensure numeric N/P/K and round to integers to align with f2
        for col in ['Nitrogen', 'Phosphorus', 'Potassium']:
            if col in df_append.columns:
                df_append[col] = pd.to_numeric(df_append[col], errors='coerce').round().astype('Int64')

        # Fill required feature columns
        df_append['Temperature'] = 28  # default values
        df_append['Humidity'] = 60
        df_append['Moisture'] = 50
        df_append['Soil_Type'] = np.nan
        df_append['Crop_Type'] = np.nan
        df_append['Fertilizer'] = np.nan

        # Keep only columns that exist in merged, in that order
        df_append = df_append[[c for c in merged.columns if c in df_append.columns or c in ['Temperature','Humidity','Moisture','Soil_Type','Crop_Type','Nitrogen','Potassium','Phosphorus','Fertilizer','District','pH']]]

        # Ensure all required columns present
        for c in merged.columns:
            if c not in df_append.columns:
                df_append[c] = np.nan
        df_append = df_append[merged.columns]

        # Take only as many rows as needed
        df_append = df_append.head(deficit)
        merged = pd.concat([merged, df_append], ignore_index=True)

    # Enforce exactly 1000 rows (trim if exceeded due to prior runs)
    target_rows = 1000
    if len(merged) > target_rows:
        merged = merged.head(target_rows)

    merged.to_csv('f2.csv', index=False)
    print(f'Wrote combined data to f2.csv with {len(merged)} rows.')


if __name__ == '__main__':
    main()


