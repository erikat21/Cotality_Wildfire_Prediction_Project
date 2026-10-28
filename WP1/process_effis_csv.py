import pandas as pd
import numpy as np
from pathlib import Path

def process_effis_csv(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # filter for Spain (ES) and Portugal (PT)
    df_filtered = df[df['COUNTRY'].isin(['ES', 'PT'])]

    # standardize dates and extract year and month
    df_filtered['FIREDATE'] = pd.to_datetime(df_filtered['FIREDATE'], format = 'mixed')
    df_filtered['FINALDATE'] = pd.to_datetime(df_filtered['FINALDATE'], format = 'mixed')
    df_filtered['year'] = df_filtered['FIREDATE'].dt.year
    df_filtered['month'] = df_filtered['FIREDATE'].dt.month

    # product_era mapping
    conditions = [
        (df_filtered['year'] <= 2018),
        (df_filtered['year'] == 2019),
        (df_filtered['year'] >= 2020)
    ]
    choices = [0, 1, 2]
    df_filtered['product_era'] = np.select(conditions, choices, default=2)

    # size_class mapping
    conditions_size = [
        (df_filtered['AREA_HA'] < 30),
        (df_filtered['AREA_HA'] < 383),
        (df_filtered['AREA_HA'] >= 383)
    ]

    choices_size = ['below 30 ha', '30 to 382 ha', '383 ha or more']

    df_filtered['size_class'] = np.select(conditions_size, choices_size, default='383 ha or more') 

    # fire duration
    df_filtered['duration_days'] = (df_filtered['FINALDATE'] - df_filtered['FIREDATE']).dt.days

    # drop land cover columns and other unnecessary columns
    df_filtered = df_filtered.drop(columns=['BROADLEA', 'CONIFER', 'MIXED', 'SCLEROPH', 'TRANSIT', 'OTHERNATLC', 
    'AGRIAREAS', 'ARTIFSURF', 'OTHERLC', 'PERCNA2K', 'CLASS', 'FINALDATE', 'LASTUPDATE', 'COMMUNE'])

    fires_master = df_filtered.copy()
    fires_master = fires_master.rename(columns={
        'COUNTRY': 'country',
        'id': 'fire_id',
        'PROVINCE': 'nuts2',
        'AREA_HA': 'area_ha'
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fires_master.to_parquet(output_path)

if __name__ == "__main__":
    input_file = "../data/raw/effis_data_table_29thSept2026.csv"  # Update with your local filename
    output_file = "../data/processed/fires_master.parquet"
    process_effis_csv(input_file, output_file)
