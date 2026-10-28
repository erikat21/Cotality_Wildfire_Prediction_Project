import pandas as pd

# Load the parquet file
df = pd.read_parquet("../data/processed/fires_master.parquet")

print("overall dataset summary")
print(f"Total Fires Recorded: {len(df):,}")
print(f"Date Range: {df['FIREDATE'].min()} to {df['FIREDATE'].max()}")
print(f"Total Burned Area: {df['area_ha'].sum():,.2f} ha")

print("\ncountry breakdown")
country_stats = df.groupby('country').agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum'),
    mean_area_ha=('area_ha', 'mean'),
    median_area_ha=('area_ha', 'median')
).reset_index()
print(country_stats.to_markdown(index=False))

print("\n year breakdown")
yearly_stats = df.groupby('year').agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum')
).reset_index()
print(yearly_stats.to_markdown(index=False))

print("\nproduct_era distribution")
era_stats = df.groupby('product_era').agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum')
).reset_index()
print(era_stats.to_markdown(index=False))

print("\nsize class breakdown")
size_summary = df.groupby('size_class').agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum')
).reset_index()
print(size_summary.to_markdown(index=False))

print("\n year-country breakdown")
year_country_stats = df.groupby(['year', 'country']).agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum'),
    mean_area_ha=('area_ha', 'mean')
).reset_index()
pivoted_counts = year_country_stats.pivot(index='year', columns='country', values='fire_count').fillna(0)
pivoted_area = year_country_stats.pivot(index='year', columns='country', values='total_area_ha').fillna(0)
pivoted_mean_area = year_country_stats.pivot(index='year', columns='country', values='mean_area_ha').fillna(0)

print("\nFire Counts by Year & Country")
print(pivoted_counts.to_markdown())

print("\nTotal Burned Area (ha) by Year & Country")
print(pivoted_area.to_markdown())

print("\nMean Burned Area (ha) by Year & Country")
print(pivoted_mean_area.to_markdown())

print("\n product era-size class breakdown")
era_size = df.groupby(['product_era', 'size_class']).agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum')
).reset_index()

pivoted_era_size = era_size.pivot(index='product_era', columns='size_class', values='fire_count').fillna(0)
print(pivoted_era_size.to_markdown())

print("\n size_class-country breakdown")
country_size = df.groupby(['country', 'size_class']).agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum'),
    median_area_ha=('area_ha', 'median')
).reset_index()

pivoted_country_size = country_size.pivot(index='country', columns='size_class', values='fire_count').fillna(0)
print(pivoted_country_size.to_markdown())

print("\n month-country breakdown")
monthly_country = df.groupby(['month', 'country']).agg(
    fire_count=('fire_id', 'count'),
    total_area_ha=('area_ha', 'sum')
).reset_index()

pivoted_monthly_country = monthly_country.pivot(index='month', columns='country', values='fire_count').fillna(0)
print(pivoted_monthly_country.to_markdown())