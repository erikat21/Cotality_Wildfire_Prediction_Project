# Basic Fire Statistics From the Fire Master Dataset

**Datset:** `fires_master.parquet`
**Geographic Scope:** Spain(ES) and Portugal(PT)
**Temporal COverage:** May 2016 to Septermber 2026

## Overall Dataset Summary
The master dataset captures over a decade of wildfire activity across the Iberian Peninsula, recording nearly 17,000 individual fire events totaling over 3.1 million hectares burned.
* **Total Fires Recorded:** 16913
* **Total Burned Area:** 3161672 ha
* **Date Range:** 2016-04-30 to 2026-09-28

## Country Breakdown
Spain records a higher total volume of fires, while Portugal exhibits a slightly higher mean burned area per fire but Spain has a slighlty higher median_area burned, pointing to differences in regional spread dynamics.
| country   |   fire_count |   total_area_ha |   mean_area_ha |   median_area_ha |
|:----------|-------------:|----------------:|---------------:|-----------------:|
| ES        |         9414 |         1597230 |        169.665 |               15 |
| PT        |         7499 |         1564442 |        208.62  |               11 |

## Year Breakdown
Annual totals highlight significant inter-annual volatility, with extreme burned area spiking in 2017 and 2025.
|   year |   fire_count |   total_area_ha |
|-------:|-------------:|----------------:|
|   2016 |          450 |          218826 |
|   2017 |          736 |          694732 |
|   2018 |          190 |           50145 |
|   2019 |          646 |          101073 |
|   2020 |          937 |          130072 |
|   2021 |         1649 |          122885 |
|   2022 |         2726 |          427768 |
|   2023 |         2334 |          144219 |
|   2024 |         1510 |          195210 |
|   2025 |         2358 |          685278 |
|   2026 |         3377 |          391464 |

## Product_era Breakdown
Era 2 (2020+) displays a surge in recorded fire counts, however it does encompass multiple years.
|   product_era |   fire_count |   total_area_ha |
|--------------:|-------------:|----------------:|
|             0 |         1376 |          963703 |
|             1 |          646 |          101073 |
|             2 |        14891 |         2096896 |

## Size_class Breakdown
The distribution demonstrates extreme right-skewness: while small fires (under 30 ha) dominate total frequency, mega-fires (383+ ha) account for the vast majority of total burned area.
| size_class     |   fire_count |   total_area_ha |
|:---------------|-------------:|----------------:|
| 30 to 382 ha   |         4814 |          465780 |
| 383 ha or more |          849 |         2597967 |
| below 30 ha    |        11250 |           97925 |

## Year-Country Breakdown
This breakdown demonstrates regional differences between years such as Portugal's worst fire year based on total area burned being 2017 and Spain's in 2025.
### Fire Counts by Year & Country
|   year |   ES |   PT |
|-------:|-----:|-----:|
|   2016 |  128 |  322 |
|   2017 |  322 |  414 |
|   2018 |  104 |   86 |
|   2019 |  424 |  222 |
|   2020 |  472 |  465 |
|   2021 |  900 |  749 |
|   2022 | 1490 | 1236 |
|   2023 | 1397 |  937 |
|   2024 |  769 |  741 |
|   2025 | 1359 |  999 |
|   2026 | 2049 | 1328 |

### Total Burned Area (ha) by Year & Country
|   year |     ES |     PT |
|-------:|-------:|-------:|
|   2016 |  52730 | 166096 |
|   2017 | 131050 | 563682 |
|   2018 |  12789 |  37356 |
|   2019 |  66408 |  34665 |
|   2020 |  64286 |  65786 |
|   2021 |  91307 |  31578 |
|   2022 | 315705 | 112063 |
|   2023 | 101184 |  43035 |
|   2024 |  47607 | 147603 |
|   2025 | 401266 | 284012 |
|   2026 | 312898 |  78566 |

### Mean Burned Area (ha) by Year & Country
|   year |       ES |        PT |
|-------:|---------:|----------:|
|   2016 | 411.953  |  515.826  |
|   2017 | 406.988  | 1361.55   |
|   2018 | 122.971  |  434.372  |
|   2019 | 156.623  |  156.149  |
|   2020 | 136.199  |  141.475  |
|   2021 | 101.452  |   42.1602 |
|   2022 | 211.883  |   90.6659 |
|   2023 |  72.4295 |   45.9285 |
|   2024 |  61.9077 |  199.194  |
|   2025 | 295.266  |  284.296  |
|   2026 | 152.708  |   59.1611 |

## Product_era-Size_class Breakdown
The reporting discontinuity is clear here: small fires were rarely logged in Era 0 compared to Era 2.
|   product_era |   30 to 382 ha |   383 ha or more |   below 30 ha |
|--------------:|---------------:|-----------------:|--------------:|
|             0 |           1034 |              296 |            46 |
|             1 |            456 |               36 |           154 |
|             2 |           3324 |              517 |         11050 |

## Size_class-Country Breakdown
Both nations maintain a very similar proportional breakdown across size classes, however Spain is slightly higher across the board.
| country   |   30 to 382 ha |   383 ha or more |   below 30 ha |
|:----------|---------------:|-----------------:|--------------:|
| ES        |           2857 |              450 |          6107 |
| PT        |           1957 |              399 |          5143 |

## Month-Country Breakdown
Seasonality exhibits a dual peak: early spring months and post summer heatwave in July and August.
|   month |   ES |   PT |
|--------:|-----:|-----:|
|       1 |  599 |  754 |
|       2 | 1291 |  755 |
|       3 | 2138 | 1479 |
|       4 | 1490 |  806 |
|       5 |  277 |  220 |
|       6 |  433 |  169 |
|       7 |  888 |  668 |
|       8 | 1018 | 1156 |
|       9 |  522 |  767 |
|      10 |  416 |  413 |
|      11 |  162 |  173 |
|      12 |  180 |  139 |