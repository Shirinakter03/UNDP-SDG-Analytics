# Data Sources

## sdg_indicators.csv

| Column | Source | Dataset ID | Year |
|--------|--------|-----------|------|
| hdi | UNDP Human Development Report | HDR 2023 | 2023 |
| poverty_rate | World Bank Poverty & Inequality Platform | SI.POV.DDAY | 2023 |
| literacy_rate | UNESCO Institute for Statistics | UIS.LR_AG15T99 | 2023 |
| co2_per_capita | Our World in Data / Global Carbon Project | — | 2023 |
| internet_access | World Bank WDI | IT.NET.USER.ZS | 2023 |
| gni_per_capita | World Bank Atlas Method | NY.GNP.PCAP.PP.CD | 2023 |

## hdi_trends.csv

| Column | Source | Notes |
|--------|--------|-------|
| hdi | UNDP Human Development Report (2015–2023) | 2-year intervals; 2021 captures COVID-19 impact |

## Notes on Data Curation
- 24 countries selected across 6 regions for geographic diversity
- Only countries with complete data for all 6 indicators included
- Poverty rate uses $2.15/day international poverty line (2017 PPP)
- GNI per capita in current international dollars (PPP)
- Internet access = % of population using the internet
