using DrWatson:
    @quickactivate,
    datadir

@quickactivate "DeathsOfDespair"

using TigerLine:
    TIGER_DICT,
    download_tiger

# CONSTANTS
LAYER = "state"
YEAR = 2024

census_data_dir = datadir(
    "exp_raw",
    LAYER
)

!isdir(census_data_dir) ? mkdir(census_data_dir) : nothing

download_tiger(
    census_data_dir;
    year = YEAR,
    layer = LAYER
)
