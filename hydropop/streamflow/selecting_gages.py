import os
import sys
import geopandas as gpd

sys.path.append(".")
from hydropop import config
config.vote_db()
from VotE.streamflow import export_streamflow as es

path_bounding_box = r"data/roi_small.gpkg"
bb = gpd.read_file(path_bounding_box)
gage_params = {
    "within": bb.geometry.values[0],
    "max_drainarea_km2": 10000,
    "fraction_valid": 0.9,
    "vote_snapped": True,
    "end_date": "2000-01-01",
    "min_span_years": 10,
}
gage_ids = es.gage_selector(gage_params)
gages = es.get_gages(gage_ids)

# Export the gages and their watersheds (two files)
keepkeys = [k for k in gages.keys() if "geom" not in k]
keepkeys = [k for k in keepkeys if "chunk" not in k]
keepkeys.remove("id_duplicates")
gage_locs = gpd.GeoDataFrame(
    data=gages[keepkeys], geometry=gages["mapped_geom"], crs=gages.crs
)
basins = gpd.GeoDataFrame(
    data=gages[keepkeys], geometry=gages["basin_geom_vote"], crs=gages.crs
)

gage_locs["start_date"] = gage_locs["start_date"].astype(str)
gage_locs["end_date"] = gage_locs["end_date"].astype(str)
basins["start_date"] = basins["start_date"].astype(str)
basins["end_date"] = basins["end_date"].astype(str)

# gage_locs.to_file(r'X:\Research\CIMMID\Data\Hydropop Layers\Finals\na_10k\gage_selection\na_10k_gages.gpkg', driver='GPKG')
# basins.to_file(r'X:\Research\CIMMID\Data\Hydropop Layers\Finals\na_10k\gage_selection\na_10k_basins.gpkg', driver='GPKG')

# # Updating to fix end_date
# current_gages = r"X:\Research\CIMMID\Data\Hydropop Layers\Finals\na_10k\gage_selection\trap_data_basins.gpkg"
# current_gages = gpd.read_file(current_gages)
# gages = es.get_gages(current_gages['id_gage'].values.tolist())

gage_locs.to_file(
    r"data/initial_gages.gpkg", driver="GPKG"
)
basins.to_file(
    r"data/initial_basins.gpkg", driver="GPKG"
)

""" Download streamflow data """
path_dir = "data/na_10k/streamflow/"
os.makedirs(path_dir, exist_ok=True)

for id_gage in basins["id_gage"].values:
    print(id_gage)
    df = es.get_streamflow_timeseries(
        [int(id_gage)], start_date="1981-01-01", expand=True, trim=True
    )
    df.drop("id_gage", inplace=True, axis=1)
    df.sort_values(by="date", inplace=True)    
    path_out = path_dir + str(id_gage) + ".csv"
    df.to_csv(path_out, index=False)
