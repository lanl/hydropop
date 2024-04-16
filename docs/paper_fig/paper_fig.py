import geopandas as gpd
import matplotlib.pyplot as plt

run_names = ["coarse_coarse_small", "coarse_fine_small", "fine_coarse_small", "fine_fine_small"]
run_name = run_names[3]

path_gpkg = "results/" + run_name + "/" + run_name + "_hpus.gpkg"

dt = gpd.read_file(path_gpkg)
# results/{run_name}/{run_name}_hpu_classes.gpkg

dt.plot()
plt.savefig("test.pdf")
