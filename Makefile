.PHONY: docs clean

docs: docs/build/html/index.html

docs/build/html/index.html:
	cd docs && sphinx-build -M html source build

clean: 
	rm docs/build/html/index.html

rabpro:
	pip install --upgrade -e ../rabpro

initial_data: data/initial_basins.gpkg data/initial_gages.gpkg

data/initial_basins.gpkg data/initial_gages.gpkg: ecopop/streamflow/selecting_gages.py
	python $<
