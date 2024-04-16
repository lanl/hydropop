# https://code.earthengine.google.com/

all: data figures

data:
	python hydropop/dev/end_to_end_new.py \
		--pop_breaks -11 -10 -4 0 100 \
		--hthi_breaks -0.01 0.4 0.7 1.01 \
		--path_bounding_box data/roi_small.gpkg \
		--run_name coarse_coarse_small
	
	python hydropop/dev/end_to_end_new.py \
		--pop_breaks -11 -10 -4 0 100 \
		--hthi_breaks -0.01 0.3 0.55 0.75 0.9 1.01 \
		--path_bounding_box data/roi_small.gpkg \
		--run_name coarse_fine_small
	
	python hydropop/dev/end_to_end_new.py \
		--pop_breaks -11 -10 -4 -1 1 2 100 \
		--hthi_breaks -0.01 0.4 0.7 1.01 \
		--path_bounding_box data/roi_small.gpkg \
		--run_name fine_coarse_small
	
	python hydropop/dev/end_to_end_new.py \
		--pop_breaks -11 -10 -4 -1 1 2 100 \
		--hthi_breaks -0.01 0.3 0.55 0.75 0.9 1.01 \
		--path_bounding_box data/roi_small.gpkg \
		--run_name fine_fine_small

figures: docs/paper_fig/paper_fig.py
	python $<

clean:
	-@rm ../../results/*.tif
	-@rm ../../results/*.cpg
	-@rm ../../results/*.dbf
	-@rm ../../results/*.gpkg
	-@rm ../../results/*.prj
	-@rm ../../results/*.shp
	-@rm ../../results/*.shx
	-@rm ../../results/*.csv