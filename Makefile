.PHONY: docs clean

docs: docs/build/html/index.html

docs/build/html/index.html:
	cd docs && sphinx-build -M html source build

clean: 
	rm docs/build/html/index.html
