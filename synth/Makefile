wheel:
	poetry build
setup: dist/*-`poetry version -s`.tar.gz
	tar -xvf dist/*-`poetry version -s`.tar.gz -O '*/setup.py' > setup.py
