wheel:
	poetry build
setup: wheel
	tar -xvf dist/*-`poetry version -s`.tar.gz -O '*/setup.py' > setup.py
