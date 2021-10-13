wheel:
	poetry build
setup: wheel
	tar -xvf dist/*-`poetry version -s`.tar.gz -O '*/setup.py' > setup.py
install_e: setup
	mv pyproject.toml pyproject.bak
	-pip install -e .
	mv pyproject.bak pyproject.toml