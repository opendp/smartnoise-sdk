# make folder for static files
rm -rf /tmp/docs
mkdir /tmp/docs

# landing page
pip install -r requirements.txt
make html
cp redirect.html /tmp/docs/index.html
mkdir /tmp/docs/en
cp -R build/html /tmp/docs/en/stable
make clean
cd ..

# sql
cd sql
pip install -e .
cd docs
make html
cp -R build/html /tmp/docs/en/stable/sql
make clean
cd ../..

# synth
cd synth
pip install -e .
cd docs
make html
cp -R build/html /tmp/docs/en/stable/synth
make clean
cd ../..

cd eval
pip install -e .
cd docs
make html
cp -R build/html /tmp/docs/en/stable/eval
make clean
cd ../..

open /tmp/docs/index.html


