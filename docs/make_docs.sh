# make folder for static files
rm -rf /tmp/docs
mkdir /tmp/docs

# landing page
pip install -r requirements.txt
make html
cp -R build/html/* /tmp/docs/
make clean
cd ..

# sql
cd sql
pip install -e .
cd docs
make html
cp -R build/html /tmp/docs/sql
make clean
cd ../..

# synth
cd synth
pip install -e .
cd docs
make html
cp -R build/html /tmp/docs/synth
make clean
cd ../..

open /tmp/docs/index.html


