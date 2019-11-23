In order to run the service tests you must first initialize the service
TODO: can we remove the need for this a separate call to start the server?

Both terminals below are run from the root directory of the git repository

terminal 1:
python service/app.py

terminal 2:

>>cd tests
>>pip install -r requirements.txt  
>>cd ..
>>pytest tests/service/

The requirements.txt currently contains relative paths and must therefore be run from the tests directory.
