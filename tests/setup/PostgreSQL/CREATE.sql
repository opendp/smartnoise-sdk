SELECT 'CREATE DATABASE PUMS' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'PUMS')\gexec
GRANT ALL PRIVILEGES ON DATABASE "PUMS" TO postgres;
DROP SCHEMA IF EXISTS PUMS CASCADE;
CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married boolean )
	CREATE TABLE PUMS_large (PersonID bigint, state int, puma bigint, sex int, age int, educ int, income float, latino boolean, black boolean, asian boolean, married boolean);

\copy PUMS.PUMS FROM 'PUMS.csv' CSV;
\copy PUMS.PUMS_large FROM 'PUMS_large.csv' CSV;
