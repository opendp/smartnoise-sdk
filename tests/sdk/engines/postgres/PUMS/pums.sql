CREATE DATABASE pums;
\c pums
CREATE SCHEMA pums
CREATE TABLE pums.pums (age int, sex char(2), educ int, race char(2), income float, married boolean );
CREATE TABLE pums.pums_large (PersonID bigint, state int, puma bigint, sex int, age int, educ int, income float, latino boolean, black boolean, asian boolean, married boolean);

DELETE FROM pums.pums;
\copy pums.pums FROM 'PUMS.csv' CSV;
DELETE FROM pums.pums_large;
\copy pums.pums_large FROM 'PUMS_large.csv' CSV;

CREATE DATABASE pums_pid;
\c pums_pid
CREATE SCHEMA pums
CREATE TABLE pums.pums (age int, sex char(2), educ int, race char(2), income float, married boolean, pid int);
CREATE TABLE pums.pums_dup (age int, sex char(2), educ int, race char(2), income float, married boolean, pid int);

DELETE FROM pums.pums;
\copy pums.pums FROM 'PUMS_pid.csv' CSV;
DELETE FROM pums.pums_dup;
\copy pums.pums_dup FROM 'PUMS_dup.csv' CSV;
