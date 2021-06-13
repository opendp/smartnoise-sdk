CREATE DATABASE pums;
\c pums
CREATE TABLE pums (age int, sex char(2), educ int, race char(2), income float, married boolean );
CREATE TABLE pums_large (PersonID bigint, state int, puma bigint, sex int, age int, educ int, income float, latino boolean, black boolean, asian boolean, married boolean);

DELETE FROM pums;
\copy pums FROM 'PUMS.csv' CSV;
DELETE FROM pums_large;
\copy pums_large FROM 'PUMS_large.csv' CSV;