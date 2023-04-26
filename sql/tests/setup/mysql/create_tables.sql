CREATE DATABASE IF NOT EXISTS PUMS;
CREATE DATABASE IF NOT EXISTS PUMS_pid;
CREATE DATABASE IF NOT EXISTS PUMS_dup;
CREATE DATABASE IF NOT EXISTS PUMS_null;
CREATE DATABASE IF NOT EXISTS PUMS_large;


CREATE TABLE IF NOT EXISTS PUMS.pums (
    age INT,
    sex INT,
    educ INT,
    race INT,
    income INT,
    married INT
);


CREATE TABLE IF NOT EXISTS PUMS_pid.pums (
    pid INT PRIMARY KEY,
    age INT,
    sex INT,
    educ INT,
    race INT,
    income INT,
    married INT
);

CREATE TABLE IF NOT EXISTS PUMS_dup.pums (
    pid INT,
    age INT,
    sex INT,
    educ INT,
    race INT,
    income INT,
    married INT
);


CREATE TABLE IF NOT EXISTS PUMS_null.pums (
    pid INT NULL,
    age INT NULL,
    sex INT NULL,
    educ INT NULL,
    race INT NULL,
    income INT NULL,
    married INT NULL
);

CREATE TABLE IF NOT EXISTS PUMS_large.pums_large (
    PersonID INT PRIMARY KEY,
    state INT,
    puma INT,
    sex INT,
    age INT,
    educ INT,
    income FLOAT,
    latino BOOLEAN,
    black BOOLEAN,
    asian BOOLEAN,
    married BOOLEAN
);


USE PUMS;
LOAD DATA INFILE '/var/lib/mysql-files/PUMS.csv'
INTO TABLE pums
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

USE PUMS_pid;
LOAD DATA INFILE '/var/lib/mysql-files/PUMS_pid.csv'
INTO TABLE pums
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

USE PUMS_dup;
LOAD DATA INFILE '/var/lib/mysql-files/PUMS_dup.csv'
INTO TABLE pums
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

USE PUMS_large;
LOAD DATA INFILE '/var/lib/mysql-files/PUMS_large.csv'
INTO TABLE pums_large
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

USE PUMS_null;
LOAD DATA INFILE '/var/lib/mysql-files/PUMS_null.csv'
INTO TABLE pums
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

