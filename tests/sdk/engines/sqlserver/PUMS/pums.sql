CREATE DATABASE PUMS;
GO

USE PUMS;
GO

CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married bit )
	CREATE TABLE PUMS_large (PersonID int, state int, puma int, sex int, age int, educ int, income float, latino bit, black bit, asian bit, married bit);
GO

CREATE DATABASE PUMS_dup;
GO

USE PUMS_dup;
GO

CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married bit, pid int)
	CREATE TABLE PUMS_dup (age int, sex char(2), educ int, race char(2), income float, married bit, pid int)
GO
