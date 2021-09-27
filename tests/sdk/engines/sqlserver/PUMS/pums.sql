IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE database_id = DB_ID(N'PUMS'))
    CREATE DATABASE PUMS;
GO

USE PUMS;
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS_large]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS_large]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS]
GO

DROP SCHEMA IF EXISTS PUMS;
GO

CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married bit )
	CREATE TABLE PUMS_large (PersonID int, state int, puma int, sex int, age int, educ int, income float, latino bit, black bit, asian bit, married bit);
GO

IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE database_id = DB_ID(N'PUMS_pid'))
    CREATE DATABASE PUMS_pid;
GO

USE PUMS_pid;
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS_dup]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS_dup]
GO

DROP SCHEMA IF EXISTS PUMS;
GO

CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married bit, pid int)
	CREATE TABLE PUMS_dup (age int, sex char(2), educ int, race char(2), income float, married bit, pid int)
GO