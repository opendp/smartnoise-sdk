-- PUMS and PUM_large daabases

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

-- PUMS_pid database

IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE database_id = DB_ID(N'PUMS_pid'))
    CREATE DATABASE PUMS_pid;
GO

USE PUMS_pid;
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS]
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS]
GO

DROP SCHEMA IF EXISTS PUMS;
GO

CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married bit, pid int)
GO

-- PUMS_dup database

IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE database_id = DB_ID(N'PUMS_dup'))
    CREATE DATABASE PUMS_dup;
GO

USE PUMS_dup;
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS]
GO

DROP SCHEMA IF EXISTS PUMS;
GO

CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married bit, pid int)
GO

-- PUMS_null database

IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE database_id = DB_ID(N'PUMS_null'))
    CREATE DATABASE PUMS_null;
GO

USE PUMS_null;
GO

IF  EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[PUMS].[PUMS]') AND type in (N'U'))
DROP TABLE [PUMS].[PUMS]
GO

DROP SCHEMA IF EXISTS PUMS;
GO

CREATE SCHEMA PUMS
	CREATE TABLE PUMS (age int, sex char(2), educ int, race char(2), income float, married bit, pid int)
GO
