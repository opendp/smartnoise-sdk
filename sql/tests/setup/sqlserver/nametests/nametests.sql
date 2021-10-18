-- requires full file paths; edit below
CREATE DATABASE nametests
GO
USE nametests
GO
CREATE TABLE nametests (EDUC int, [2Educ"] int, [3395] int, [SELECT] float);
GO
DELETE FROM nametests
GO
BULK INSERT nametests
    FROM '<full path to this folder>\..\..\test_csv\nametests.csv'
    WITH
    (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',  --CSV field delimiter
    ROWTERMINATOR = '\n',   --Use to shift the control to next row
    ERRORFILE = '<full path to this folder>\errors_nametests',
    TABLOCK
    )
GO
CREATE TABLE [2NameTests] (EDUC int, [2Educ"] int, [3395] int, [SELECT] float);
GO
DELETE FROM [2NameTests]
GO
BULK INSERT [2NameTests]
    FROM '<full path to this folder>\..\..\test_csv\nametests_case.csv'
    WITH
    (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',  --CSV field delimiter
    ROWTERMINATOR = '\n',   --Use to shift the control to next row
    ERRORFILE = '<full path to this folder>\errors_nametests_case',
    TABLOCK
    )
GO

CREATE DATABASE [2NameTests]
GO
USE [2NameTests]
GO
CREATE TABLE nametests (EDUC int, [2Educ"] int, [3395] int, [SELECT] float);
DELETE FROM nametests
GO
BULK INSERT nametests
    FROM '<full path to this folder>\..\..\test_csv\2NameTests.csv'
    WITH
    (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',  --CSV field delimiter
    ROWTERMINATOR = '\n',   --Use to shift the control to next row
    ERRORFILE = '<full path to this folder>\errors_nametests',
    TABLOCK
    )
GO
