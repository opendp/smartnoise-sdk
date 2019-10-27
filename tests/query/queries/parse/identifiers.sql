-- test different identifier syntaxes and aliases
-- all of these should succeed.  see *_fail.sql for failure cases
SELECT FOO FROM BAR;
SELECT FOO AS BAZ FROM BAR;
SELECT Foo FROM bar;
SELECT [Employee Name] FROM BAR;
SELECT [Employee Name], [Employee Details] FROM BAR;
SELECT [Employee Name] AS foo, [Employee Details] FROM BAR;
SELECT [Employee Name] AS bax, [Employee Details] AS baz FROM BAR;
SELECT [Employee Name], Ename FROM BAR, BAZ;
SELECT [Employee Last Name] AS ELN FROM BAZ;
SELECT alias.colname FROM schema.tablename;
SELECT alias.colname FROM dbname.schema.tablename;
SELECT prod.[Employee Name] AS enam FROM [Data Files].dbo.[Product List];
SELECT [Employee Name] FROM [Data].dbo.[Columns] AS dboc;
SELECT [Employee Name] AS Ename, baz.[Employee ID] AS eid FROM [Data].dbo.[Columns] AS dboc, Foo AS baz;
SELECT * FROM BAR;
SELECT BAR.*, BAZ.ABC FROM BAR, BAZ;
