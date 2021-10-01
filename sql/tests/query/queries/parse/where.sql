SELECT X FROM Y WHERE RECENT = 'true';
SELECT X FROM Y WHERE COL1 < COL2 AND COL3 > COL4 OR COL5 >= COL3 AND RECENT != 'true';
SELECT X FROM Z WHERE NOT COL1 = BAZ;
SELECT X, Y, Z FROM BAR WHERE X NOT IN ( Y);
SELECT Sales FROM SalesTable WHERE Receipts IS NOT NULL;
SELECT Sales FROM SalesTable WHERE (5 + (Sales * 0.01)) > Gross;
SELECT Sales * 99.9 AS NewSales, SUM(Sales) AS Gross FROM SalesTable HAVING (5 + (Sales * 0.01)) > Gross;
SELECT Name FROM Invoices GROUP BY Name, Sales HAVING SalesDate = SalesDate OR Name >= 'Shipping' ORDER BY Total_Sales ASC;