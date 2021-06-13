CREATE DATABASE nametests;
\c nametests
CREATE TABLE nametests (educ int, "Educ" int, "3395" int, "select" float);
CREATE TABLE "NameTests" (educ int, "Educ" int, "3395" int, "select" float);
\copy nametests FROM 'nametests.csv' CSV;
\copy "NameTests" FROM 'nametests_case.csv' CSV;

CREATE DATABASE "2NameTests";
\c "2NameTests"
CREATE TABLE nametests (educ int, "Educ" int, "3395" int, "select" float);
\copy nametests FROM '2NameTests.csv' CSV;
