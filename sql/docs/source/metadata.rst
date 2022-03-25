########
Metadata
########

To ensure differential privacy over tabular data, the system needs some metadata with sensitivity, identifiers, and other information about the data source.

This metadata is loaded from a YAML file path during instantiation of a private SQL connections.  The same metadata is used for all queries issued against that instantiated connection.

.. code-block:: python

  import pandas as pd
  from snsql import *
  pums = df.read_csv('PUMS.csv')
  meta_path = 'PUMS.yaml'

  reader = from_df(pums, privacy=Privacy(), metadata=meta_path)


YAML Format
-----------

The YAML metadata format looks like this:

.. code-block:: yaml

  "":
    MySchema:
        MyTable:
            max_ids: 1
            user_id:
                private_id: True
                type: int
            age:
                type: int
                lower: 0
                upper: 100

The root node is a collection of tables that all exist in the same namespace. The root node can also be referred to as a "database" or a "schema".

Each ``schema`` node can have multiple ``table`` nodes.  Tables represent tabular data.

Each ``table`` node can have an assortment of properties that can be set by the data curator to control per-table differential privacy.  In addition, each ``table`` node has multiple ``column`` nodes.

Each ``column`` node has attributes specifying the atomic datatype for use in differential privacy, sensitivity bounds, and more.

Dictionary Format
-----------------

Although loading from file path is preferred, you can also pass in the metadata as a nested dictionary.

.. code-block:: python

  import pandas as pd
  from snsql import *
  pums = df.read_csv('PUMS.csv')

  metadata = {
    '':{
      'MySchema': {
        'MyTable': {
          'max_ids': 1,
          'row_privacy': False,
          'user_id': {
            'name': 'user_id',
            'type': 'int',
            'private_id': True
          },
          'age': {
            'name': 'age',
            'type': 'int',
            'upper': 100,
            'lower': 0
          }
        }
      }
    }
  }
  reader = from_df(pums, privacy=Privacy(), metadata=meta_path)


Table Options
-------------

In many cases, the underlying database engine will be configured to enforce constraints that impact differential privacy.  In these cases, the data curator can inform the system that these constraints are already enforced, allowing those steps to be optimized in the differential privacy layer.

These overrides should be used with caution, because they may affect privacy if used incorrectly.

* ``max_ids``: Integer.  Default is ``1``.  Specifies how many rows each unique user can appear in.   If any user appears in more rows than specified, the system will randomly sample to enforce this limit (see ``sample_max_ids``).
* ``row_privacy``: Boolean.  Default is ``False``. Tells the system to treat each row as being a single individual.  This is common with social science datasets. 
* ``sample_max_ids``: Boolean.  Default is ``True``.  If the data curator can be certain that each user appears at most ``max_ids`` times in the table, this setting can be enabled to skip the reservoir sampling step.  This may be the case if the ``private_id`` is the sole primary key of the database.  This will not be the case when the source data are in a denormalized fact table where a single ``private_id`` can appear in multiple rows.
* ``censor_dims``: Boolean.  Default is ``True``.  Drops GROUP BY output rows that might reveal the presence of individuals in the database.  For example, a query doing GROUP BY over last names would reveal the existence of an individual with a rare last name.  Data curators may override this setting if the dimensions are public or non-sensitive.
* ``clamp_counts``: Boolean.  Default is ``False``.  Differentially private counts can sometimes be negative.  Setting this option to ``True`` will clamp negative counts to be 0.  Does not affect privacy, but may impact utility.
* ``clamp_columns``: Boolean.  Default is ``True``.  By default, the system clamps all input data to ensure that it falls within the ``lower`` and ``upper`` bounds specified for that column.  If the data curator can be certain that the data never fall outside the specified ranges, this step can be disabled.
* ``use_dpsu``: Boolean.  Default is ``False``.  Tells the system to use Differential Private Set Union for censoring of rare dimensions.  Does not impact privacy.

Column Options
--------------

* ``type``: Required. The type attribute indicates the simple type for all values in the column. Type may be one of “int”, “float”, “string”, “boolean”, or “datetime”. The “datetime” type includes date or time types.  If type is set to "unknown", the column will be ignored by the system.
* ``private_id``: Boolean.  Default is ``False``.  indicates that this column is the private identifier (e.g. “UserID”, “Household”).  This column is optional.  Only columns which have private_id set to true are treated as individuals subject to privacy protection.
* ``lower``: Valid on numeric columns.  Specifies the lower bound for values in this column.
* ``upper``: Valid on numeric columns.  Specifies the upper bound for values in this column.
* ``nullable``: Boolean.  Default is ``True``.  Indicates that this column can contain null values.  If set to ``False``, the system will assume that all values are set.  This is useful when the data curator knows that all values are set, and will allow some budget to be preserved by sharing counts across columns.
* ``missing_value``: A value of the same type as the ``type`` for this column.  Default is ``None``.  If set, the system will replace NULL with the specified value, ensuring that all values are set.  If set, ``nullable`` will be treated as ``False``, regardless of its value.
* ``sensitivity``: The sensitivity to be used when releasing sums from this column.  Default is ``None``.  If not set, the system will compute the sensitivity from upper and lower bounds.  If ``sensitivity`` is set, the upper and lower bounds will be ignored for sensitivity, and this value will be used.  The upper and lower bounds will still be used to clamp the columns. If this value is set, and no bounds are provided, the metadata must specify ``clamp_columns`` as ``False``. Note that counts will always use a sensitivity of 1, regardless of the value of this attribute.

Database Names
--------------

Any table or column objects referenced in queries must resolve to objects described in the metadata.  For example, the following describes a table named "Sales" in a database named "Finance"

.. code-block:: yaml

  Finance:
    Sales:
        Orders:
            row_privacy: True
            SaleAmount:
                type: int
                lower: 0
                upper: 100000

The following two queries will work, and are equivalent:

.. code-block:: sql

  SELECT SUM(SaleAmount) FROM Sales.Orders;
  SELECT SUM(SaleAmount) FROM Finance.Sales.Orders;

If the database engine being used is not case-sensitive, the query can also use lower case.

.. code-block:: sql

  SELECT SUM(saleamount) FROM sales.orders;

Note that the object names used in the SQL query must match the metadata, but the database name need not match the database name of the underlying connection.  For example, if the the existing connection is to a database named "FinanceTest", the following query will work:

.. code-block:: sql

  SELECT SUM(SaleAmount) FROM Sales.Orders;

despite the database name in the metadata being "Finance".  However, in this scenario, the following query will not work:

.. code-block:: sql

  SELECT SUM(SaleAmount) FROM Finance.Sales.Orders;

because the query will be executed against the connection using a database name that doesn't match the currently connected database.

Data curators may choose to leave the database name blank, in which case any database name will match.  To make a blank key in YAML, use double quotes:

.. code-block:: yaml

  "":
    Sales:
        Orders:
            row_privacy: True
            SaleAmount:
                type: int
                lower: 0
                upper: 100000

With the above YAML, all three queries will work:

.. code-block:: sql

  SELECT SUM(SaleAmount) FROM Sales.Orders;
  SELECT SUM(SaleAmount) FROM Finance.Sales.Orders;
  SELECT SUM(SaleAmount) FROM FinanceTest.Sales.Orders;


Special Characters
------------------

The SQL-92 specification allows object identifiers to include special characters such as hyphens or spaces.  For example:

.. code-block:: yaml

  "":
    Sales:
        Backlog-Orders:
            row_privacy: True
            Sale-Amount:
                type: int
                lower: 0
                upper: 100000

A query against this table would look like:

.. code-block:: sql

  SELECT SUM("Sale-Amount") FROM Sales."Backlog-Orders";

In the above example, ``Backlog-Orders`` and ``Sale-Amount`` are not escaped in the metadata, which might give unexpected results.  For example, the lowercase versions of these identifiers would also work.  When specifying objects in metadata that need to be escaped in the SQL query, it's a good idea to escape the objects in the metadata:

.. code-block:: yaml

  "":
    Sales:
        '"Backlog-Orders"':
            row_privacy: True
            '"Sale-Amount"':
                type: int
                lower: 0
                upper: 100000

Note the single-quotes wrapping the identifiers escaped with double-quotes.  This is necessary to ensure the YAML parser preserves the escaping.

The above uses the SQL-92 syntax for quoting identifiers with special characters.  Some engines may not support this syntax, and may instead use backticks or square brackets.  You can use whatever syntax is appropriate for your engine.  For example, the following YAML is suitable when backticks are the preferred escape character:

.. code-block:: yaml

  "":
    Sales:
        "`Backlog-Orders`":
            row_privacy: True
            '`Sale-Amount`':
                type: int
                lower: 0
                upper: 100000

Since our escape character is not a double quote in this case, we can wrap the string for the YAML parser equivalently using either single or double quotes.

It's good practice to use the same escape character in the metadata as is preferred by your database engine, but it's not strictly necessary.  As long as the escape character used in the queries is approproate for the engine, the system should be able to fine the correct metadata.  For example, the following query will work fine against SQL Server, using square bracket escaping, despite the fact that the metadata uses backticks:

.. code-block:: sql

  SELECT SUM([Sale-Amount]) FROM Sales.[Backlog-Orders];


Other Considerations
--------------------

The metadata described here is concerned only with differentially private processing, and is agnostic to storage engine (e.g. Spark, SQL, CSV).  Engine-specific metadata, such as database connection strings or credentials, are beyond the scope of this metadata.

The root of the metadata is a collection, which represents a collection of named tabular datasets.  Tabular datasets in a collection may be joined together, and budget is shared across all tables in the collection sharing a private identifier.

A ``table`` element must specify at least one child column.  If the data curator chooses not to expose specific columns in the source table via metadata, the existence of these columns is not revealed to the analyst, and analysts may not add references to data source columns not exposed in the metadata.

The table name is the identifier used to reference the tabular dataset.  The name restrictions will depend on the semantics of the data source and implementation.  For example, a SQL database may support a single dot-separated namespace name (e.g. “dbo.TableName”), while a CSV file encoded with UTF-8 may support arbitrary Unicode table names.

If present, ``max_ids`` must be the same for all tables that share a ``private_id``.

Although row-level privacy is often assumed to be default behavior in the literature, we require this to be explicitly opted in, because incorrect assumption will compromise privacy.  If ``row_privacy`` is true, ``max_ids`` must be absent or set to 1.

Queries on tables with ``row_privacy`` may query only one table at a time.  Joins or other combinations that include more than one relation with ``row_privacy`` on any of the relations, are not allowed.

The analyst may not add columns not specified in the metadata.

Expressions in queries may combine and/or change types of values.

Note that ``private_id`` is not necessarily the same as the primary key.  For example, an “Orders” table might have a compound primary key consisting of “CustomerID” and “OrderID”.  In this case, orders are not private individuals, so we would specify private_id = ``True`` on the CustomerID, but not on the OrderID.

If ``private_id`` is not set on any column in the table, and ``row_privacy`` is false at the table level, no queries may be performed against this table.  If row_privacy is set to true, private_id need not be set, but joins will be disabled, and budget will be shared across all queries touching any table in the collection.

If private_id is set on more than one column in a dataset, the combination of columns is considered to be the compounded private identifier.  This should be uncommon.

We require that the same ``private_id`` be used across all tables in a collection, because budget is shared across all queries that access the same private individuals.  In some cases, the data curator may wish to allow data to be privatized based on multiple alternative identifiers.  For example, a collection of datasets might be keyed by both CustomerID and CompanyID, and the data curator may wish to allow analysts to choose between one or the other.  To support this scenario, the data curator can supply two different metadata files.

The data curator should take care not to use the actual min and max values from the dataset, when setting ``lower`` and ``upper``, if these are sensitive, but instead should use a domain specific lower and upper (e.g. 0-100 for age).

If ``lower`` and ``upper`` are not provided, numeric aggregates such as SUM, AVG, STDDEV, will be unavailable for that column.

The collection element supports an optional ``engine`` attribute, which specifies a set of name disambiguation rules to be used.  For example, some data sources may be case-sensitive, such that ‘CustomerID’ and ‘customerID’ refer to two different columns. To provide deterministic evaluation of metadata, implementations must be able to determine how to handle case-sensitivity, character sets, namespaces, and escaping of special characters such as spaces in column names.

If the ``engine`` attribute is not specified, implementations may define implementation-specific name disambiguation rules, presumably tied to a very limited number of supported data sources.  Implementations should reject data sources with unknown engine specified.

Collection names, and switching between collections, are implementation-dependent and out of scope for this document.

In some cases, implementation may map the attribute names specified above to avoid collision with reserved keywords.  For example, ‘type’ is a reserved keyword in some programming languages, so in-memory objects will use a different attribute name, such as ‘val_type’.  Implementations may choose to serialize using different conventions, such as camel casing or snake casing.  It is not expected that serializations from one implementation will be used in another.

All typed columns are assumed to allow NULL or missing values.  In cases where the data curator knows that missing values are impossible, it may be desirable to allow specification of a ‘no_nulls’ attribute, to improve some calculation.  This is out of scope for this document.


Example Metadata
----------------

The following is an example of a collection containing 3 tables, representing Crashes, Rollouts, and Census for a population of devices.  The collection name is “Telemetry”, and names are serialized as headings.

.. code-block:: yaml

  Collection:
    Telemetry:
      Crashes:
        rows: 103000
        Refurbished:
          type: boolean
        Temperature:
          type: float
          lower:  25.0
          upper:  65.0
        Building:
          type: string
        Region:
          type: string
        DeviceID:
          type: int
          private_id: True
        Crashes:
          type: int
          lower:  0
          upper:  10
      Census:
        DeviceID:
          type: int
          private_id: true
        OEM:
          type: string
        Memory:
          type: string
        Disk:
          type: int
          lower:  100
          upper:  10000
      Rollouts:
        DeviceID:
          type: int
          private_id: true
        RolloutID:
          type: int
        StartTrial:
          type: datetime
        EndTrial:
          type: datetime
        TrialGroup:
          type: int