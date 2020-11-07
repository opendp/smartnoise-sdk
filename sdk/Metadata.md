# Data Source Metadata

To perform processing over tabular data, the system needs some metadata with sensitivity, identifiers, and other information needed for differential private processing.

This metadata is typically loaded directly from a YAML file supplied by the data curator, or read from a database table stored with the source data.

## YAML Metadata

The YAML metadata format looks something like this:

```yaml
MyDatabase:
    MyTable:
        max_ids: 1
        user_id:
            private_id: True
            type: int
        age:
            type: int
            lower: 0
            upper: 100
```
The root node is a collection of tables that all exist in the same namespace. The root node can also be referred to as a "database" or a "schema".

Each `schema` node can have multiple `table` nodes.  Tables represent tabular data.

Each `table` node can have an assortment of properties that can be set by the data curator to control per-table differential privacy.  In addition, each `table` node has multiple `column` nodes.

Each `column` node has attributes specifying the atomic datatype for use in differential privacy, sensitivity bounds, and more.

### Table Options

The following options can be set for each table in the collection.

#### Hints

The following options are hints that may be used by the system to optimize processing.  These are optional

* `rowcount`: Approximate number of rows in the table.  Default is `0`, meaning that the rowcount is not supplied.  If set, this count should be an approximate count that preserves privacy.
* `rows_exact`: Exact number of rows in the table.  Default is `None`.  If set, the system must take care not to expose this count to untrusted parties.

#### Overrides

In many cases, the underlying database engine will be configured to enforce constraints that impact differential privacy.  In these cases, the data curator can inform the system that these constraints are already enforced, allowing those steps to be optimized in the differential privacy layer.

These overrides should be used with caution, because they may affect privacy if used incorrectly.

* `max_ids`: Integer.  Default is `1`.  Specifies how many rows each unique user can appear in.   If any user appears in more rows than specified, the system will randomly sample to enforce this limit (see `sample_max_ids`).
* `row_privacy`: Boolean.  Default is `False`. Tells the system to treat each row as being a single individual.  This is common with social science datasets. 
* `sample_max_ids`: Boolean.  Default is `True`.  If the data curator can be certain that each user appears at most `max_ids` times in the table, this setting can be enabled to skip the reservoir sampling step.
* `clamp_counts`: Boolean.  Default is `False`.  Differentially private counts can sometimes be negative.  Setting this option to `True` will clamp negative counts to be `0`.  Does not affect privacy, but may impact utility.
* `clamp_columns`: Boolean.  Default is `True`.  By default, the system clamps all input data to ensure that it falls within the `lower` and `upper` bounds specified for that column.  If the data curator can be certain that the data never fall outside the specified ranges, this step can be disabled.
* `use_dpsu`: Boolean.  Default is `False`.  Tells the system to use Differential Private Set Union for censoring of rare dimensions.  Does not impact privacy.

## Column Options

* `type`: Required. This type attribute indicates the simple type for all values in the column. Type may be one of “int”, “float”, “string”, “boolean”, or “date”. The “date” type includes date or time types.  If type is set to "unknown", the column will be ignored by the system.
* `private_key`: Boolean.  Default is `False`.  indicates that this column is the private identifier (e.g. “UserID”, “Household”).  This column is optional.  Only columns which have private_id set to ‘true’ are treated as individuals subject to privacy protection.
* `lower`: Valid on numeric columns.  Specifies the lower bound for values in this column.
* `upper`: Valid on numeric columns.  Specifies the upper bound for values in this column.
* `cardinality`: Integer.  This is an optional hint, valid on columns intended to be used as categories or keys in a GROUP BY. Specifies the approximate number of distinct keys in this column.

## CollectionMetadata

The `CollectionMetadata` object is the in-memory metadata that gets loaded from storage.  It is supplied to `PrivateReader` to control query processing.  It is intended for read-only use.  To change properties in this object, change the underlying metadata file or database.

```python
database = CollectionMetadata.from_file("PUMS.yaml")
for table in database:
    for column in table:
        if column.is_key:
            print("Table {0} has key {1}".format(table.name, column.name))
```

## Other Considerations

The metadata described here is concerned only with differentially private processing, and is agnostic to storage engine (e.g. Spark, SQL, CSV).  Engine-specific metadata, such as database connection strings or credentials, are beyond the scope of this metadata.

The root of the metadata is a collection, which represents a collection of named tabular datasets.  Tabular datasets in a collection may be joined together, and budget is shared across all tables in the collection sharing a private identifier.

A `table` element must specify at least one child column.  If the data curator chooses not to expose specific columns in the source table via metadata, the existence of these columns is not revealed to the analyst, and analysts may not add references to data source columns not exposed in the metadata.

The table name is the identifier used to reference the tabular dataset.  The name restrictions will depend on the semantics of the data source and implementation.  For example, a SQL database may support a single dot-separated namespace name (e.g. “dbo.TableName”), while a CSV file encoded with UTF-8 may support arbitrary Unicode table names.

If present, max_ids must be the same for all tables that share a private_id.

Although row-level privacy is often assumed to be default behavior in the literature, we require this to be explicitly opted in, because incorrect assumption will compromise privacy.  If row_privacy is true, max_ids must be absent or set to 1.

Queries on tables with row_privacy may query only one table at a time.  Joins or other combinations that include more than one relation with row_privacy on any of the relations, are not allowed.

The analyst may not add columns not specified in the metadata.

Expressions in queries may combine and/or change types of values.

Note that `private_id` is not necessarily the same as the primary key.  For example, an “Orders” table might have a compound primary key consisting of “CustomerID” and “OrderID”.  In this case, orders are not private individuals, so we would specify private_id = `True` on the CustomerID, but not on the OrderID.

If `private_id` is not set on any column in the table, and row_privacy is false at the table level, no queries may be performed against this table.  If row_privacy is set to true, private_id need not be set, but joins will be disabled, and budget will be shared across all queries touching any table in the collection.

If private_id is set on more than one column in a dataset, the combination of columns is considered to be the compounded private identifier.  This should be uncommon.

We require that the same `private_id` be used across all tables in a collection, because budget is shared across all queries that access the same private individuals.  In some cases, the data curator may wish to allow data to be privatized based on multiple alternative identifiers.  For example, a collection of datasets might be keyed by both CustomerID and CompanyID, and the data curator may wish to allow analysts to choose between one or the other.  To support this scenario, the data curator can supply two different metadata files.

The data curator should take care not to use the actual min and max values from the dataset, when setting `lower` and `upper`, if these are sensitive, but instead should use a domain specific lower and upper (e.g. 0-100 for age).

If `lower` and `upper` are not provided, numeric aggregates such as SUM, AVG, STDDEV, will be unavailable for that column.

The collection element supports an optional `engine` attribute, which specifies a set of name disambiguation rules to be used.  For example, some data sources may be case-sensitive, such that ‘CustomerID’ and ‘customerID’ refer to two different columns. To provide deterministic evaluation of metadata, implementations must be able to determine how to handle case-sensitivity, character sets, namespaces, and escaping of special characters such as spaces in column names.

If the `engine` attribute is not specified, implementations may define implementation-specific name disambiguation rules, presumably tied to a very limited number of supported data sources.  Implementations should reject data sources with unknown engine specified.

Collection names, and switching between collections, are implementation-dependent and out of scope for this document.

In some cases, implementation may map the attribute names specified above to avoid collision with reserved keywords.  For example, ‘type’ is a reserved keyword in some programming languages, so in-memory objects will use a different attribute name, such as ‘val_type’.  Implementations may choose to serialize using different conventions, such as camel casing or snake casing.  It is not expected that serializations from one implementation will be used in another.

All typed columns are assumed to allow NULL or missing values.  In cases where the data curator knows that missing values are impossible, it may be desirable to allow specification of a ‘no_nulls’ attribute, to improve some calculation.  This is out of scope for this document.


## Example Metadata

The following is an example of a collection containing 3 tables, representing Crashes, Rollouts, and Census for a population of devices.  The collection name is “Telemetry”, and names are serialized as headings.

```yaml
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
        cardinality: 12
        type: string
      Region:
        cardinality: 13
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
        cardinality: 100
      Memory:
        type: string
        cardinality: 1000
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
```
