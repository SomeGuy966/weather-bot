# Data Parsing

## Source of Data
This document discusses the methods I use to parse data, and how I arrived to those methods.

https://mesonet.agron.iastate.edu/request/asos/1min.phtml

The link above leads to a portal that the Iowa Environmental Mesonet (IEM) uses to share with the public 
1-minute observational data for weather stations across the country. This is where I get all 
the data to train my model.

IEM allows you to select the features you want in your data from a list of 17 of them. I only chose
10 of these to use in my data.

Regardless, let's go through each available feature and explain what it is; then go through my selected
features and explain why I chose them.



### Initial Data Reformatting

Because we have so much data, it makes more sense to save it as a .parquet file. This is for a few reasons:

- **Columnar storage format**  
  Parquet stores data by column rather than by row, enabling highly efficient read patterns when you only need a subset of columns.

- **Built-in compression and encoding**  
  Each column can use an optimal compression algorithm (Snappy, GZip, etc.) and encoding (RLE, dictionary), often reducing file sizes by 60–90 %.

- **Predicate pushdown**  
  Filters applied at read time (e.g. `date >= '2025-01-01'`) are pushed into the Parquet reader, so only matching row-groups are loaded into memory.

- **Partitioning support**  
  You can organize Parquet files into directory hierarchies (e.g. `year=2025/month=06/day=23`), making it trivial to scan or skip entire date ranges.

- **Schema enforcement and evolution**  
  Parquet stores type and schema metadata, so you automatically catch unexpected changes (missing columns, type mismatches) and can evolve schemas over time.

- **Vectorized I/O & analytics**  
  Parquet readers (in Spark, pandas, Dask, etc.) load entire column chunks in bulk, maximizing CPU and memory throughput for analytical workloads.

- **Interoperability with data ecosystems**  
  Parquet is the de facto format for data lakes and big-data platforms (Hive, Presto, Athena, Databricks), ensuring compatibility across tools.

- **Faster serialization/deserialization**  
  Compared to CSV or text, Parquet’s binary format parses much more quickly, reducing ETL runtimes.

- **Rich metadata and statistics**  
  Parquet files include per-column min/max statistics, enabling efficient skipping of irrelevant data without reading page contents.

- **Support for nested and complex types**  
  If you ever need to store structured arrays or maps, Parquet handles them natively—plain text formats do not.

- **Consistent checksums and integrity checks**  
  Parquet includes built-in CRC checks for data pages, helping detect corruption early.

By switching to Parquet, we gain significant speed, storage, compatibility, and robustness—critical when managing millions of rows of weather observations.  



### All Available Features Explanation

| Column       | Description                                   | Units/Format                    | Notes                                                       |
|--------------|-----------------------------------------------|---------------------------------|-------------------------------------------------------------|
| **tmpf**     | Air temperature                               | °F                              | ASOS “air temp” (≈2 m above ground)                         |
| **dwpf**     | Dew-point temperature                         | °F                              | ASOS “dew point” (≈2 m above ground)                        |
| **sknt**     | Wind speed                                    | knots                           | One-minute average                                          |
| **drct**     | Wind direction                                | degrees true (from north)       | One-minute average                                          |
| **gust_sknt**| Peak wind gust speed                          | knots                           | Highest 5-second gust in the minute                         |
| **gust_drct**| Direction of that peak gust                   | degrees true                    | “PK WND” remark from METAR                                   |
| **vis1_coeff** | Visibility extinction coefficient (sensor 1) | unitless                        | Raw optical coefficient; larger → clearer air               |
| **vis1_nd**  | Visibility 1 data flag                        | “M” = measured, “N” = no data   | Use to detect sensor 1 failures                             |
| **vis2_coeff** | Visibility extinction coefficient (sensor 2) | unitless                        | Second redundant visibility sensor                          |
| **vis2_nd**  | Visibility 2 data flag                        | “M” = measured, “N” = no data   |                                                             |
| **vis3_coeff** | Visibility extinction coefficient (sensor 3) | unitless                        | Third redundant visibility sensor                           |
| **vis3_nd**  | Visibility 3 data flag                        | “M” = measured, “N” = no data   |                                                             |
| **ptype**    | Precipitation type code                       | METAR code (e.g. “NP”, “RA”)     | NP = none, RA = rain, SN = snow, etc.                      |
| **precip**   | One-minute precipitation total                | inches                          | Reset each minute; melted snow/rain mix                     |
| **pres1**    | Station pressure (sensor 1)                   | in Hg                           | One of three independent barometers                         |
| **pres2**    | Station pressure (sensor 2)                   | in Hg                           |                                                             |
| **pres3**    | Station pressure (sensor 3)                   | in Hg                           |                                                             |


**Note:** 
The IEM portal has features like pres1, pres2, and pres3 to ensure redundancy; in case one of the
sensors breaks, you can always fall back on the other two.

However, in practice the sensors rarely malfunction (I believe, I haven't checked to make sure),
so for the purposes of this trading bot we only use vis1_coeff and pres1.


### Weather Elements used in Our Model
| **Field**    | **Bytes** | **Description**                                          | **Raw Units**               | **Parsing / Conversion**                                                       |
|--------------|-----------|----------------------------------------------------------|-----------------------------|---------------------------------------------------------------------------------|
| **tmpf**     | N/A       | Air temperature                                          | °F                          | Float; missing “M” → `NaN`                                                      |
| **dwpf**     | N/A       | Dew‐point temperature                                    | °F                          | Float; missing “M” → `NaN`                                                      |
| **sknt**     | N/A       | Wind speed (1-min average)                               | knots                       | Float; zero padded; missing “M” → `NaN`                                         |
| **drct**     | N/A       | Wind direction (1-min average)                           | degrees true (0–360°)       | Integer; “VRB” or 999 → `NaN`                                                   |
| **vis1_coeff** | N/A     | Visibility extinction coefficient (sensor 1)              | unitless                    | Float; larger → clearer air; missing/“ND” → `NaN`                                |
| **ptype**    | N/A       | Precipitation type code                                  | METAR code (`NP`, `RA`, etc.) | Categorical string; “NP” = none                                                  |
| **precip**   | N/A       | 1-minute precipitation total                             | inches                      | Float; reset each minute; missing → `0.0` or `NaN` as appropriate               |
| **pres1**    | N/A       | Station pressure (sensor 1)                              | inches of mercury (in Hg)   | Float; missing → `NaN`                                                          |
| **pres2**    | N/A       | Station pressure (sensor 2)                              | inches of mercury (in Hg)   | Float; missing → `NaN`                                                          |
| **pres3**    | N/A       | Station pressure (sensor 3)                              | inches of mercury (in Hg)   | Float; missing → `NaN`                                                          |



&nbsp;


## Feature Engineering


