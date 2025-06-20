# Database Partition Management Strategy

This document outlines the strategy for managing range partitions for the `claim_line_items` and `claims_production` tables in the claims processing database.

## `claim_line_items` Table

Partitions for `claim_line_items` are created monthly based on the `service_date` column.

### Example DDL for Future Partitions (`claim_line_items`)

The following are examples for creating partitions for the 6 months starting from April 2024:

```sql
-- April 2024 Partition
CREATE TABLE claim_line_items_y2024m04 PARTITION OF claim_line_items
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

-- May 2024 Partition
CREATE TABLE claim_line_items_y2024m05 PARTITION OF claim_line_items
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');

-- June 2024 Partition
CREATE TABLE claim_line_items_y2024m06 PARTITION OF claim_line_items
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');

-- July 2024 Partition
CREATE TABLE claim_line_items_y2024m07 PARTITION OF claim_line_items
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');

-- August 2024 Partition
CREATE TABLE claim_line_items_y2024m08 PARTITION OF claim_line_items
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');

-- September 2024 Partition
CREATE TABLE claim_line_items_y2024m09 PARTITION OF claim_line_items
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
```

## `claims_production` Table

Partitions for `claims_production` are currently created yearly based on the `service_from_date` column.

### Example DDL for Future Partitions (`claims_production`)

The following are examples for creating partitions for the 3 years starting from Y2026:

```sql
-- Y2026 Partition
CREATE TABLE claims_production_y2026 PARTITION OF claims_production
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Y2027 Partition
CREATE TABLE claims_production_y2027 PARTITION OF claims_production
    FOR VALUES FROM ('2027-01-01') TO ('2028-01-01');

-- Y2028 Partition
CREATE TABLE claims_production_y2028 PARTITION OF claims_production
    FOR VALUES FROM ('2028-01-01') TO ('2029-01-01');
```

## General Partition Management Strategy

### 1. Introduction

This document outlines the strategy for managing range partitions for the `claim_line_items` and `claims_production` tables in the claims processing database. Partitioning is implemented to improve query performance, simplify data management (like archiving and purging), and enhance overall system maintainability as data volumes grow.

### 2. Partition Creation Strategy

Proactive creation of future partitions is crucial to ensure data can always be inserted and to maintain system stability.

#### 2.1. `claim_line_items` Table

*   **Partitioning Scheme:** Monthly, by the `service_date` column.
*   **Naming Convention:** `claim_line_items_yYYYYmm` (e.g., `claim_line_items_y2024m04` for April 2024).
*   **Creation Schedule:**
    *   New partitions should be created at least 1-2 months in advance of the period they cover.
    *   **Recommendation:** Implement a monthly scheduled task (e.g., a cron job or a database scheduled job) that runs around the 15th of each month. This task should automatically create the partition for the month that is two months ahead.
        *   Example: On January 15th, the script creates the partition for March. On February 15th, it creates the partition for April.
    *   This ensures there is always a buffer and reduces the risk of ingestion failures due to missing partitions.

#### 2.2. `claims_production` Table

*   **Partitioning Scheme:** Currently Yearly, by the `service_from_date` column.
*   **Naming Convention:** `claims_production_yYYYY` (e.g., `claims_production_y2026` for the year 2026).
*   **Creation Schedule (Yearly):**
    *   New partitions should be created well in advance, ideally 3-6 months before the new year begins.
    *   **Recommendation:** Implement a yearly scheduled task that runs, for example, in September or October of each year. This task should create the partition for the *next* calendar year.
        *   Example: In September 2025, the script creates the partition for the year 2026 (`FOR VALUES FROM ('2026-01-01') TO ('2027-01-01')`).
*   **Alternative Considerations for `claims_production`:**
    *   If data volume in the `claims_production` table becomes extremely high, or if query patterns show a clear benefit from more granular partitioning (e.g., frequent queries on specific quarters or months), this table could also be partitioned more frequently (e.g., quarterly or monthly).
    *   This would require adjusting the DDL for the parent table's partitioning scheme, the partition creation DDL, and the automation schedule accordingly.

### 3. Old Partition Management

A clear data retention policy must be established based on business and regulatory requirements.

*   **Archival:**
    *   Partitions containing data older than the defined retention period should be considered for archival.
    *   Archival can be performed using tools like `pg_dump` to dump the specific partition table:
        `pg_dump -t partition_name > partition_name_archive.sql`
    *   The resulting dump files should be stored securely in a designated archive location (e.g., cloud storage, tape backup).
*   **Verification:** Before dropping any partition, ensure the archival process was successful and the archived data is valid and restorable.
*   **Dropping Partitions:**
    *   Once a partition is successfully archived and verified (and is no longer needed for active querying), it can be removed from the database to reclaim disk space.
    *   This is a two-step process:
        1.  `DETACH PARTITION partition_name FROM parent_table_name;`
        2.  `DROP TABLE partition_name;`
    *   **Caution:** Dropping tables is an irreversible operation. Always ensure backups are in place and the correct partition is being targeted. Perform these operations during maintenance windows if possible.

### 4. Monitoring

Regular monitoring is essential to ensure the partitioning strategy remains effective:

*   **Disk Space:** Monitor overall disk space usage for the tablespaces and database where these partitioned tables reside.
*   **Partition Count:** Keep an eye on the number of partitions, especially if a very large number of partitions could impact planning time (though this is less of an issue with modern PostgreSQL versions for range partitioning).
    *   Example Query: `SELECT count(*) FROM pg_catalog.pg_partitions WHERE partitiontablename IN ('claim_line_items', 'claims_production');` (Note: `pg_partitions` is a view, actual catalog might be `pg_class` joined with `pg_inherits`). Or more simply, query `information_schema.tables` for tables matching the partition naming pattern.
*   **Query Performance:** Periodically run `EXPLAIN` on common queries against the partitioned tables to ensure the query planner is effectively using partition pruning (i.e., only scanning relevant partitions).
*   **Ingestion Logs:** Monitor application and database logs for any errors related to data insertion, particularly "no partition found" errors, which would indicate a failure in the proactive partition creation process.

### 5. Automation

*   **Partition Creation:** The creation of new partitions **must be automated**. Manual creation is prone to human error and can be easily forgotten, leading to data ingestion failures.
    *   Use cron jobs (for Linux/macOS) or Task Scheduler (for Windows) to execute SQL scripts that create the necessary partitions based on the schedules defined above.
    *   Database-native schedulers (like `pg_cron` for PostgreSQL, or similar features in other databases) can also be used.
*   **Archival/Dropping (Optional Automation):** While archival and dropping can also be automated, it often involves more complex decision-making and verification steps. Initial implementations might involve semi-automated scripts with manual checks before final execution.

### 6. Error Handling for Ingestion

*   If data is attempted to be inserted into a partitioned table for which no suitable partition exists (i.e., the value of the partitioning key falls outside the range of any existing partition), the insert operation will fail with an error (e.g., "no partition of relation found for row").
*   The proactive creation of partitions as outlined in Section 2 is the primary defense against this.
*   While PostgreSQL (version 11 and later) supports a `DEFAULT` partition to catch data that doesn't fit into any other defined partition, relying on this for routine operations is generally not recommended as it can become a performance bottleneck and a data management challenge. It's better to ensure specific range partitions are always available.
```
