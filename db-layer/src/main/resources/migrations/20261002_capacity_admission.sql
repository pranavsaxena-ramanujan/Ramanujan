-- Apply with an administrator account before enabling RAMANUJAN_CAPACITY_AWARE.
-- Existing host/task tables are intentionally unchanged.
CREATE TABLE IF NOT EXISTS capacityClusterLock (
    clusterId VARCHAR(100) NOT NULL PRIMARY KEY
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS workerCapacity (
    hostId VARCHAR(200) NOT NULL PRIMARY KEY,
    clusterId VARCHAR(100) NOT NULL,
    snapshot LONGTEXT NOT NULL,
    updatedAt BIGINT NOT NULL,
    INDEX workerCapacityCluster (clusterId)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS llmCapacityPlan (
    planId VARCHAR(36) NOT NULL PRIMARY KEY,
    clusterId VARCHAR(100) NOT NULL,
    plan LONGTEXT NOT NULL,
    updatedAt BIGINT NOT NULL,
    INDEX llmCapacityPlanCluster (clusterId)
) ENGINE=InnoDB;
