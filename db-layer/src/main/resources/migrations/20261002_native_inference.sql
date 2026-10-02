-- Apply once after 20261002_cluster_routing.sql, before enabling central native inference.
ALTER TABLE `asyncTaskOrchestrator`
  ADD COLUMN `llm` longtext,
  ADD COLUMN `nativeResult` longtext,
  ADD COLUMN `nativeState` varchar(20) DEFAULT NULL,
  ADD COLUMN `nativeDeadline` bigint DEFAULT NULL,
  ADD COLUMN `nativeFiles` mediumtext,
  ADD COLUMN `nativeBindings` mediumtext,
  ADD COLUMN `binaryArrayFiles` mediumtext;
ALTER TABLE `hostMapping` ADD COLUMN `assignedNonce` char(36) DEFAULT NULL;

CREATE TABLE IF NOT EXISTS `nativeAffinityOwner` (
  `bindingId` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `hostId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `state` varchar(20) NOT NULL,
  `files` mediumtext,
  `lastUsed` bigint DEFAULT NULL,
  `lastTask` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `lastPosition` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`bindingId`),
  KEY `nativeAffinityOwner_hostId_IDX` (`hostId`),
  KEY `nativeAffinityOwner_hostState_IDX` (`hostId`,`state`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS `nativePositionClaim` (
  `claimId` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `taskUuid` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `bindingId` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `position` varchar(32) NOT NULL,
  PRIMARY KEY (`claimId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

CREATE TABLE IF NOT EXISTS `nativeTaskDelivery` (
  `uuid` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `taskUuid` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `hostId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `deliveredAt` bigint NOT NULL,
  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
