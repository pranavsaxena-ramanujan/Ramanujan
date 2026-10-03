CREATE TABLE `nativeAffinityOwner` (
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
