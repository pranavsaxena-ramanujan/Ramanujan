-- ramanujan.asyncTaskOrchestrator definition

CREATE TABLE `asyncTaskOrchestrator` (
  `llm` longtext,
  `nativeResult` longtext,
  `nativeState` varchar(20) DEFAULT NULL,
  `nativeDeadline` bigint DEFAULT NULL,
  `nativeFiles` mediumtext,
  `nativeBindings` mediumtext,
  `binaryArrayFiles` mediumtext,
  `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `uuid` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `status` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `firstCommandId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `debug` varchar(6) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  KEY `asyncTaskOrchestrator_uuid_IDX` (`uuid`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;