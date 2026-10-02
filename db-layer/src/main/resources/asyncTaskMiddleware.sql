
CREATE TABLE `asyncTaskMiddleware` (
  `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `taskId` varchar(100) DEFAULT NULL,
  `taskStatus` varchar(100) DEFAULT NULL,
  `result` mediumtext,
  KEY `asyncTaskMiddleware_taskId_IDX` (`taskId`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;