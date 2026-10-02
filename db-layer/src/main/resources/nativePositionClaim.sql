CREATE TABLE `nativePositionClaim` (
  `claimId` char(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `taskUuid` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `bindingId` varchar(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `position` varchar(32) NOT NULL,
  PRIMARY KEY (`claimId`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
