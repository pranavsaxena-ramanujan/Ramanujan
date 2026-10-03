-- Apply once to an existing database before deploying cluster-aware services.
-- Existing rows remain NULL and retain untagged/legacy routing.
ALTER TABLE `asyncTaskOrchestrator`
  ADD COLUMN `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL;
ALTER TABLE `asyncTaskMiddleware`
  ADD COLUMN `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL;
ALTER TABLE `dagElementMetadata`
  ADD COLUMN `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL;
ALTER TABLE `availableHost`
  ADD COLUMN `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL;
ALTER TABLE `hostMapping`
  ADD COLUMN `clusterId` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL;
