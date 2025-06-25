package in.ramanujan.rest.verticles.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import in.ramanujan.base.enums.Topics;
import in.ramanujan.base.pojo.CheckStatusQueueEventWithMetadata;
import in.ramanujan.base.pojo.MachineAssignmentTask;
import in.ramanujan.base.pojo.MachineAssignmentTaskWithMetadata;
import in.ramanujan.data.MachineRetryCallback;
import in.ramanujan.data.QueueingDao;
import in.ramanujan.data.queingDaoImpl.KafkaImpl;
import in.ramanujan.service.EventConsumer;
import io.vertx.core.AbstractVerticle;
import io.vertx.core.CompositeFuture;
import io.vertx.core.Future;
import io.vertx.core.Promise;
import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import lombok.Data;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.List;

@Component
public class ConsumerVerticle extends AbstractVerticle {

    @Autowired
    private EventConsumer eventConsumer;
    
    @Autowired
    private QueueingDao queueingDao;
    
    @Autowired
    private MachineRetryCallback machineRetryCallback;

    private ObjectMapper objectMapper = new ObjectMapper();

    Logger logger= LoggerFactory.getLogger(ConsumerVerticle.class);

    final String topicName = Topics.next_element_topic.name();
    final String machineAssignmentTopicName = Topics.run_dag.name();
    final Long pollingIntervalInMillis = 2_00L;

    @Override
    public void start(Promise<Void> startPromise) throws Exception {
        logger.info("Consumer is up");
        
        // Subscribe to next_element_topic
        queueingDao.subscribe().setHandler(handler -> {
           if(handler.succeeded()) {
               startConsumption();
               logger.info("Subscribed to " + topicName);
           } else {
               logger.error("Failed to subscribe to " + topicName, handler.cause());
           }
        });

        // Subscribe to run-dag topic for machine assignment retries
        queueingDao.subscribeMachineAssignment().setHandler(handler -> {
           if(handler.succeeded()) {
               startMachineAssignmentConsumption();
               logger.info("Subscribed to " + machineAssignmentTopicName);
           } else {
               logger.error("Failed to subscribe to " + machineAssignmentTopicName, handler.cause());
           }
        });

        startConsumption();
        startMachineAssignmentConsumption();
    }

    private void startConsumption() {
        final BlockingWrapper blockingWrapper = new BlockingWrapper();
        blockingWrapper.unblock();

        vertx.setPeriodic(pollingIntervalInMillis, handler -> {
            if(blockingWrapper.getBlocked()) {
                return;
            }
            blockingWrapper.block();
            queueingDao.consume().setHandler(consumeHandler -> {
                if(consumeHandler.succeeded()) {
                    List<Future> consumeFutureList = new ArrayList<>();
                    List<CheckStatusQueueEventWithMetadata> checkStatusQueueEventWithMetadataList = consumeHandler.result();
                    if(checkStatusQueueEventWithMetadataList == null) {
                        blockingWrapper.unblock();
                        return;
                    }
                    for(CheckStatusQueueEventWithMetadata checkStatusQueueEventWithMetadata : checkStatusQueueEventWithMetadataList) {
                        consumeFutureList.add(eventConsumer.consume(checkStatusQueueEventWithMetadata.getCheckStatusQueueEvent(), vertx));
                    }

                    CompositeFuture.all(consumeFutureList).setHandler(consumerListHandler -> {
                        if(checkStatusQueueEventWithMetadataList.size() == 0) {
                            blockingWrapper.unblock();
                            return;
                        }
                        Object metadata = (queueingDao.getClass() == KafkaImpl.class) ?
                                checkStatusQueueEventWithMetadataList.get(checkStatusQueueEventWithMetadataList.size() -1).getMetadata() :
                                getPubSubMetadata(checkStatusQueueEventWithMetadataList);
                        queueingDao.commit(metadata).setHandler(commitHandler -> {
                            blockingWrapper.unblock();
                        });
                    });
                } else {
                    logger.error(consumeHandler.cause());
                    blockingWrapper.unblock();
                }

            });
        });
    }

    private List<String> getPubSubMetadata(List<CheckStatusQueueEventWithMetadata> checkStatusQueueEventWithMetadataList) {
        List<String> list = new ArrayList<>();
        for(CheckStatusQueueEventWithMetadata checkStatusQueueEventWithMetadata : checkStatusQueueEventWithMetadataList) {
            list.add((String) checkStatusQueueEventWithMetadata.getMetadata());
        }
        return list;
    }

    private void startMachineAssignmentConsumption() {
        final BlockingWrapper blockingWrapper = new BlockingWrapper();
        blockingWrapper.unblock();

        vertx.setPeriodic(pollingIntervalInMillis, handler -> {
            if(blockingWrapper.getBlocked()) {
                return;
            }
            blockingWrapper.block();
            queueingDao.consumeMachineAssignment().setHandler(consumeHandler -> {
                if(consumeHandler.succeeded()) {
                    List<Future<Boolean>> retryFutureList = new ArrayList<>();
                    List<MachineAssignmentTaskWithMetadata> machineAssignmentTasksWithMetadata = consumeHandler.result();
                    if(machineAssignmentTasksWithMetadata == null || machineAssignmentTasksWithMetadata.isEmpty()) {
                        logger.error("Nothing to consume for machine retry");
                        blockingWrapper.unblock();
                        return;
                    }
                    
                    // Track successful assignments to commit them
                    List<MachineAssignmentTaskWithMetadata> successfulTasks = new ArrayList<>();
                    List<MachineAssignmentTaskWithMetadata> failedTasks = new ArrayList<>();
                    
                    for(MachineAssignmentTaskWithMetadata taskWithMetadata : machineAssignmentTasksWithMetadata) {
                        MachineAssignmentTask task = taskWithMetadata.getTask();
                        // Try to assign machine for each task
                        Future<Boolean> retryFuture = machineRetryCallback.retryMachineAssignment(task, vertx);
                        Future<Boolean> listAdded = Future.future();
                        retryFutureList.add(listAdded);
                        
                        retryFuture.setHandler(retryHandler -> {
                            if(retryHandler.succeeded() && Boolean.TRUE.equals(retryHandler.result())) {
                                // Machine assignment succeeded
                                successfulTasks.add(taskWithMetadata);
                                logger.info("Successfully assigned machine for task " + task.getAsyncId());
                                listAdded.complete();
                            } else {
                                // Machine assignment failed or future failed
                                failedTasks.add(taskWithMetadata);
                                if(retryHandler.failed()) {
                                    logger.error("Error retrying machine assignment for task " + 
                                                task.getAsyncId(), retryHandler.cause());
                                } else {
                                    logger.info("Failed to assign machine for task " + task.getAsyncId() + 
                                              ", will retry later");
                                }
                                listAdded.complete();
                            }
                        });
                    }

                    CompositeFuture.all(new ArrayList<>(retryFutureList)).setHandler(retryListHandler -> {
                        // Only commit the messages for tasks that were successfully assigned
                        if(successfulTasks.size() > 0) {
                            // Get the metadata for pubsub
                            Object metadata = getPubSubMetadataForSuccessfulTasks(machineAssignmentTasksWithMetadata, successfulTasks);
                            if(metadata != null) {
                                queueingDao.commitMachineAssignment(metadata).setHandler(commitHandler -> {
                                    logger.info("Committed " + successfulTasks.size() + 
                                              " successful machine assignments");
                                    blockingWrapper.unblock();
                                });
                            } else {
                                logger.info("No metadata to commit for successful tasks");
                                blockingWrapper.unblock();
                            }
                        } else {
                            logger.info("No successful machine assignments to commit");
                            blockingWrapper.unblock();
                        }
                    });
                } else {
                    logger.error("Failed to consume from machine assignment topic", consumeHandler.cause());
                    blockingWrapper.unblock();
                }
            });
        });
    }
    
    private Object getPubSubMetadataForSuccessfulTasks(List<MachineAssignmentTaskWithMetadata> allTasks, 
                                                     List<MachineAssignmentTaskWithMetadata> successfulTasks) {
        // For PubSub, we need to get the ackIds for successful tasks only
        if(queueingDao.getClass() == KafkaImpl.class) {
            // For Kafka, use the last successful task's metadata
            if(!successfulTasks.isEmpty()) {
                // Get the index of the last successful task in the original list
                int lastSuccessfulIndex = -1;
                for(int i = allTasks.size() - 1; i >= 0; i--) {
                    if(successfulTasks.contains(allTasks.get(i))) {
                        lastSuccessfulIndex = i;
                        break;
                    }
                }
                if(lastSuccessfulIndex >= 0) {
                    // Return metadata based on the index
                    return lastSuccessfulIndex;
                }
            }
        } else {
            // For PubSub, we need to collect the ackIds for all successful tasks
            List<String> ackIds = new ArrayList<>();
            for(MachineAssignmentTaskWithMetadata task : successfulTasks) {
                ackIds.add(task.getAckId());
            }
            if(!ackIds.isEmpty()) {
                return ackIds;
            }
        }
        return null;
    }

    @Data
    private class BlockingWrapper {
        private Boolean blocked;

        public void block() {
            blocked = true;
        }

        public void unblock() {
            blocked = false;
        }
    }

}
