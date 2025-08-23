"""
Streaming data connectors for Kafka, Kinesis, and Pub/Sub.
"""
import asyncio
import logging
from typing import Any, Dict, List, Tuple, Optional
import json
from datetime import datetime

from .base_connector import BaseConnector

logger = logging.getLogger(__name__)

class StreamingConnector(BaseConnector):
    """Connector for streaming data sources."""
    
    def __init__(self):
        self.supported_streams = {
            "kafka": self._handle_kafka,
            "kinesis": self._handle_kinesis,
            "pubsub": self._handle_pubsub
        }
    
    async def test_connection(self, connection_config: Dict[str, Any], 
                            auth_config: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Test streaming connection."""
        try:
            stream_type = connection_config.get("stream_type")
            if stream_type not in self.supported_streams:
                return False, f"Unsupported stream type: {stream_type}", {}
            
            handler = self.supported_streams[stream_type]
            success, error, details = await handler("test", connection_config, auth_config)
            
            return success, error, details
            
        except Exception as e:
            logger.error(f"Streaming connection test failed: {str(e)}")
            return False, str(e), {}
    
    async def create_connection(self, connection_config: Dict[str, Any], 
                              auth_config: Dict[str, Any]) -> Any:
        """Create streaming connection."""
        try:
            stream_type = connection_config.get("stream_type")
            handler = self.supported_streams[stream_type]
            success, connection, details = await handler("connect", connection_config, auth_config)
            
            if success:
                return connection
            else:
                raise Exception(f"Failed to create connection: {details}")
                
        except Exception as e:
            logger.error(f"Failed to create streaming connection: {str(e)}")
            raise
    
    async def discover_schema(self, connection: Any) -> List[Dict[str, Any]]:
        """Discover streaming schema by sampling messages."""
        try:
            stream_type = connection.get("stream_type")
            handler = self.supported_streams[stream_type]
            success, schemas, details = await handler("schema", connection, {})
            
            if success:
                return schemas
            else:
                raise Exception(f"Schema discovery failed: {schemas}")
                
        except Exception as e:
            logger.error(f"Streaming schema discovery failed: {str(e)}")
            return []
    
    async def read_data(self, connection: Any, query_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read data from streaming source."""
        try:
            stream_type = connection.get("stream_type")
            handler = self.supported_streams[stream_type]
            
            # Merge query config with connection config
            read_config = {**connection, **query_config}
            success, data, details = await handler("read", read_config, {})
            
            if success:
                return data
            else:
                raise Exception(f"Data reading failed: {data}")
                
        except Exception as e:
            logger.error(f"Streaming data reading failed: {str(e)}")
            raise
    
    async def _handle_kafka(self, operation: str, connection_config: Dict[str, Any], 
                          auth_config: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Handle Kafka operations."""
        try:
            # Import Kafka libraries (optional dependency)
            try:
                from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
                from kafka.errors import KafkaError
            except ImportError:
                return False, "aiokafka library not installed", {}
            
            bootstrap_servers = connection_config.get("bootstrap_servers", ["localhost:9092"])
            topic = connection_config.get("topic")
            group_id = connection_config.get("group_id", "pipeline-automation")
            
            if operation == "test":
                try:
                    # Test connection by creating a consumer
                    consumer = AIOKafkaConsumer(
                        bootstrap_servers=bootstrap_servers,
                        group_id=f"{group_id}-test",
                        auto_offset_reset='latest',
                        enable_auto_commit=False,
                        consumer_timeout_ms=5000
                    )
                    
                    await consumer.start()
                    
                    # Get cluster metadata
                    cluster = consumer._client.cluster
                    topics = cluster.topics()
                    
                    await consumer.stop()
                    
                    return True, None, {
                        "available_topics": list(topics),
                        "bootstrap_servers": bootstrap_servers
                    }
                    
                except Exception as e:
                    return False, f"Kafka connection failed: {str(e)}", {}
            
            elif operation == "connect":
                consumer = AIOKafkaConsumer(
                    topic,
                    bootstrap_servers=bootstrap_servers,
                    group_id=group_id,
                    auto_offset_reset='earliest',
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None
                )
                
                return True, {
                    "consumer": consumer,
                    "stream_type": "kafka",
                    "topic": topic,
                    "config": connection_config
                }, {}
            
            elif operation == "schema":
                consumer = connection_config.get("consumer")
                if not consumer:
                    return False, "No consumer available", {}
                
                # Sample messages to infer schema
                await consumer.start()
                
                sample_messages = []
                try:
                    # Consume a few messages for schema inference
                    async for msg in consumer:
                        if len(sample_messages) >= 10:
                            break
                        
                        if msg.value:
                            sample_messages.append(msg.value)
                        
                        # Timeout after 10 seconds
                        if len(sample_messages) == 0:
                            await asyncio.sleep(0.1)
                
                finally:
                    await consumer.stop()
                
                # Infer schema from sample messages
                if sample_messages:
                    schema = self._infer_message_schema(sample_messages)
                    return True, [{
                        "schema_name": "kafka_stream",
                        "table_name": connection_config.get("topic", "unknown_topic"),
                        "columns": schema
                    }], {"sample_count": len(sample_messages)}
                else:
                    return True, [], {"sample_count": 0}
            
            elif operation == "read":
                consumer = connection_config.get("consumer")
                limit = connection_config.get("limit", 100)
                timeout_seconds = connection_config.get("timeout", 30)
                
                if not consumer:
                    return False, "No consumer available", {}
                
                await consumer.start()
                
                messages = []
                start_time = asyncio.get_event_loop().time()
                
                try:
                    async for msg in consumer:
                        if len(messages) >= limit:
                            break
                        
                        if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                            break
                        
                        if msg.value:
                            message_data = {
                                "offset": msg.offset,
                                "partition": msg.partition,
                                "timestamp": msg.timestamp,
                                "key": msg.key.decode('utf-8') if msg.key else None,
                                "value": msg.value
                            }
                            messages.append(message_data)
                
                finally:
                    await consumer.stop()
                
                return True, messages, {"messages_read": len(messages)}
            
        except Exception as e:
            return False, str(e), {}
    
    async def _handle_kinesis(self, operation: str, connection_config: Dict[str, Any], 
                            auth_config: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Handle AWS Kinesis operations."""
        try:
            # Import AWS libraries (optional dependency)
            try:
                import boto3
                from botocore.exceptions import ClientError
            except ImportError:
                return False, "boto3 library not installed", {}
            
            region = connection_config.get("region", "us-east-1")
            stream_name = connection_config.get("stream_name")
            
            # AWS credentials from auth_config
            aws_access_key = auth_config.get("aws_access_key_id")
            aws_secret_key = auth_config.get("aws_secret_access_key")
            aws_session_token = auth_config.get("aws_session_token")
            
            if operation == "test":
                try:
                    session = boto3.Session(
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key,
                        aws_session_token=aws_session_token,
                        region_name=region
                    )
                    
                    kinesis = session.client('kinesis')
                    
                    # Describe stream to test connection
                    response = kinesis.describe_stream(StreamName=stream_name)
                    stream_info = response['StreamDescription']
                    
                    return True, None, {
                        "stream_status": stream_info['StreamStatus'],
                        "shard_count": len(stream_info['Shards']),
                        "retention_period": stream_info['RetentionPeriodHours']
                    }
                    
                except ClientError as e:
                    return False, f"Kinesis connection failed: {str(e)}", {}
            
            elif operation == "connect":
                session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    aws_session_token=aws_session_token,
                    region_name=region
                )
                
                kinesis = session.client('kinesis')
                
                return True, {
                    "client": kinesis,
                    "stream_type": "kinesis",
                    "stream_name": stream_name,
                    "config": connection_config
                }, {}
            
            elif operation == "schema":
                client = connection_config.get("client")
                stream_name = connection_config.get("stream_name")
                
                if not client:
                    return False, "No Kinesis client available", {}
                
                # Get shard iterator and sample records
                try:
                    response = client.describe_stream(StreamName=stream_name)
                    shards = response['StreamDescription']['Shards']
                    
                    if not shards:
                        return True, [], {"sample_count": 0}
                    
                    # Use first shard for sampling
                    shard_id = shards[0]['ShardId']
                    
                    shard_iterator_response = client.get_shard_iterator(
                        StreamName=stream_name,
                        ShardId=shard_id,
                        ShardIteratorType='TRIM_HORIZON'
                    )
                    
                    shard_iterator = shard_iterator_response['ShardIterator']
                    
                    # Get sample records
                    records_response = client.get_records(
                        ShardIterator=shard_iterator,
                        Limit=10
                    )
                    
                    records = records_response['Records']
                    
                    if records:
                        sample_data = []
                        for record in records:
                            data = json.loads(record['Data'].decode('utf-8'))
                            sample_data.append(data)
                        
                        schema = self._infer_message_schema(sample_data)
                        return True, [{
                            "schema_name": "kinesis_stream",
                            "table_name": stream_name,
                            "columns": schema
                        }], {"sample_count": len(sample_data)}
                    else:
                        return True, [], {"sample_count": 0}
                
                except Exception as e:
                    return False, f"Schema discovery failed: {str(e)}", {}
            
            elif operation == "read":
                client = connection_config.get("client")
                stream_name = connection_config.get("stream_name")
                limit = connection_config.get("limit", 100)
                
                if not client:
                    return False, "No Kinesis client available", {}
                
                try:
                    # Get all shards
                    response = client.describe_stream(StreamName=stream_name)
                    shards = response['StreamDescription']['Shards']
                    
                    all_records = []
                    
                    for shard in shards:
                        if len(all_records) >= limit:
                            break
                        
                        shard_id = shard['ShardId']
                        
                        shard_iterator_response = client.get_shard_iterator(
                            StreamName=stream_name,
                            ShardId=shard_id,
                            ShardIteratorType='TRIM_HORIZON'
                        )
                        
                        shard_iterator = shard_iterator_response['ShardIterator']
                        
                        while shard_iterator and len(all_records) < limit:
                            records_response = client.get_records(
                                ShardIterator=shard_iterator,
                                Limit=min(limit - len(all_records), 100)
                            )
                            
                            records = records_response['Records']
                            
                            for record in records:
                                try:
                                    data = json.loads(record['Data'].decode('utf-8'))
                                    record_data = {
                                        "sequence_number": record['SequenceNumber'],
                                        "partition_key": record['PartitionKey'],
                                        "approximate_arrival_timestamp": record.get('ApproximateArrivalTimestamp'),
                                        "data": data
                                    }
                                    all_records.append(record_data)
                                except json.JSONDecodeError:
                                    # Skip invalid JSON records
                                    continue
                            
                            shard_iterator = records_response.get('NextShardIterator')
                            
                            if not records:
                                break
                    
                    return True, all_records, {"records_read": len(all_records)}
                
                except Exception as e:
                    return False, f"Data reading failed: {str(e)}", {}
            
        except Exception as e:
            return False, str(e), {}
    
    async def _handle_pubsub(self, operation: str, connection_config: Dict[str, Any], 
                           auth_config: Dict[str, Any]) -> Tuple[bool, Any, Dict[str, Any]]:
        """Handle Google Cloud Pub/Sub operations."""
        try:
            # Import Google Cloud libraries (optional dependency)
            try:
                from google.cloud import pubsub_v1
                from google.auth.exceptions import DefaultCredentialsError
            except ImportError:
                return False, "google-cloud-pubsub library not installed", {}
            
            project_id = connection_config.get("project_id")
            subscription_name = connection_config.get("subscription_name")
            topic_name = connection_config.get("topic_name")
            
            if operation == "test":
                try:
                    # Test connection by creating subscriber client
                    subscriber = pubsub_v1.SubscriberClient()
                    
                    subscription_path = subscriber.subscription_path(project_id, subscription_name)
                    
                    # Try to get subscription info
                    subscription = subscriber.get_subscription(request={"subscription": subscription_path})
                    
                    return True, None, {
                        "subscription_name": subscription.name,
                        "topic": subscription.topic,
                        "ack_deadline_seconds": subscription.ack_deadline_seconds
                    }
                    
                except Exception as e:
                    return False, f"Pub/Sub connection failed: {str(e)}", {}
            
            elif operation == "connect":
                subscriber = pubsub_v1.SubscriberClient()
                
                return True, {
                    "subscriber": subscriber,
                    "stream_type": "pubsub",
                    "project_id": project_id,
                    "subscription_name": subscription_name,
                    "config": connection_config
                }, {}
            
            elif operation == "schema":
                subscriber = connection_config.get("subscriber")
                project_id = connection_config.get("project_id")
                subscription_name = connection_config.get("subscription_name")
                
                if not subscriber:
                    return False, "No Pub/Sub subscriber available", {}
                
                subscription_path = subscriber.subscription_path(project_id, subscription_name)
                
                # Pull sample messages for schema inference
                try:
                    response = subscriber.pull(
                        request={
                            "subscription": subscription_path,
                            "max_messages": 10,
                        },
                        timeout=10.0
                    )
                    
                    sample_data = []
                    ack_ids = []
                    
                    for received_message in response.received_messages:
                        try:
                            data = json.loads(received_message.message.data.decode('utf-8'))
                            sample_data.append(data)
                            ack_ids.append(received_message.ack_id)
                        except json.JSONDecodeError:
                            ack_ids.append(received_message.ack_id)
                    
                    # Acknowledge messages
                    if ack_ids:
                        subscriber.acknowledge(
                            request={
                                "subscription": subscription_path,
                                "ack_ids": ack_ids,
                            }
                        )
                    
                    if sample_data:
                        schema = self._infer_message_schema(sample_data)
                        return True, [{
                            "schema_name": "pubsub_stream",
                            "table_name": subscription_name,
                            "columns": schema
                        }], {"sample_count": len(sample_data)}
                    else:
                        return True, [], {"sample_count": 0}
                
                except Exception as e:
                    return False, f"Schema discovery failed: {str(e)}", {}
            
            elif operation == "read":
                subscriber = connection_config.get("subscriber")
                project_id = connection_config.get("project_id")
                subscription_name = connection_config.get("subscription_name")
                limit = connection_config.get("limit", 100)
                timeout = connection_config.get("timeout", 30)
                
                if not subscriber:
                    return False, "No Pub/Sub subscriber available", {}
                
                subscription_path = subscriber.subscription_path(project_id, subscription_name)
                
                try:
                    messages = []
                    
                    while len(messages) < limit:
                        batch_size = min(limit - len(messages), 100)
                        
                        response = subscriber.pull(
                            request={
                                "subscription": subscription_path,
                                "max_messages": batch_size,
                            },
                            timeout=timeout
                        )
                        
                        if not response.received_messages:
                            break
                        
                        ack_ids = []
                        
                        for received_message in response.received_messages:
                            try:
                                data = json.loads(received_message.message.data.decode('utf-8'))
                                message_data = {
                                    "message_id": received_message.message.message_id,
                                    "publish_time": received_message.message.publish_time,
                                    "attributes": dict(received_message.message.attributes),
                                    "data": data
                                }
                                messages.append(message_data)
                                ack_ids.append(received_message.ack_id)
                            except json.JSONDecodeError:
                                ack_ids.append(received_message.ack_id)
                        
                        # Acknowledge messages
                        if ack_ids:
                            subscriber.acknowledge(
                                request={
                                    "subscription": subscription_path,
                                    "ack_ids": ack_ids,
                                }
                            )
                    
                    return True, messages, {"messages_read": len(messages)}
                
                except Exception as e:
                    return False, f"Data reading failed: {str(e)}", {}
            
        except Exception as e:
            return False, str(e), {}
    
    def _infer_message_schema(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer schema from sample messages."""
        if not messages:
            return []
        
        # Collect all unique keys from all messages
        all_keys = set()
        for message in messages:
            if isinstance(message, dict):
                all_keys.update(message.keys())
        
        columns = []
        for key in sorted(all_keys):
            # Infer type from first non-null value
            inferred_type = "string"
            nullable = False
            
            for message in messages:
                if isinstance(message, dict) and key in message:
                    value = message[key]
                    if value is not None:
                        inferred_type = self._infer_value_type(value)
                        break
                else:
                    nullable = True
            
            columns.append({
                "name": key,
                "type": inferred_type,
                "nullable": nullable,
                "primary_key": key in ["id", "message_id", "key"]
            })
        
        return columns
    
    def _infer_value_type(self, value: Any) -> str:
        """Infer type from a value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"
    
    def validate_config(self, connection_config: Dict[str, Any]) -> List[str]:
        """Validate streaming connection configuration."""
        errors = []
        
        stream_type = connection_config.get("stream_type")
        if not stream_type:
            errors.append("Missing required field: stream_type")
        elif stream_type not in self.supported_streams:
            errors.append(f"Unsupported stream type: {stream_type}")
        
        if stream_type == "kafka":
            if not connection_config.get("topic"):
                errors.append("Missing required field for Kafka: topic")
        
        elif stream_type == "kinesis":
            if not connection_config.get("stream_name"):
                errors.append("Missing required field for Kinesis: stream_name")
            if not connection_config.get("region"):
                errors.append("Missing required field for Kinesis: region")
        
        elif stream_type == "pubsub":
            if not connection_config.get("project_id"):
                errors.append("Missing required field for Pub/Sub: project_id")
            if not connection_config.get("subscription_name"):
                errors.append("Missing required field for Pub/Sub: subscription_name")
        
        return errors