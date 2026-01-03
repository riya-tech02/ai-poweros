"""Create Kafka topics"""
from kafka.admin import KafkaAdminClient, NewTopic
import os

def create_topics():
    try:
        servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(',')
        admin = KafkaAdminClient(bootstrap_servers=servers)
        
        topics = [
            NewTopic(name="user-events", num_partitions=3, replication_factor=1),
            NewTopic(name="predictions", num_partitions=3, replication_factor=1),
            NewTopic(name="task-updates", num_partitions=3, replication_factor=1),
        ]
        
        admin.create_topics(new_topics=topics, validate_only=False)
        print("✓ Kafka topics created")
    except Exception as e:
        print(f"⚠ Kafka topics may already exist: {e}")

if __name__ == "__main__":
    create_topics()
