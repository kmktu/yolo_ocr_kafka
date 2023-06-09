from kafka import KafkaProducer
import time
from json import dumps

def main():
    producer = KafkaProducer(bootstrap_servers=["kafka:9092"], api_version=(0, 11, 5),
                             value_serializer=lambda x: dumps(x).encode('utf-8'))
    producer_value = {"hello": 1}
    print("first_producer_value", producer_value)

    while True:
        time.sleep(5)
        print("value : ", producer_value)
        producer_value["hello"] += 1
        try:
            response = producer.send(topic='topic1', value=producer_value).get()
            print(response)
        except Exception as e:
            print("e : ", e)

if __name__ == '__main__':
    main()