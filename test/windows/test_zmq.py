import zmq

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5555")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

try:
    while True:
        message = subscriber.recv_json()
        print(f"Received message: {message}")
except KeyboardInterrupt:
    pass
finally:
    subscriber.close()
    context.term()
