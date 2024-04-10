# real_time_server.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import zmq
from threading import Thread

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, ping_timeout=10, ping_interval=10)

class RealTimeServer_class:
    def __init__(self):
        self.app = app
        self.socketio = socketio
        self.latest_rul_data = {'blade1': 'N/A', 'blade2': 'N/A', 'blade3': 'N/A'}
        self.register_routes()
        self.start_zmq_listener()

    def register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('connect')
        def on_connect():
            print('Client connected')
            self.emit_rul_update(self.get_latest_rul_data())  # Emit latest RUL data upon client connection

        @self.socketio.on('disconnect')
        def on_disconnect():
            print('Client disconnected')

        @socketio.on('request_latest_rul')
        def handle_request_latest_rul():
            #print("Latest RUL data requested by client")
            rul_data = self.get_latest_rul_data()  # Directly call the instance method
            self.emit_rul_update(rul_data)         # Emit the latest data to the client

    def start_zmq_listener(self):
        thread = Thread(target=self.listen_for_rul_updates)
        thread.daemon = True
        thread.start()

    def listen_for_rul_updates(self):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.SUB)
        zmq_socket.connect("tcp://localhost:5556")
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        while True:
            message = zmq_socket.recv_json()
            rul_values = message.get('rul_values')
            if rul_values:
                self.latest_rul_data = rul_values
                #print(f"Received RUL updates: {rul_values}")
                self.emit_rul_update(rul_values)

    def emit_rul_update(self, rul_values):
        #print(f"Preparing to emit RUL updates to clients: {rul_values}")
        self.socketio.emit('update_rul', rul_values)
        #print(f"Emitted RUL updates to clients: {rul_values}")

    def get_latest_rul_data(self):
        return self.latest_rul_data

    def run(self, host='127.0.0.1', port=5005):
        self.socketio.run(self.app, host=host, port=port, debug=True)

if __name__ == '__main__':
    server = RealTimeServer_class()
    server.run()
