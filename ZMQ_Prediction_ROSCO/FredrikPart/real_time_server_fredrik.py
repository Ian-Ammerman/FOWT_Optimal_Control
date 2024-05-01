# real_time_server.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import zmq
from threading import Thread

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", 
                    logger=True, engineio_logger=True, 
                    ping_timeout=20, ping_interval=10)

class RealTimeServer_class:
    def __init__(self):
        self.app = app
        self.socketio = socketio
        self.latest_rul_data = {
        #    'blades_rosco': {'ROSCO_RUL_blade1': 0.0, 'ROSCO_RUL_blade2': 0.0, 'ROSCO_RUL_blade3': 0.0},
            'blades_openfast': {'OpenFAST_RUL_blade1': 20.0, 'OpenFAST_RUL_blade2': 20.0, 'OpenFAST_RUL_blade3': 20.0},
            'tower_openfast': {'OpenFAST_RUL_Tower': 20.0}
        }
        self.register_routes()
        self.start_zmq_listener()

    def register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('website.html')

        @self.socketio.on('connect')
        def on_connect():
            print('Client connected')
            # Emit the latest known data on new client connections
            self.emit_initial_data()

        @self.socketio.on('disconnect')
        def on_disconnect():
            print('Client disconnected')

        @self.socketio.on('request_latest_rul')
        def handle_request_latest_rul():
            print("Latest RUL data requested by client")
            self.emit_all_rul_updates()

    def emit_initial_data(self):
        # Emit all RUL data upon client connection
        self.emit_all_rul_updates()

    def emit_all_rul_updates(self):
        #self.emit_rul_update(self.latest_rul_data['blades_rosco'], 'blades_rosco')
        self.emit_rul_update(self.latest_rul_data['blades_openfast'], 'blades_openfast')
        self.emit_rul_update(self.latest_rul_data['tower_openfast'], 'tower_openfast')

    def start_zmq_listener(self):
        thread = Thread(target=self.listen_for_rul_updates)
        thread.daemon = True
        thread.start()

    def listen_for_rul_updates(self):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.SUB)
        zmq_socket.connect("tcp://localhost:5556")
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        print("Listening for RUL updates...")
        while True:
            message = zmq_socket.recv_json()
            self.process_rul_updates(message)

    def process_rul_updates(self, message):
        #if 'rul_values_blade_rosco' in message:
        #    self.update_rul_data('blades_rosco', message['rul_values_blade_rosco'])
        if 'rul_values_blade_openfast' in message:
            self.update_rul_data('blades_openfast', message['rul_values_blade_openfast'])
        if 'rul_values_tower_openfast' in message:
            # Ensure the key expected matches the key received
            tower_data = message['rul_values_tower_openfast']
            if 'OpenFAST_RUL_Tower' in tower_data:
                new_tower_value = tower_data['OpenFAST_RUL_Tower']
                self.latest_rul_data['tower_openfast']['OpenFAST_RUL_Tower'] = new_tower_value
                print(f"Received RUL updates for tower_openfast: {new_tower_value}")
                self.emit_rul_update({'OpenFAST_RUL_Tower': new_tower_value}, 'tower_openfast')
  
    def update_rul_data(self, category, new_data):
        # Apply updates cleanly without adding duplicate keys
        for key, value in new_data.items():
            # This line should update existing keys without adding new format keys
            self.latest_rul_data[category][key] = value
        print(f"Received and updated RUL data for {category}: {self.latest_rul_data[category]}")
        self.emit_rul_update(self.latest_rul_data[category], category)

    def emit_rul_update(self, rul_values, data_type):
        # Log the data being emitted to ensure it's structured correctly
        print(f"Emitting update_rul of type {data_type}: {rul_values}")
        # Emitting only the necessary and correctly formatted data
        self.socketio.emit('update_rul', {'type': data_type, 'data': rul_values})

    def run(self, host='127.0.0.1', port=5005):
        self.socketio.run(self.app, host=host, port=port, debug=True)

if __name__ == '__main__':
    server = RealTimeServer_class()
    server.run()
