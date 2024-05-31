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
            'blades_openfast': {'OpenFAST_RUL_blade1': 20.0, 'OpenFAST_RUL_blade2': 20.0, 'OpenFAST_RUL_blade3': 20.0},
            'tower_openfast': {'OpenFAST_RUL_Tower': 20.0}
        }
        self.latest_pred_data = {
            'Pred_B': [],
            't_pred': [],
            'present_state_web': [],
            'current_time': "",
            'Pred_Delta_B': [],
            'Pred_B_Buffered': [],
            'RotSpeed': [],
            'WE_Vw': [],
            'VS_GenPwr': []
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
        # Emit all RUL data and prediction data upon client connection
        self.emit_all_rul_updates()
        self.emit_pred_updates()

    def emit_all_rul_updates(self):
        self.emit_rul_update(self.latest_rul_data['blades_openfast'], 'blades_openfast')
        self.emit_rul_update(self.latest_rul_data['tower_openfast'], 'tower_openfast')

    def emit_pred_updates(self):
        self.emit_pred_update(self.latest_pred_data['Pred_B'], 'Pred_B')
        self.emit_pred_update(self.latest_pred_data['t_pred'], 't_pred')
        self.emit_pred_update(self.latest_pred_data['present_state_web'], 'present_state_web')
        self.emit_pred_update(self.latest_pred_data['current_time'], 'current_time')
        self.emit_pred_update(self.latest_pred_data['Pred_Delta_B'], 'Pred_Delta_B')
        self.emit_pred_update(self.latest_pred_data['Pred_B_Buffered'], 'Pred_B_Buffered')
        self.emit_pred_update(self.latest_pred_data['RotSpeed'], 'RotSpeed')
        self.emit_pred_update(self.latest_pred_data['WE_Vw'], 'WE_Vw')
        self.emit_pred_update(self.latest_pred_data['VS_GenPwr'], 'VS_GenPwr')

    def start_zmq_listener(self):
        thread = Thread(target=self.listen_for_rul_updates)
        thread.daemon = True
        thread.start()

    def listen_for_rul_updates(self):
        context = zmq.Context()
        zmq_socket = context.socket(zmq.SUB)
        zmq_socket.connect("tcp://localhost:5556")
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '')

        print("Listening for RUL updates...")  # Debug print
        while True:
            message = zmq_socket.recv_json()
            print(f"Received ZMQ message: {message}")  # Debug print
            self.process_rul_updates(message)

    def process_rul_updates(self, message):
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
        if 'Pred_B' in message:
            self.update_pred_data('Pred_B', message['Pred_B'])
            #print(f"Received prediction data for Pred_B: {message['Pred_B']}")
        if 't_pred' in message:
            self.update_pred_data('t_pred', message['t_pred'])
            #print(f"Received prediction data for t_pred: {message['t_pred']}")
        if 'present_state_web' in message:
            self.update_pred_data('present_state_web', message['present_state_web'])
            #print(f"Received prediction data for present_state_web: {message['present_state_web']}")
        if 'current_time' in message:
            self.update_pred_data('current_time', message['current_time'])
            #print(f"Received current time data: {message['current_time']}")
        if 'Pred_Delta_B' in message:
            self.update_pred_data('Pred_Delta_B', message['Pred_Delta_B'])
            #print(f"Received Pred_Delta_B data: {message['Pred_Delta_B']}")
        if 'Pred_B_Buffered' in message:
            self.update_pred_data('Pred_B_Buffered', message['Pred_B_Buffered'])
            #print(f"Received Pred_Delta_B data: {message['Pred_Delta_B']}")
        if 'RotSpeed' in message:
            self.update_pred_data('RotSpeed', message['RotSpeed'])
            #print(f"Received Pred_Delta_B data: {message['Pred_Delta_B']}")
        if 'WE_Vw' in message:
            self.update_pred_data('WE_Vw', message['WE_Vw'])
            #print(f"Received Pred_Delta_B data: {message['Pred_Delta_B']}")
        if 'VS_GenPwr' in message:
            self.update_pred_data('VS_GenPwr', message['VS_GenPwr'])
            #print(f"Received Pred_Delta_B data: {message['Pred_Delta_B']}")

    def update_rul_data(self, category, new_data):
        for key, value in new_data.items():
            self.latest_rul_data[category][key] = value
        print(f"Received and updated RUL data for {category}: {self.latest_rul_data[category]}")
        self.emit_rul_update(self.latest_rul_data[category], category)

    def update_pred_data(self, category, new_data):
        self.latest_pred_data[category] = new_data
        print(f"Received and updated prediction data for {category}: {self.latest_pred_data[category]}")
        self.emit_pred_update(self.latest_pred_data[category], category)

    def emit_rul_update(self, rul_values, data_type):
        print(f"Emitting update_rul of type {data_type}: {rul_values}")
        self.socketio.emit('update_rul', {'type': data_type, 'data': rul_values})

    def emit_pred_update(self, pred_values, data_type):
        # print(f"Emitting update_pred of type {data_type}: {pred_values}")
        self.socketio.emit('update_pred', {'type': data_type, 'data': pred_values})

    def run(self, host='127.0.0.1', port=5005):
        self.socketio.run(self.app, host=host, port=port, debug=True)

if __name__ == '__main__':
    server = RealTimeServer_class()
    server.run()
