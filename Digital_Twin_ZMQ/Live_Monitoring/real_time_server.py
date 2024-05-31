<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Operation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <style>
        body, html {
            height: 100%;
            margin: 0;
            font-family: 'Helvetica Neue', sans-serif;
            background-color: #f5f7fa;
            color: #333;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
        }
        
        .container {
            text-align: center;
            width: 90%;
            max-width: 1200px;
            padding: 20px;
            box-sizing: border-box;
        }

        .title {
            font-size: 1.55vw;
            margin-top: 0.5px;
            margin-bottom: 20px;
            font-weight: bold;
            color: #16a085;
        }

        .subtitle {
            font-size: 1vw;
            margin-top: 10px;
            margin-bottom: 10px;
            font-weight: bold;
            color: #16a085;
        }
        
        .rul-container {
            display: flex;
            justify-content: space-evenly;
            flex-wrap: wrap;
            gap: 5px;
            padding: 5px;
            width: 100%;
            box-sizing: border-box;
        }

        .parameter {
            flex: 1 1 calc(25% - 10px);
            min-width: 250px;
            background-color: #ffffff;
            padding: 10px;
            border: 1px solid #d0d7de;
            border-radius: 10px;
            color: #333;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            text-align: center;
        }

        .parameter:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }

        .openfast-blade1-box { border-left: 5px solid #0073e6; }
        .openfast-blade2-box { border-left: 5px solid #8e44ad; }
        .openfast-blade3-box { border-left: 5px solid #e67e22; }
        .openfast-tower-box { border-left: 5px solid #95a5a6; }
        .predicted-state-box { border-left: 5px solid #16a085; }
        .prediction-time-box { border-left: 5px solid #16a085; }
        .current-state-box { border-left: 5px solid #16a085; }
        .current-time-box { border-left: 5px solid #16a085; }
        .pred-delta-box { border-left: 5px solid #16a085; }
        
        .chart-container {
            width: 100%;
            max-width: 100%;
            height: 40vh;
            position: relative;
            margin: 0 auto 20px;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-sizing: border-box;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        #rulChart {
            width: 100%;
            height: 100%;
        }
        
        @media (max-width: 768px) {
            .parameter { 
                flex: 1 1 calc(50% - 10px);
            }
        }
        
        @media (max-width: 480px) {
            .parameter { 
                flex: 1 1 100%;
            }
        }
        
        .bottom-offset {
            width: 80%;
            max-width: 500px;
            background-color: #16a085;
            color: white;
            padding: 10px;
            margin: 20px auto;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .connection-status {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 2px;
            color: white;
            background-color: gray;
            text-align: center;
            border-radius: 0 0 0 0;
        }


        .operational-data {
            background-color: #eef7ee;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin-top: 20px;
            text-align: right;
            color: #4CAF50;
            font-size: 1.2em;
            position: absolute;
            top: 100px;
            right: 10px; /* Move more to the right */
            width: 200px;
        }
        .operational-data h3 {
            margin-top: 0;
            margin-bottom: 10px;
        }
        .data-row {
            margin-bottom: 10px;
        }
        
    </style>
</head>
<body>
    <div class="container">
        <div class="title">VolturnUS-S 15-MW Digital Twin: Live Monitoring</div>
        <div class="subtitle">Fatigue Model: Remaining Useful Life Estimation</div>
        <div class="chart-container">
            <canvas id="rulChart"></canvas>
        </div>
        <div class="rul-container">
            <div id="rul_openfast_blade1" class="parameter openfast-blade1-box">Blade Root 1 RUL: <span id="openfast_rul1_value">--</span></div>
            <div id="rul_openfast_blade2" class="parameter openfast-blade2-box">Blade Root 2 RUL: <span id="openfast_rul2_value">--</span></div>
            <div id="rul_openfast_blade3" class="parameter openfast-blade3-box">Blade Root 3 RUL: <span id="openfast_rul3_value">--</span></div>
            <div id="rul_tower_openfast" class="parameter openfast-tower-box">Tower Base RUL: <span id="tower_openfast_value">--</span></div>
        </div>
        <div class="subtitle">Prediction Model: Coll. Blade Pitch Angle Prediction</div>
        <div class="rul-container">
            <div id="current_state" class="parameter current-state-box">Current Coll. BlPitch: <span id="present_state_value">--</span> deg</div>
            <div id="predicted_state" class="parameter predicted-state-box">Predicted Coll. BlPitch: <span id="pred_b_value">--</span> deg</div> 
        </div>
        <div class="rul-container">
            <div id="Pred_B_Buffered" class="parameter pred-delta-box">Buffered Prediction: <span id="Pred_B_Buffered_value">--</span> deg</div>
            <div id="pred_delta_b" class="parameter pred-delta-box">Buffered Prediction Offset: <span id="pred_delta_b_value">--</span> deg</div>
        </div>
        <div class="rul-container">
            <div id="current_time" class="parameter current-time-box">Current Time: <span id="current_time_value">--</span> s</div>
            <div id="prediction_time" class="parameter prediction-time-box">Predicted Time: <span id="t_pred_value">--</span> s</div>
        </div>
        <div class="operational-data">
            <h4>Live Operation Data:</h4>
            <div id="WE_Vw" class="data-row"><span id="WE_Vw_value">--</span></div>
            <div id="RotSpeed" class="data-row"><span id="RotSpeed_value">--</span></div>
            <div id="VS_GenPwr" class="data-row"><span id="VS_GenPwr_value">--</span></div>
        </div>
        
        <div id="connection_status" class="connection-status">Connected</div>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var socket = io.connect('http://localhost:5005', {
                    reconnection: true,
                    reconnectionDelay: 1000,
                    reconnectionDelayMax: 5000,
                    reconnectionAttempts: Infinity
                });

                function updateConnectionStatus(status) {
                    var connectionStatusElement = document.getElementById('connection_status');
                    if (connectionStatusElement) {
                        connectionStatusElement.textContent = status;
                        connectionStatusElement.style.backgroundColor = status === 'Connected' ? 'green' : 'red';
                    }
                }
                
                function updateRULValues(values) {
                    var rulValueIds = ['openfast_rul1_value', 'openfast_rul2_value', 'openfast_rul3_value', 'tower_openfast_value'];
                    rulValueIds.forEach(function(valueId, index) {
                        var element = document.getElementById(valueId);
                        if (element) {
                            var rul = values[index];
                            element.textContent = rul !== null ? parseFloat(rul).toFixed(3) + ' years' : 'N/A';
                        }
                    });
                }

                function updatePredictionValues(type, values) {
                let elementId;
                let formattedValue;

                switch (type) {
                    case 'Pred_B':
                        elementId = 'pred_b_value';
                        formattedValue = parseFloat(values).toFixed(2);
                        break;
                    case 't_pred':
                        elementId = 't_pred_value';
                        formattedValue = Math.round(values);
                        break;
                    case 'present_state_web':
                        elementId = 'present_state_value';
                        formattedValue = parseFloat(values).toFixed(2);
                        break;
                    case 'current_time':
                        elementId = 'current_time_value';
                        formattedValue = parseFloat(values).toFixed(1);
                        break;
                    case 'Pred_Delta_B':
                        elementId = 'pred_delta_b_value';
                        formattedValue = parseFloat(values).toFixed(2);
                        break;
                    case 'Pred_B_Buffered':
                        elementId = 'Pred_B_Buffered_value';
                        formattedValue = parseFloat(values).toFixed(2);
                        break;
                    case 'RotSpeed':
                        elementId = 'RotSpeed';
                        formattedValue = "Rotor speed: " + parseFloat(values).toFixed(2) + "rpm";
                        break;
                    case 'WE_Vw':
                        elementId = 'WE_Vw';
                        formattedValue = "Wind speed: " + parseFloat(values).toFixed(2) + "m/s";
                        break;
                    case 'VS_GenPwr':
                        elementId = 'VS_GenPwr';
                        formattedValue = "Gen. power: " + (parseFloat(values) / 1000000).toFixed(2) + 'MW';
                        break;
                    default:
                        console.error('Unknown type for prediction update:', type);
                        return;
                }

                    var element = document.getElementById(elementId);
                    if (element) {
                        element.textContent = formattedValue !== undefined ? formattedValue : 'N/A';
                    }
                }
            
                socket.on('connect', function() {
                    updateConnectionStatus('Connected');
                    socket.emit('request_latest_rul');
                });

                socket.on('update_rul', function(update) {
                    if (update.data && update.data !== 'N/A') {
                        if (update.type === 'blades_openfast') {
                            lastKnownRUL.blade1 = update.data['OpenFAST_RUL_blade1'];
                            lastKnownRUL.blade2 = update.data['OpenFAST_RUL_blade2'];
                            lastKnownRUL.blade3 = update.data['OpenFAST_RUL_blade3'];
                            updateElementText('openfast_rul1_value', lastKnownRUL.blade1);
                            updateElementText('openfast_rul2_value', lastKnownRUL.blade2);
                            updateElementText('openfast_rul3_value', lastKnownRUL.blade3);
                            updateChart(update.data, update.type);
                        } else if (update.type === 'tower_openfast') {
                            lastKnownRUL.tower = update.data['OpenFAST_RUL_Tower'];
                            updateElementText('tower_openfast_value', lastKnownRUL.tower);
                            updateChart(update.data, update.type);
                        }                                             
                    }
                });

                socket.on('update_pred', function(update) {
                    if (update.data && update.data !== 'N/A') {
                        updatePredictionValues(update.type, update.data);
                    }
                });

                socket.on('connect_error', function(error) {
                    updateConnectionStatus('Disconnected');
                });

                socket.on('connect_timeout', function(timeout) {
                    updateConnectionStatus('Disconnected');
                });

                socket.on('error', function(error) {
                    updateConnectionStatus('Disconnected');
                });

                socket.on('reconnect_attempt', function() {
                    updateConnectionStatus('Reconnecting...');
                });

                socket.on('disconnect', function() {
                    updateConnectionStatus('Disconnected');
                    updateElementText('openfast_rul1_value', lastKnownRUL.blade1);
                    updateElementText('openfast_rul2_value', lastKnownRUL.blade2);
                    updateElementText('openfast_rul3_value', lastKnownRUL.blade3);
                    updateElementText('tower_openfast_value', lastKnownRUL.tower);
                });
                
                var lastKnownRUL = {
                    blade1: '20',
                    blade2: '20',
                    blade3: '20',
                    tower: '20'
                };

                function updateElementText(elementId, value) {
                    var element = document.getElementById(elementId);
                    if (element) {
                        if (value !== undefined && value !== 'N/A') {
                            element.textContent = parseFloat(value).toFixed(3) + ' years';
                        } else {
                            element.textContent = 'N/A';
                        }
                    }
                }

                function updateChart(data, type) {
                    if (!rulChart || !rulChart.data || !rulChart.data.datasets) {
                        return;
                    }
                
                    const maxDataPoints = 60;
                    const newLabel = new Date().toLocaleTimeString();
                    const newTime = new Date();

                    rulChart.data.labels.push(newLabel);
                    
                    if (rulChart.data.labels.length > maxDataPoints) {
                        rulChart.data.labels.shift();
                    }
                    
                    switch (type) {
                        case 'blades_openfast':
                            rulChart.data.datasets[0].data.push({ x: newTime, y: data['OpenFAST_RUL_blade1'] });
                            rulChart.data.datasets[1].data.push({ x: newTime, y: data['OpenFAST_RUL_blade2'] });
                            rulChart.data.datasets[2].data.push({ x: newTime, y: data['OpenFAST_RUL_blade3'] });
                            break;
                        case 'tower_openfast':
                            rulChart.data.datasets[3].data.push({ x: newTime, y: data['OpenFAST_RUL_Tower'] });
                            break;
                        default:
                            return;
                    }
                
                    rulChart.data.datasets.forEach((dataset) => {
                        while (dataset.data.length > maxDataPoints) {
                            dataset.data.shift();
                        }
                    });

                    rulChart.update();
                }

                var ctx = document.getElementById('rulChart').getContext('2d');
                var rulChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [
                            {
                                label: 'Blade 1',
                                data: [],
                                borderColor: '#0073e6',
                                tension: 0.1
                            },
                            {
                                label: 'Blade 2',
                                data: [],
                                borderColor: '#8e44ad',
                                tension: 0.1
                            },
                            {
                                label: 'Blade 3',
                                data: [],
                                borderColor: '#e67e22',
                                tension: 0.1
                            },
                            {
                                label: 'Tower',
                                data: [],
                                borderColor: '#95a5a6',
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: false,
                                max: 21,
                                min: 15,
                                ticks: {
                                    stepSize: 1
                                }
                            }
                        }
                    }
                });
            });
        </script>
    </div>
</body>
</html>
