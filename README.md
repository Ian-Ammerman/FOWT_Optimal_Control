Run ZMQ_Prediction_ROS/Driver.py to initiate OpenFAST simulation with ROSCO.

The DOLPHINN framework will receive measurements within ZMQ_Prediction_ROSCO/DOLPHINN/prediction/pitch_prediction.py, and return predicted measurements to Driver.py w/ ZMQ.

Driver.py will then send setpoints directly to the ROSCO controller as collective blade pitch angle setpoints using ZMQ.
