import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from ZMQ_Prediction_ROSCO.DOLPHINN.examples.MLSTM import run_prediction

run_prediction()