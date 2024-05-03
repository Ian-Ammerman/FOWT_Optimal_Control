from collections import deque
import numpy as np

# Global variables
Pred_B_buffer = deque()
last_used_Pred_B = None  # Initially set to zero
last_used_t_pred = None  # Initially set to None
first_delta_received = False
printed_first_Pred_B = False
last_whole_second = None  # Track the last whole second for countdown

def buffer(Pred_B, t_pred, current_time, measurements, buffer_duration):
    global Pred_B_buffer, last_used_Pred_B, last_used_t_pred, first_delta_received, printed_first_Pred_B, last_whole_second

    # Buffering Pred_B with its predicted time and the time it was predicted
    if Pred_B is not None:
        Pred_B_buffer.append((Pred_B, current_time + buffer_duration - 1, t_pred))
        if not first_delta_received and not printed_first_Pred_B:
            printed_first_Pred_B = True # Make sure only first Pred_B is printed
            first_delta_received = True  # Set the flag on receiving the first Pred_B
            last_whole_second = int(current_time)  # Initialize countdown start time
            print(f"First pitch angle offset received and buffered: {Pred_B - measurements["BlPitchCMeas"]} radians at time {current_time}")

    # Release buffer based on current time and buffer_duration
    while Pred_B_buffer and Pred_B_buffer[0][1] <= current_time:
        last_used_Pred_B, _, last_used_t_pred = Pred_B_buffer.popleft()
        first_delta_received = False  # Reset the flag after the first Pred_B is used

    # Use the last released Pred_B as the control pitch command
    if last_used_Pred_B is not None:
        Pred_Delta_B = last_used_Pred_B - measurements["BlPitchCMeas"]
    else:
        Pred_Delta_B = 0.0

    # Countdown for the first Pred_B in the buffer
    if first_delta_received:
        current_whole_second = int(current_time)
        if current_whole_second != last_whole_second:
            last_whole_second = current_whole_second
            if Pred_B_buffer:
                time_to_use = int(Pred_B_buffer[0][1] - current_time)
                print(f"Countdown until first Pred_B prediction is used: {time_to_use} s")

    # Print the current time and prediction time when a Pred_B is used
    if last_used_t_pred is not None and current_time % 1 == 0:
        print(f"Current Time: {current_time}, Last Used Prediction Time: {last_used_t_pred:.0f}")
        print(f"Sending predicted blade pitch offset setpoint: {Pred_Delta_B:.3f} ({Pred_Delta_B*180/np.pi:.3f} deg)")
    elif last_used_t_pred is None and current_time % 5 == 0:
        print("Blade Pitch Offset Setpoint:", Pred_Delta_B)

    return Pred_Delta_B
