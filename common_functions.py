import cv2
import numpy as np

def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state

def generate_state_frame_stack_from_queue(deque):
    frame_stack = np.array(deque)
    # Move stack dimension to the channel dimension (stack, x, y) -> (x, y, stack)
    return np.transpose(frame_stack, (1, 2, 0))


def write_updates(state):
    from datetime import datetime
    with open(f"./resources/states.txt", "a") as file:
        file.write(f"{datetime.strftime(datetime.now(), '%Y%m%d%H%M')}:\n {state}")
