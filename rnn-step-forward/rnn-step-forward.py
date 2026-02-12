import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    input_part = np.dot(x_t, Wx)
    hidden_part = np.dot(h_prev, Wh)
    pre_activation = input_part + hidden_part + b
    h_t = np.tanh(pre_activation)
    return h_t