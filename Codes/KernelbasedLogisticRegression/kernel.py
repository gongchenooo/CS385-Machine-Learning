import numpy as np

def rbf(x_i_list, x_j_list, sigma):
    diff_norm = np.linalg.norm(x_i_list[:, None] - \
                               x_j_list[None, :], axis=2) ** 2
    return np.exp(-0.5 * diff_norm / sigma ** 2)

def poly(x_i_list, x_j_list, degree):
    product = np.matmul(x_i_list[:, None, None], x_j_list[None, :, :, None]).reshape([x_i_list.shape[0], x_j_list.shape[0]])
    return product ** degree

def cos(x_i_list, x_j_list):
    product1 = np.matmul(x_i_list[:, None, None], x_j_list[None, :, :, None]).reshape([x_i_list.shape[0], x_j_list.shape[0]])
    product2 = (np.linalg.norm(x_i_list, axis=1, keepdims=True)[:, None] *
               np.linalg.norm(x_j_list, axis=1, keepdims=True)[None, :]).reshape([x_i_list.shape[0], x_j_list.shape[0]])
    return product1 / product2

def sigmoid(x_i_list, x_j_list, alpha, c):
    tobe_tanh = alpha * np.matmul(
        x_i_list[:, None, None], x_j_list[None, :, :, None]) + c
    exp_tobe_tanh = np.exp(-2 * tobe_tanh.reshape((x_i_list.shape[0], x_j_list.shape[0])))