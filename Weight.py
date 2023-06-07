import numpy as np
import torch
from utils import *
s_data, s_y, s_n, t_data, t_y, t_n, s_adj, t_adj, t_name = load_data()

def convert_to_onehot(sca_label, class_num=t_n):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=t_n):
        # print(s_label.shape)
        batch_size = s_label.shape[0]
        tbatch_size = t_label.shape[0]

        s_sca_label = np.squeeze(s_label,axis=1)
        s_vec_label = convert_to_onehot(s_sca_label)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)
        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        # print(s_sca_label,t_sca_label)
        # print(s_vec_label,t_vec_label)
        # t_size = t_label.shape[0]

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((tbatch_size, tbatch_size))
        weight_st = np.zeros((batch_size, tbatch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(tbatch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                # print(s_tvec, t_tvec)
                st = s_tvec * t_tvec.T
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')