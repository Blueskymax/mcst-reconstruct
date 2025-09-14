import torch
import torch.nn as nn

from config import Args


class Tracker(object):
    def __init__(self):
        self.x_sigma = [] # 卡尔曼滤波跟踪参数: 方差
        self.x_predict_history = [] # 卡尔曼滤波跟踪参数: 历史先验值
        self.x_update_history = [] # 卡尔曼滤波跟踪参数: 历史后验值

    def track_init(self, obj, predictor_time_series_len, predictor_in_features, batch_size, t, device):
        """
        初始化跟踪器参数 方差 历史先验值 历史后验值
        :param obj: 观察值
        :param predictor_time_series_len: 预测器输入帧数
        :param predictor_in_features: 预测器输入维度
        :param batch_size: 批量大小
        :param t: 采样周期
        :param device: cpu or gpu
        """
        # 初始化方差
        for i in range(predictor_time_series_len):
            self.x_sigma.append((torch.zeros(batch_size, 1, predictor_in_features, requires_grad=True)).to(device))
        
        velocity1 = ((obj[:, 4, 0] - obj[:, 0, 0]) / (4 * t)).to(device) # 0-4帧x轴平均速度
        velocity2 = ((obj[:, 4, 1] - obj[:, 0, 1]) / (4 * t)).to(device) # 0-4帧y轴平均速度
        velocity3 = ((obj[:, 4, 2] - obj[:, 0, 2]) / (4 * t)).to(device) # 0-4帧z轴平均速度

        for index in range(predictor_time_series_len):
            # tmp (batch_size, 1, 6) 每一帧的 (input_update,v_x,y,v_y,z,v_z)
            tmp = torch.cat(
                (obj[:, index, 0].unsqueeze(dim=1).unsqueeze(dim=2), # input_update
                 velocity1.unsqueeze(dim=1).unsqueeze(dim=2), # v_x
                 obj[:, index, 1].unsqueeze(dim=1).unsqueeze(dim=2), # y
                 velocity2.unsqueeze(dim=1).unsqueeze(dim=2), # v_y
                 obj[:, index, 2].unsqueeze(dim=1).unsqueeze(dim=2), # z
                 velocity3.unsqueeze(dim=1).unsqueeze(dim=2)), dim=2) # v_z

            self.x_predict_history.append(tmp.to(device))
            self.x_update_history.append(tmp.to(device))

        print()


