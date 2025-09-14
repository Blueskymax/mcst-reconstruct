import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import config
from models.AttentionMechanism import AdditiveAttention
from models.MLP import ResMLP
import matplotlib.pyplot as plt



class Updater(nn.Module):
    def __init__(self, in_features: int,hidden_features: int, out_features: int, dropout_rate, time_series_len, device):
        super(Updater, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.embed = nn.Linear(in_features, int(hidden_features))
        self.embed_detection = nn.Linear(int(in_features) * time_series_len, int(hidden_features * in_features))

        self.linear1 = nn.Linear(int(hidden_features), int(out_features))

        self.linear2 = nn.Linear(int(hidden_features), 1)
        self.linear3 = nn.Linear(int(hidden_features), 1)

        self.softPlus = nn.Softplus()

        self.mlp_w = ResMLP(self.in_features + self.out_features, int(hidden_features),
                            int(hidden_features * hidden_features), 8, dropout_rate)

        self.mlp_sigma = ResMLP(int(hidden_features * 1), int(hidden_features),
                                int(hidden_features * 1), 8, dropout_rate)

        self.mlp_detection = ResMLP(int(hidden_features * 1 ), int(hidden_features),
                                    int(hidden_features * 1), 4, dropout_rate)

        self.mlp_detection_sigma = ResMLP(int(hidden_features * 1), int(hidden_features),
                                          int(hidden_features * 1), 4, dropout_rate)

        self.layer_norm = nn.LayerNorm(int(hidden_features), eps=1e-6).to(device)

        self.linear4 = nn.Linear(int(hidden_features), int(hidden_features))
        self.linear5 = nn.Linear(int(hidden_features), out_features)
        self.leakyRelu = nn.LeakyReLU(0.5, inplace=True)
        self.rest_parameter()

    def rest_parameter(self):
        nn.init.kaiming_normal_(self.embed.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.embed_detection.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear4.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear5.weight, mode='fan_in', nonlinearity='tanh')

    def forward(self, input_update, x_predict_sigma_log, input_detection):
        """
        更新器
        :param input_update: 历史后验 + 当前先验
        :param x_predict_sigma_log: 先验方差
        :param input_detection: 历史量测 + 当前量测
        :return:
        """

        # 1. 先验值与当前量测值之间的差值 (batch_size, 1, 3)
        residuals = (input_update[:, -1, 0::2] - input_detection[:, -1, :]).unsqueeze(dim=1)
        # 嵌入层: (batch_size, 1, 3) -> (batch_size, 1, 128)
        residuals_embedding = self.leakyRelu(self.embed(residuals))

        # 2. 量测序列嵌入层：首先 reshape (batch_size, 5, 3) -> (batch_size, 15)
        # 然后输入到 embedding 层然后 reshape 为 (batch_size, 3, 128) 表示 x y z维度的高维度向量
        input_detection_reshape = input_detection.reshape(input_detection.shape[0], -1)
        detection_embedding = self.embed_detection(input_detection_reshape)
        detection_embedding = self.leakyRelu(detection_embedding).reshape([input_detection.shape[0], self.in_features, -1])

        # 3. 计算量测方差（以log形式表示）
        res = detection_embedding +self.mlp_detection_sigma(detection_embedding)
        detection_sigma_log = torch.log(self.softPlus(self.linear3(res))+1e-5).squeeze(dim=2).unsqueeze(dim=1)

        # 4. 融合检测方差(detection_sigma_log) 与 先验方差特征(x_predict_sigma_log)，生成用于残差特征空间变换的权重矩阵 W
        sigma_log = torch.cat([torch.sqrt(torch.exp(detection_sigma_log)), torch.sqrt(torch.exp(x_predict_sigma_log))], dim=-1)
        W = torch.tanh(self.mlp_w(sigma_log))
        W= W.reshape(-1, self.hidden_features, self.hidden_features)

        # 5. 矩阵相乘 得到后验值
        x_predict_hidden = torch.einsum("ijk,ikl -> ijl",[residuals_embedding, W])
        x_predict_hidden = self.linear4(x_predict_hidden)
        x_predict_hidden = self.leakyRelu(x_predict_hidden)
        x_predict_hidden = self.layer_norm(x_predict_hidden)
        x_predict = torch.tanh(self.linear5(x_predict_hidden)) +  input_update[:, -1, :].unsqueeze(dim=1)

        # 6. 后验方差
        x_predict_sigma_log = torch.log(self.softPlus(self.linear1(residuals_embedding + self.mlp_sigma(residuals_embedding))) + 1e-5)

        return x_predict, x_predict_sigma_log, detection_sigma_log


class DualStageAttention(nn.Module):
    def __init__(self, hidden_features):
        super(DualStageAttention, self).__init__()
        self.hidden_features = hidden_features # 128
        self.linear1 = nn.Linear(hidden_features, int(hidden_features * 2))
        self.norm1 = nn.LayerNorm(int(hidden_features * 2))  # 正则化 防止过拟合
        self.leakyRelu = nn.LeakyReLU(0.5, inplace=True)

        self.attention_time = AdditiveAttention(int(hidden_features), int(hidden_features), int(hidden_features * 2), dropout=0.1)
        self.attention_sample_points = AdditiveAttention(int(hidden_features * 2), int(hidden_features * 2),
                                                         int(hidden_features * 2), dropout= 0.1)

        self.rest_parameter()

    def rest_parameter(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.5)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, lstm_output, batch_size, sample_num):
        attention_time_output = self.attention_time(lstm_output, lstm_output, lstm_output[:, -1, :].unsqueeze(dim=1))

        attention_time_output = attention_time_output.permute(1, 0, 2).reshape([batch_size, sample_num, -1]).transpose(1,0)
        attention_time_output_linear = self.leakyRelu(self.norm1(self.linear1(attention_time_output.transpose(0, 1))))
        attention_sample_points_output = self.attention_sample_points(attention_time_output_linear,attention_time_output_linear,
                                                           attention_time_output_linear[:,0,:].unsqueeze(dim=1))

        return attention_sample_points_output

class ManeuverCompensationUnit(nn.Module):
    def __init__(self, fft_point):
        super(ManeuverCompensationUnit, self).__init__()
        self.fft_point = fft_point
        self.gate = ResMLP(self.fft_point, self.fft_point, self.fft_point, 2, 0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, innovation_error):
        innovationError_reshape = innovation_error.permute(0, 2, 1)
        innovationError_frequency_spectrum = torch.fft.fft(innovationError_reshape, n=self.fft_point, dim=-1)
        gate = self.sigmoid(self.gate(torch.abs(innovationError_frequency_spectrum)) - 0) * 1 # 缩放到[0,1]之间
        innovationError_frequency_spectrum_save = innovationError_frequency_spectrum * gate
        maneuver_compensation = torch.fft.ifft(innovationError_frequency_spectrum_save, n=self.fft_point, dim=-1)
        maneuver_compensation_downsample = torch.real(maneuver_compensation[:, :, :innovationError_reshape.shape[2]])
        maneuver_compensation_downsample = maneuver_compensation_downsample.permute(0, 2, 1)
        return maneuver_compensation_downsample


class MCU(nn.Module):
    def __init__(self, mcu_layer, mcu_hidden_dim):
        super(MCU, self).__init__()
        cell_list = nn.ModuleList([])
        for i in range(mcu_layer):
            cell_list.append(ManeuverCompensationUnit(mcu_hidden_dim))
        self.maneuverCompensationLayer = cell_list

    def forward(self, normalized_detection_mcu, normalized_update_history_mcu):
        # a. 残差 保留高频分量 高频分量意味着机动残差 这里的0::2表示取0 2 4 取出
        innovationError = (normalized_detection_mcu - normalized_update_history_mcu[:, :, 0::2])

        maneuver_compensation_input_list = []
        maneuver_compensation_output_list = []
        maneuver_compensation_input_list.append(innovationError)
        layer_index = 0
        for layer in self.maneuverCompensationLayer:
            if layer_index == 0:
                output = layer(maneuver_compensation_input_list[0])
                maneuver_compensation_output_list.append(output.unsqueeze(dim=0))
            else:
                #  后续层处理(前一层输入 - 前一层输出的残差)（类似ResNet的跳跃连接）
                output = layer(maneuver_compensation_input_list[layer_index - 1] - maneuver_compensation_output_list[layer_index - 1].squeeze(dim=0))
                maneuver_compensation_input_list.append(maneuver_compensation_input_list[layer_index - 1] - maneuver_compensation_output_list[layer_index - 1].squeeze(dim=0))
                maneuver_compensation_output_list.append(output.unsqueeze(dim=0))
            layer_index = layer_index + 1

        maneuver_compensation_output_list_cat = torch.cat(maneuver_compensation_output_list, dim=0)
        maneuver_compensation = torch.sum(maneuver_compensation_output_list_cat, dim=0, keepdim = True).squeeze(dim=0)

        return maneuver_compensation

class Predictor(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout_rate, num_layers, sample_num,mcu_layer, mcu_hidden_features,device):
        super(Predictor, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.sample_num = sample_num
        self.device = device

        self.x_embed = nn.Linear(int(in_features), int(hidden_features))

        self.maneuverWeight_embed = nn.Linear(int(in_features / 2), int(hidden_features))

        self.linear1 = nn.Linear(sample_num, 1)
        self.linear2 = nn.Linear(hidden_features * self.num_layers * 2, out_features)
        self.linear3 = nn.Linear(hidden_features * self.num_layers * 2, out_features)

        self.softPlus = nn.Softplus()

        self.biLstm = nn.LSTM(input_size=hidden_features,
                              hidden_size=hidden_features,
                              num_layers=self.num_layers,
                              bidirectional=True,
                              batch_first=True)

        self.mlp_x = ResMLP(int(hidden_features * 4), int(hidden_features),
                                out_features, 2, 0.1)

        self.mlp_sigma = ResMLP(int(hidden_features * self.num_layers * 2), int(hidden_features),
                                int(hidden_features * self.num_layers * 2), 2, 0.1)

        self.MCU = MCU(mcu_layer, mcu_hidden_features)
        self.dual_stage_attention = DualStageAttention(hidden_features * self.num_layers)
        self.linear4 = nn.Linear(hidden_features * 2, hidden_features)

        self.norm1 = nn.LayerNorm(int(hidden_features * 4))  # 正则化 防止过拟合
        self.norm2 = nn.LayerNorm(hidden_features * self.num_layers * 2)  # 正则化 防止过拟合

        self.softPlus = nn.Softplus()
        self.leakyRelu = nn.LeakyReLU(0.5, inplace=True)

        self.rest_parameter()

    def rest_parameter(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.5)
        nn.init.kaiming_normal_(self.x_embed.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.maneuverWeight_embed.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear4.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, update_sigma_history, normalized_update_history, normalized_detection_mcu,
                normalized_update_history_mcu, hidden_states=None):
        """
        预测器
        :param update_sigma_history: 历史后验方差
        :param normalized_update_history: 历史后验值
        :param normalized_detection_mcu: mcu模块 历史量测值
        :param normalized_update_history_mcu: mcu模块 历史后验值
        :param hidden_states: 
        :return: 
        """
        # batch_size: 批量大小 seq_len: 序列长度
        batch_size, seq_len, _ = normalized_update_history.size()

        # 1. 采样 (batch_size, 5, 6, 5): (batch_size, 5, 6) -> (batch_size, 5, 6, 5)
        normalized_update_history = self.unscented_transform(normalized_update_history, update_sigma_history)
        # 将采样的数据拼接批量维度： (batch_size*5, 5, 6): (batch_size, 5, 6, 5) -> (batch_size*5, 5, 6)
        normalized_update_history = normalized_update_history.permute(0, 3, 1, 2).reshape(-1,normalized_update_history.shape[1],normalized_update_history.shape[2])

        # 2. MCU补偿 (batch_size, 16, 3)
        maneuver_compensation = self.MCU(normalized_detection_mcu, normalized_update_history_mcu)

        # 3. BiLSTM 嵌入层
        # 3.1 后验值经过嵌入层 (batch_size*5, 5, 6) -> (batch_size*5, 5, 64)
        xPrevious_embedding = self.leakyRelu(self.x_embed(normalized_update_history))
        # 3.2 MCU补偿经过嵌入层 (batch_size, 16, 3) -> (batch_size, 16, 64)
        maneuver_weight_embed = self.leakyRelu(self.maneuverWeight_embed(maneuver_compensation))
        # 3.3 MCU补偿重复采样次数，取历史5帧 (batch_size, 16, 64) -> (batch_size*5, 5, 64)
        maneuver_weight_embed = (maneuver_weight_embed[:, -seq_len:, :].unsqueeze(dim=3)
                                 .repeat(1, 1, 1,self.sample_num)
                                 .permute(0, 3, 1, 2)
                                 .reshape(-1, seq_len, maneuver_weight_embed.shape[2]))
        # 3.4 历史后验值序列 和 MCU补偿序列 拼接形成一个(batch_size*5, 5, 128)张量
        # 然后经过一个线性层 (batch_size*5, 5, 128) -> (batch_size*5, 5, 64)
        x = self.linear4(torch.cat([xPrevious_embedding, maneuver_weight_embed], dim=2))
        # 3.5 经过激活函数
        x = self.leakyRelu(x)

        # 4. biLstm 层 输入: (batch_size*5, 5, 64) 输出: (batch_size*5, 5, 128)
        output, (h_predict, c_predict) = self.biLstm(x, hidden_states)

        # 5. 双阶段注意力机制 (batch_size*5, 5, 128) -> (batch_size, 1, 256)
        attention_output = self.dual_stage_attention(output, batch_size, self.sample_num)

        # 6. 输出层
        # 6.1 先验值 (batch_size, 1, 256) -> (batch_size, 1, 6)
        x = self.mlp_x(self.norm1(attention_output)).permute(0, 2, 1)
        x_predict = self.leakyRelu(x).permute(0, 2, 1)
        # 6.2 先验方差 (batch_size, 1, 256) -> (batch_size, 1, 6)
        x_predict_sigma = self.softPlus(self.leakyRelu(self.linear3(self.norm2(self.mlp_sigma(attention_output)))))
        x_predict_sigma_log = torch.log(x_predict_sigma)

        return x_predict, x_predict_sigma_log, (h_predict, c_predict)

    def init_hidden(self, input_size):
        return torch.zeros(self.num_layers * 2, input_size, self.hidden_features, requires_grad=True, device=self.device)

    def init_cell(self, input_size):
        return torch.zeros(self.num_layers * 2, input_size, self.hidden_features, requires_grad=True, device=self.device)

    def unscented_transform(self, normalized_update_history, update_sigma_history):
        device = normalized_update_history.device
        normalized_update_history = normalized_update_history.unsqueeze(dim=3).repeat(1, 1, 1, self.sample_num)
        update_sigma_history = update_sigma_history.unsqueeze(dim=3).repeat(1, 1, 1, self.sample_num)

        original_size = torch.Size([normalized_update_history.shape[0],normalized_update_history.shape[1],normalized_update_history.shape[2], 1])
        sample_size = torch.Size([normalized_update_history.shape[0], normalized_update_history.shape[1], normalized_update_history.shape[2], normalized_update_history.shape[3] - 1])
        # 通过公式 μ + σ×ε 生成最终采样
        try:
            normalized_update_history = (normalized_update_history + torch.sqrt(torch.exp(update_sigma_history)) * torch.cat([torch.zeros(original_size).to(device),
                                                                                                                          Normal(0, 1).sample(sample_size).to(device)], dim=3))
        except:
            print(normalized_update_history.shape)
            print(update_sigma_history.shape)
            raise Exception
        return normalized_update_history

# 引入残差序列 求子相关 FFT后得到功率谱密度 作为机动检测 （机动后目标残差不再是高斯白噪声 功率谱密度不再是常数）
class ManeuverCompensationStrongTracker(nn.Module):
    def __init__(self, predictor_in_features: int, predictor_hidden_features: int, predictor_out_features: int, predictor_dropout_rate, predictor_num_layers, predictor_sample_num,
                 predictor_mcu_layer, predictor_mcu_hidden_features, updater_in_features, updater_hidden_features, updater_out_features, updater_dropout_rate, time_series_len, device):
        super(ManeuverCompensationStrongTracker, self).__init__()
        self.predictor = Predictor(predictor_in_features,
                                   predictor_hidden_features,
                                   predictor_out_features,
                                   predictor_dropout_rate,
                                   predictor_num_layers,
                                   predictor_sample_num,
                                   predictor_mcu_layer,
                                   predictor_mcu_hidden_features,
                                   device)
        self.updater = Updater(updater_in_features,
                               updater_hidden_features,
                               updater_out_features,
                               updater_dropout_rate,
                               time_series_len,
                               device)

    def forward(self, update_sigma, normalized_detections, normalized_update_history, normalized_detections_mcu, normalized_update_history_mcu,
                hidden_states_encoder, detection_flag=0):
        # 预测器
        x_predict, x_predict_sigma_log, hidden_states_encoder = self.predictor(update_sigma, normalized_update_history,
                                                                               normalized_detections_mcu, normalized_update_history_mcu, hidden_states_encoder)

        if detection_flag == 0:
            # 滤波  检测到点迹
            # 本次的先验值与历史后验值进行拼接 (batch_size, 5, 6)
            input_update = torch.cat([normalized_update_history[:, 1:, :], x_predict], dim=1)
            # 当前时刻的量测和历史量测 (batch_size, 5, 3)
            input_detection = normalized_detections[:, 1:, :]
            # x_predict_sigma_log: 先验误差方差
            output_update, output_update_sigma, detection_sigma_log = self.updater(input_update, x_predict_sigma_log, input_detection)
        else:
            # 滤波  未检测到点迹
            input_update = torch.cat([normalized_update_history[:, 1:, :], x_predict], dim=1)
            input_detection = torch.cat([normalized_detections[:, 1:-1, :], x_predict[:, :, 0::2]], dim=1)
            output_update, output_update_sigma, detection_sigma_log = self.updater(input_update, x_predict_sigma_log, input_detection)


        return x_predict, x_predict_sigma_log, hidden_states_encoder, output_update, output_update_sigma, detection_sigma_log