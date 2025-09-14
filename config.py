import argparse
import torch
import os

class Args:
    def __init__(self):
        self.output_dir = None
        self.predictor_mcu_len = None
        self.max_velocity = None
        self.eval_batch_size = None
        self.predictor_sampling_num = None  # 采样数量
        self.predictor_in_features = None
        self.predictor_time_series_len = None
        self.train_batch_size = None # 训练批量大小
        self.device = None  # cpu or gpu
        self.train_epochs = None # 训练轮数
        self.T = None # 采样周期

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(description='MCST')
        return parser

    @staticmethod
    def initialize(parser):
        # 获取项目根目录的绝对路径
        base_path = os.path.abspath(os.path.dirname(__file__))
        
        # args for path
        parser.add_argument('--output_dir', default=os.path.join(base_path, 'checkpoints\\'),
                            help='the output dir for model checkpoints')
        parser.add_argument('--checkpoint', default=os.path.join(base_path, 'checkpoints', 'ManeuverCompensationStrongTracker3D', '2025_05_09_05_25_.pth'), type=str,
                            help='Path to model checkpoint')
        parser.add_argument('--data_dir', default=os.path.join(base_path, 'data'), type=str,
                            help='data dir for uer')
        parser.add_argument('--data_dir_test', default=os.path.join(base_path, 'data'), type=str,
                            help='evaluation data dir for uer')
        parser.add_argument('--log_dir', default=os.path.join(base_path, 'log', 'demo_log.log'), type=str,
                            help='log dir for uer')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed') # 随机数的种子

        parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"), # cpu or gpu
                            help='cpu or gpu')
        parser.add_argument('--train_batch_size', default=256, type=int) # 批量大小
        parser.add_argument('--train_epochs', default=50, type=int, # 训练的轮数
                            help='Max training epoch')
        parser.add_argument('--test_epochs', default=10, type=int,
                            help='Max training epoch')
        parser.add_argument('--eval_batch_size', default=64, type=int) # 测试的批量大小
        parser.add_argument('--optimizer_factor', default=0.8, type=int)
        parser.add_argument('--optimizer_patience', default=5, type=int)
        parser.add_argument('--lr', default=1e-3, type=float,
                            help='learning rate')
        # train args
        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')
        parser.add_argument('--T', default=0.4, type=float,
                            help='data rate') # 0.4  3
        parser.add_argument('--predictor_in_features', default=6, type=float)
        parser.add_argument('--predictor_hidden_features', default=64, type=float)
        parser.add_argument('--predictor_out_features', default=6, type=float,)
        parser.add_argument('--predictor_lstm_num_layers', default=2, type=float,)
        parser.add_argument('--predictor_mcu_num_layers', default=2, type=float)
        parser.add_argument('--predictor_sampling_num', default=5, type=int)
        parser.add_argument('--predictor_time_series_len', default=5, type=float)
        parser.add_argument('--predictor_mcu_len', default=16, type=float,)
        parser.add_argument('--predictor_mcu_layer', default=2, type=float, )
        parser.add_argument('--predictor_mcu_hidden_features', default=256, type=float, )

        parser.add_argument('--updater_in_features', default=3, type=float)
        parser.add_argument('--updater_hidden_features', default=128, type=float)
        parser.add_argument('--updater_out_features', default=6, type=float)
        parser.add_argument('--updater_dropout_rate', default=0.1, type=float)
        parser.add_argument('--frame_max', default=100, type=int,
                            help='the maximum number of frames in a track')
        parser.add_argument('--max_velocity', default=340*5, type=float)  #  340*5

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()

frame_index = 0