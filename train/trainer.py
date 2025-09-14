import torch
import torch.nn as nn
from scipy.optimize import bracket
from tqdm import tqdm
from config import Args
from torch.utils.data import DataLoader
from torch import optim
from utils.tracksManage import Tracker
from utils.MinMaxScaler import MyMinMaxScaler
from utils.LossCompute import LossCompute_NLL
import time
import config


def train_and_evaluate(model, train_loader: DataLoader, test_loader: DataLoader, optimizer: optim, lr_schedule, logger, args: Args):
    min_max_scaler = MyMinMaxScaler()  # 归一化方法
    min_max_scaler_mcu = MyMinMaxScaler()  # 归一化方法
    mse = nn.MSELoss(reduction='mean')  # 损失函数
    nll = LossCompute_NLL()  # 损失函数
    min_evaluation = 2e8

    for index in range(args.train_epochs):
        model.train()
        train_track_model_loss_sum = []
        test_track_model_loss_sum = []
        train_track_model_update_location_evaluation_sum = []
        train_track_model_update_velocity_evaluation_sum = []
        train_track_model_predict_location_evaluation_sum = []
        train_track_model_predict_velocity_evaluation_sum = []
        with tqdm(total=len(train_loader), desc=f'{index}/{args.train_epochs}') as pbar:
            # 1 加载一个场景的数据，包括 detection 以及 对应的标签
            for (detection, state_labels) in train_loader:
                detection = detection.squeeze(2).to(args.device)  # (batch, n_frames, 3)
                state_labels = state_labels.squeeze(2).to(args.device)  # (batch, n_frames, 9)

                if detection.shape[0] != args.train_batch_size:
                    break

                # 2 初始化缓存类
                tarTracks = Tracker()  # 假设 1 个目标
                tarTracks.track_init(detection, args.predictor_time_series_len, args.predictor_in_features,
                                     args.train_batch_size, args.T, args.device)

                # 3 初始化预测器中 lstm 层两个参数 h 和 c
                h_predict = model.predictor.init_hidden(args.predictor_sampling_num * args.train_batch_size)
                c_predict = model.predictor.init_cell(args.predictor_sampling_num * args.train_batch_size)

                # 定义了一些列表用于存放损失、结果（不用解释）
                total_loss = []
                total_predict_loss = []
                total_update_loss = []
                total_predict_location_evaluation = []
                total_predict_velocity_evaluation = []
                total_update_location_evaluation = []
                total_update_velocity_evaluation = []
                loss = 0
                for frame_index in range(args.predictor_time_series_len, detection.shape[1]):  # detections.shape[1] 为样本轨迹长度
                    config.frame_index = frame_index
                    # 对张量进行最小-最大归一化
                    # predict_history: (batch_size, 5, 6)
                    # update_history: (batch_size, 5, 6)
                    predict_history = torch.cat(tarTracks.x_predict_history, dim=1).clone()  # 其实没用到这个

                    # 4 历史后验序列，形状：(batch_size, 5, 6)
                    update_history = torch.cat(tarTracks.x_update_history, dim=1).clone()

                    # 5 历史后验方差，现状：(batch_size, 5, 6)
                    update_sigma_history = torch.cat(tarTracks.x_sigma, dim=1).detach()

                    # 6 归一化
                    # normalized_state_labels (batch_size, 6, 6) 历史5帧以及当前时刻的标签 维度： v_x y v_y z v_z
                    # normalized_detection (batch_size, 6, 3)   历史5帧以及当前时刻的量测 维度：x y z
                    # normalized_update_history (batch_size, 5, 6)  历史5帧的后验值 维度：x v_x y v_y z v_z
                    (normalized_state_labels, normalized_detection, normalized_update_history, _, _) = \
                        min_max_scaler(state_labels[:, frame_index - args.predictor_time_series_len: frame_index + 1,
                                     [0, 1, 3, 4, 6, 7]],
                                     detection[:, frame_index - args.predictor_time_series_len: frame_index + 1, :],
                                     update_history[:, -args.predictor_time_series_len:, :],
                                     args.T,
                                     args.max_velocity,
                                     mode="-1_1")

                    # 7 机动补偿单元序列（不包含当前时刻） 并进行归一化/补零
                    # normalized_detection_mcu (batch_size, predictor_MCU_len, 3) 历史 predictor_MCU_len 帧的量测
                    # normalized_update_history_mcu (batch_size, predictor_MCU_len, 6) 历史 predictor_MCU_len 帧的后验值
                    tmp = (frame_index - args.predictor_mcu_len) if (frame_index - args.predictor_mcu_len) > 0 else 0
                    (_, normalized_detection_mcu, normalized_update_history_mcu, _, _) = \
                        min_max_scaler_mcu(state_labels[:, tmp: frame_index, [0, 1, 3, 4, 6, 7]],
                                         detection[:, tmp: frame_index, :],
                                         update_history,
                                         args.T,
                                         args.max_velocity,
                                         mode="-1_1")
                    # 如果机动补偿单元序列长度不足 predictor_MCU_len 补0 例如当前预测第6帧 但是需要历史16帧的数据
                    if frame_index - args.predictor_mcu_len < 0:
                        normalized_detection_mcu = torch.cat([torch.zeros([normalized_detection_mcu.shape[0],
                                                                            args.predictor_mcu_len -
                                                                            normalized_detection_mcu.shape[1],
                                                                            normalized_detection_mcu.shape[2]]).to(args.device),
                                                               normalized_detection_mcu], dim=1)
                        normalized_update_history_mcu = torch.cat([torch.zeros([normalized_update_history_mcu.shape[0],
                                                                                args.predictor_mcu_len -
                                                                                normalized_update_history_mcu.shape[1],
                                                                                normalized_update_history_mcu.shape[2]]).to(args.device),
                                                                   normalized_update_history_mcu], dim=1)

                    # 8 预测
                    output_normalized_predict, output_predict_sigma, (h_predict, c_predict), \
                    output_normalized_update, output_update_sigma, detection_sigma_log \
                        = model(update_sigma_history, normalized_detection, normalized_update_history,
                                normalized_detection_mcu, normalized_update_history_mcu, (h_predict, c_predict))

                    # 更新 predict update
                    predict_output_data = min_max_scaler.de_min_max_scaler(output_normalized_predict)
                    update_output_data = min_max_scaler.de_min_max_scaler(output_normalized_update)

                    # 更新历史
                    if len(tarTracks.x_update_history) == args.predictor_mcu_len:
                        tarTracks.x_predict_history.pop(0)
                        tarTracks.x_predict_history.append(predict_output_data)
                        tarTracks.x_sigma.pop(0)
                        tarTracks.x_sigma.append(output_update_sigma)
                        tarTracks.x_update_history.pop(0)
                        tarTracks.x_update_history.append(update_output_data)
                    else:
                        tarTracks.x_predict_history.pop(0)
                        tarTracks.x_predict_history.append(predict_output_data)
                        tarTracks.x_sigma.pop(0)
                        tarTracks.x_sigma.append(output_update_sigma)
                        tarTracks.x_update_history.append(update_output_data)

                    # 计算loss
                    state_labels_copy0 = normalized_state_labels[:, -1, :].unsqueeze(dim=1)
                    detections_copy0 = normalized_detection[:, -1, :].unsqueeze(dim=1)

                    predict_loss = nll(output_normalized_predict, output_predict_sigma, state_labels_copy0) + \
                                   mse(output_normalized_predict, state_labels_copy0)

                    update_loss = nll(detections_copy0, detection_sigma_log, state_labels_copy0[:, :, 0::2]) \
                                  + nll(output_normalized_update, output_update_sigma, state_labels_copy0) + \
                                  mse(output_normalized_update, state_labels_copy0)

                    loss = loss + predict_loss + update_loss

                    single_loss = predict_loss.data + update_loss.data
                    # 所有损失
                    total_loss.append(single_loss.item())
                    total_predict_loss.append(predict_loss.item())
                    total_update_loss.append(update_loss.item())

                    state_labels_copy1 = state_labels[:, frame_index, [0, 1, 3, 4, 6, 7]].unsqueeze(dim=1).data

                    predict_location_evaluation = mse(predict_output_data[:,:,0::2].data, state_labels_copy1[:,:,0::2])
                    predict_velocity_evaluation = mse(predict_output_data[:,:,1::2].data, state_labels_copy1[:,:,1::2])

                    update_location_evaluation = mse(update_output_data[:,:,0::2].data, state_labels_copy1[:,:,0::2])
                    update_velocity_evaluation = mse(update_output_data[:,:,1::2].data, state_labels_copy1[:,:,1::2])

                    total_predict_location_evaluation.append(float(predict_location_evaluation.item()))
                    total_predict_velocity_evaluation.append(float(predict_velocity_evaluation.item()))

                    total_update_location_evaluation.append(float(update_location_evaluation.item()))
                    total_update_velocity_evaluation.append(float(update_velocity_evaluation.item()))

                    train_track_model_loss_sum.append(single_loss.item())

                train_track_model_update_location_evaluation_sum.append(
                    float(update_location_evaluation.item()))
                train_track_model_update_velocity_evaluation_sum.append(
                    float(update_velocity_evaluation.item()))

                train_track_model_predict_location_evaluation_sum.append(
                    float(predict_location_evaluation.item()))
                train_track_model_predict_velocity_evaluation_sum.append(
                    float(predict_velocity_evaluation.item()))

                # 梯度归零
                optimizer.zero_grad()
                # 反向传播
                loss.backward()
                # 更新
                optimizer.step()

                torch.cuda.empty_cache()

                pbar.set_postfix({
                    'train_predict_loss': float(sum(total_predict_loss) / len(total_predict_loss)),
                    'train_update_loss': float(sum(total_update_loss) / len(total_update_loss)),
                    'train_predict_location_evaluation': float(
                        sum(total_predict_location_evaluation) / len(total_predict_location_evaluation)),
                    'train_predict_velocity_evaluation': float(
                        sum(total_predict_velocity_evaluation) / len(total_predict_velocity_evaluation)),
                    'train_update_location_evaluation': float(
                        sum(total_update_location_evaluation) / len(total_update_location_evaluation)),
                    'train_update_velocity_evaluation': float(
                        sum(total_update_velocity_evaluation) / len(total_update_velocity_evaluation)),
                })
                pbar.update(1)

                del tarTracks.x_predict_history
                del tarTracks.x_sigma
                del tarTracks.x_update_history
                del state_labels
                del detection
                del tarTracks

        # evaluation
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc=f'{index}/{args.train_epochs}') as pbar:
                for (detection, state_labels) in test_loader:
                    detection = detection.squeeze(2).to(args.device)  # (batch, n_frames_Ob, ob_num_max, 2)
                    state_labels = state_labels.squeeze(2).to(args.device)  # (batch, n_frames_state_labels, tg_num_max, 4)

                    if detection.shape[0] != args.eval_batch_size:
                        break

                    # 初始化类数组
                    tarTracks = Tracker()  # 假设1个目标
                    tarTracks.track_init(detection, args.predictor_time_series_len, args.predictor_in_features, args.eval_batch_size, args.T, args.device)

                    h_predict = model.predictor.init_hidden(args.predictor_sampling_num * args.eval_batch_size)

                    c_predict = model.predictor.init_cell(args.predictor_sampling_num * args.eval_batch_size)

                    total_loss = []
                    total_predict_loss = []
                    total_update_loss = []
                    total_predict_location_evaluation = []
                    total_predict_velocity_evaluation = []
                    total_update_location_evaluation = []
                    total_update_velocity_evaluation = []

                    loss = 0
                    for frame_index in range(args.predictor_time_series_len, detection.shape[1]):
                        # 对张量进行最小-最大归一化
                        predict_history = torch.cat(tarTracks.x_predict_history, dim=1).clone()
                        update_history = torch.cat(tarTracks.x_update_history, dim=1).clone()
                        (normalized_state_labels, normalized_detection, normalized_update_history,
                         min_vals, max_vals) = \
                            min_max_scaler(
                                state_labels[:, frame_index - args.predictor_time_series_len: frame_index + 1,
                                [0, 1, 3, 4, 6, 7]],
                                detection[:, frame_index - args.predictor_time_series_len: frame_index + 1, :],
                                update_history[:, -args.predictor_time_series_len:, :], args.T, args.max_velocity,
                                mode="-1_1")

                        tmp = (frame_index - args.predictor_mcu_len) if (frame_index - args.predictor_mcu_len) > 0 else 0
                        (_, normalized_detection_mcu, normalized_update_history_mcu,
                         min_vals, max_vals) = \
                            min_max_scaler_mcu(state_labels[:, tmp: frame_index,
                                             [0, 1, 3, 4, 6, 7]],
                                             detection[:, tmp: frame_index, :],
                                             update_history, args.T, args.max_velocity, mode="-1_1")

                        if frame_index - args.predictor_mcu_len < 0:
                            normalized_detection_mcu = torch.cat([torch.zeros([normalized_detection_mcu.shape[0],
                                                                                args.predictor_mcu_len -
                                                                                normalized_detection_mcu.shape[1],
                                                                                normalized_detection_mcu.shape[2]]).to(
                                normalized_detection_mcu.device), normalized_detection_mcu], dim=1)
                            normalized_update_history_mcu = torch.cat(
                                [torch.zeros([normalized_update_history_mcu.shape[0],
                                              args.predictor_mcu_len - normalized_update_history_mcu.shape[1],
                                              normalized_update_history_mcu.shape[2]]).to(
                                    normalized_update_history_mcu.device), normalized_update_history_mcu], dim=1)
                            
                        update_sigma_history = torch.cat(tarTracks.x_sigma, dim=1).detach()

                        normalized_detection = normalized_detection.squeeze(dim=2)
                        normalized_detection_mcu = normalized_detection_mcu.squeeze(dim=2)
                        # 预测
                        output_normalized_predict, output_predict_sigma, (h_predict, c_predict), \
                        output_normalized_update, output_update_sigma, \
                        detection_sigma_log \
                            = model(update_sigma_history, normalized_detection,
                                    normalized_update_history.to(args.device), normalized_detection_mcu,
                                    normalized_update_history_mcu.to(args.device),
                                    (h_predict, c_predict))

                        # 更新 predict update
                        predict_output_data = min_max_scaler.de_min_max_scaler(output_normalized_predict)
                        update_output_data = min_max_scaler.de_min_max_scaler(output_normalized_update)

                        # 更新历史
                        if len(tarTracks.x_update_history) == args.predictor_mcu_len:
                            tarTracks.x_predict_history.pop(0)
                            tarTracks.x_predict_history.append(predict_output_data)
                            tarTracks.x_sigma.pop(0)
                            tarTracks.x_sigma.append(output_update_sigma)
                            tarTracks.x_update_history.pop(0)
                            tarTracks.x_update_history.append(update_output_data)
                        else:
                            tarTracks.x_predict_history.pop(0)
                            tarTracks.x_predict_history.append(predict_output_data)
                            tarTracks.x_sigma.pop(0)
                            tarTracks.x_sigma.append(output_update_sigma)
                            tarTracks.x_update_history.append(update_output_data)

                        # 计算loss
                        state_labels_copy0 = normalized_state_labels[:, -1, :].unsqueeze(dim=1)
                        detections_copy0 = normalized_detection[:, -1, :].unsqueeze(dim=1)

                        predict_loss = nll(output_normalized_predict, output_predict_sigma,
                                           state_labels_copy0) + \
                                       mse(output_normalized_predict, state_labels_copy0)

                        update_loss = nll(detections_copy0, detection_sigma_log,
                                          state_labels_copy0[:, :, 0::2]) \
                                      + nll(output_normalized_update, output_update_sigma,
                                            state_labels_copy0) + \
                                      mse(output_normalized_update, state_labels_copy0)

                        loss = loss + predict_loss + update_loss

                        single_loss = predict_loss.data + update_loss.data
                        # 所有损失
                        total_loss.append(single_loss.item())
                        total_predict_loss.append(predict_loss.item())
                        total_update_loss.append(update_loss.item())

                        state_labels_copy1 = state_labels[:, frame_index, [0, 1, 3, 4, 6, 7]].unsqueeze(dim=1).data

                        predict_location_evaluation = mse(predict_output_data[:, :, 0::2].data,
                                                          state_labels_copy1[:, :, 0::2])
                        predict_velocity_evaluation = mse(predict_output_data[:, :, 1::2].data,
                                                          state_labels_copy1[:, :, 1::2])

                        update_location_evaluation = mse(update_output_data[:, :, 0::2].data,
                                                         state_labels_copy1[:, :, 0::2])
                        update_velocity_evaluation = mse(update_output_data[:, :, 1::2].data,
                                                         state_labels_copy1[:, :, 1::2])

                        total_predict_location_evaluation.append(float(predict_location_evaluation.item()))
                        total_predict_velocity_evaluation.append(float(predict_velocity_evaluation.item()))

                        total_update_location_evaluation.append(float(update_location_evaluation.item()))
                        total_update_velocity_evaluation.append(float(update_velocity_evaluation.item()))

                        test_track_model_loss_sum.append(single_loss.item())

                    torch.cuda.empty_cache()

                    pbar.set_postfix({
                        'test_predict_loss': float(sum(total_predict_loss) / len(total_predict_loss)),
                        'test_update_loss': float(sum(total_update_loss) / len(total_update_loss)),
                        'test_predict_location_evaluation': float(
                            sum(total_predict_location_evaluation) / len(total_predict_location_evaluation)),
                        'test_predict_velocity_evaluation': float(
                            sum(total_predict_velocity_evaluation) / len(total_predict_velocity_evaluation)),
                        'test_update_location_evaluation': float(
                            sum(total_update_location_evaluation) / len(total_update_location_evaluation)),
                        'test_update_velocity_evaluation': float(
                            sum(total_update_velocity_evaluation) / len(total_update_velocity_evaluation)),
                    })
                    pbar.update(1)

                    del tarTracks.x_predict_history
                    del tarTracks.x_sigma
                    del tarTracks.x_update_history
                    del state_labels
                    del detection
                    del tarTracks

        lr_schedule.step(sum(train_track_model_loss_sum) / len(train_track_model_loss_sum))

        logger.debug("train_loss:{},index:{}".format(sum(train_track_model_loss_sum) / len(train_track_model_loss_sum),
                                                    index))
        logger.debug("train_predict_location_evaluation:{},index:{}".format(
            sum(train_track_model_predict_location_evaluation_sum) / len(train_track_model_predict_location_evaluation_sum), index))
        logger.debug("train_predict_velocity_evaluation:{},index:{}".format(
            sum(train_track_model_predict_velocity_evaluation_sum) / len(train_track_model_predict_velocity_evaluation_sum), index))
        logger.debug("train_update_location_evaluation:{},index:{}".format(
            sum(train_track_model_update_location_evaluation_sum) / len(train_track_model_update_location_evaluation_sum), index))
        logger.debug("train_update_velocity_evaluation:{},index:{}".format(
            sum(train_track_model_update_velocity_evaluation_sum) / len(train_track_model_update_velocity_evaluation_sum), index))

        now = time.localtime()
        nowt = time.strftime("%Y_%m_%d_%H_%M_", now)
        train_evaluation = sum(train_track_model_update_location_evaluation_sum) / len(train_track_model_update_location_evaluation_sum) \
                          + sum(train_track_model_update_velocity_evaluation_sum) / len(train_track_model_update_velocity_evaluation_sum)

        if min_evaluation > train_evaluation:
            min_evaluation = train_evaluation
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': sum(test_track_model_loss_sum) / len(test_track_model_loss_sum),
                'lr_schedule': lr_schedule.state_dict()
            }, args.output_dir + 'ManeuverCompensationStrongTracker3D/' + str(nowt) + ".pth")

