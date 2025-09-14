import numpy as np
import os
import scipy.io as io
import torch
import torch.utils.data as data
from path import Path
from torch.distributions import Normal

class DataLoadOneManeuveringTarget3d(data.Dataset):
    def __init__(self, data_path, is_train, n_frames):
        """
        param num_objects: a list of number of possible objects.
        """
        super(DataLoadOneManeuveringTarget3d, self).__init__()
        if is_train:
            self.data_root = Path(data_path + "/train")
        else:
            self.data_root = Path(data_path + "/test")
        self.dataSet_length = len(os.listdir(self.data_root))
        self.all_scene_path = [str(path) for path in os.listdir(self.data_root)]
        self.n_frames = n_frames

    def __getitem__(self, idx):
        scene_path = self.all_scene_path[idx]
        scene_path = os.path.join(self.data_root, scene_path)
        data_list = os.listdir(scene_path)

        data_list.sort()

        detail_data_path = os.path.join(scene_path, data_list[0])
        detections_ndarry = np.load(detail_data_path, allow_pickle=True).tolist()
        detections = torch.tensor(detections_ndarry, dtype=torch.float32).permute(0, 2, 1)

        detail_data_path = os.path.join(scene_path, data_list[1])
        state_labels_ndarry = np.load(detail_data_path, allow_pickle=True).tolist()
        state_labels = torch.tensor(state_labels_ndarry, dtype=torch.float32).permute(0, 2, 1)

        return [detections, state_labels]

    def __len__(self):
        return self.dataSet_length


def DataLoadFromMatlab_oneTarget_3D_for_paper():
    state_labels = []
    detections = []
    # 获取项目根目录的绝对路径
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_path, "dataset", "scene007")
    
    for frame in range(1, 201):
        path = os.path.join(dataset_path, "state_labels" + ('%02d' % frame) + ".mat")
        matr = io.loadmat(path)
        savedata_origin = matr.get('savedata')
        savedata_origin = savedata_origin.astype(float)
        state_labels.append(torch.tensor(np.transpose(savedata_origin)).unsqueeze(dim=0))
    state_labels = torch.cat(state_labels, dim=0)

    for frame in range(1, 201):
        path = os.path.join(dataset_path, "obj" + ('%02d' % frame) + ".mat")
        matr = io.loadmat(path)
        savedata_ob = matr.get('savedata')
        savedata_ob = savedata_ob.astype(float)
        detections.append(torch.tensor(np.transpose(savedata_ob)).unsqueeze(dim=0))
    detections = torch.cat(detections, dim=0)

    state_labels = state_labels.unsqueeze(dim=0)
    detections = detections.unsqueeze(dim=0)

    return state_labels, detections


def enu2rae(enu_x, enu_y, enu_z):
    distance = (enu_x ** 2 + enu_y ** 2 + enu_z ** 2) ** 0.5
    azi = torch.atan2(enu_y, enu_x)
    ele = torch.atan2(enu_z, (enu_x ** 2 + enu_y ** 2) ** 0.5)
    return (distance, azi, ele)


def rae2enu(distance, azi, ele):
    enu_x = distance * torch.cos(azi) * torch.cos(ele)
    enu_y = distance * torch.sin(azi) * torch.cos(ele)
    enu_z = distance * torch.sin(ele)
    return (enu_x, enu_y, enu_z)