import numpy as np
import matplotlib.pyplot as plt
from pygame.surfarray import pixels_red

if __name__ == '__main__':
    detections = np.load("data/train/scene0001/detections.npy")
    state_labels = np.load("data/train/scene0001/state_labels.npy")
    print("detections原始形状：", detections.shape)  # 现有代码保留
    print("state_labels原始形状：", state_labels.shape)  # 现有代码保留

    detections = np.squeeze(detections, axis=-1)
    state_labels = np.squeeze(state_labels, axis=-1)

    frames = np.arange(100)
    x_det, y_det, z_det = detections[:,0], detections[:,1], detections[:,2]
    # 图1：detection可视化（修改原fig为fig1）
    fig1 = plt.figure(figsize=(10, 5), num="Detection Visualization")
    ax3d_det = fig1.add_subplot(121, projection='3d')
    ax3d_det.plot(x_det, y_det, z_det, '-', label='Detection')
    ax3d_det.set_xlabel('X')
    ax3d_det.set_ylabel('Y')
    ax3d_det.set_zlabel('Z')
    ax3d_det.set_title('Detection 3D Trajectory')
    ax3d_det.legend()

    ax2d_det = fig1.add_subplot(122)
    ax2d_det.plot(frames, x_det, 'r-', label='X')
    ax2d_det.plot(frames, y_det, 'g-', label='Y')
    ax2d_det.plot(frames, z_det, 'b-', label='Z')
    ax2d_det.set_xlabel('Frame')
    ax2d_det.set_ylabel('Coordinate')
    ax2d_det.set_title('Detection Coordinate vs Frame')
    ax2d_det.legend()

    # 提取state_label完整分量（新增速度、加速度提取）
    x_label, y_label, z_label = state_labels[:, 0], state_labels[:, 3], state_labels[:, 6]  # 位置分量
    vx_label, vy_label, vz_label = state_labels[:, 1], state_labels[:, 4], state_labels[:, 7]  # 速度分量
    ax_label, ay_label, az_label = state_labels[:, 2], state_labels[:, 5], state_labels[:, 8]  # 加速度分量

    # ... 图1（detection）代码保持不变 ...

    # 图2：state_label可视化（修改为4个子图）
    fig2 = plt.figure(figsize=(12, 8), num="State Label Visualization")  # 调整尺寸适应4个子图

    # 子图1：位置三维轨迹（原三维轨迹）
    ax3d_pos = fig2.add_subplot(221, projection='3d')
    ax3d_pos.plot(x_label, y_label, z_label, 'r-', label='Position')
    ax3d_pos.set_xlabel('X')
    ax3d_pos.set_ylabel('Y')
    ax3d_pos.set_zlabel('Z')
    ax3d_pos.set_title('State Label Position 3D Trajectory')
    ax3d_pos.legend()

    # 子图2：位置时间序列（原时间序列）
    ax2d_pos = fig2.add_subplot(222)
    ax2d_pos.plot(frames, x_label, 'r-', label='X Position')
    ax2d_pos.plot(frames, y_label, 'g-', label='Y Position')
    ax2d_pos.plot(frames, z_label, 'b-', label='Z Position')
    ax2d_pos.set_xlabel('Frame')
    ax2d_pos.set_ylabel('Coordinate')
    ax2d_pos.set_title('Position vs Frame')
    ax2d_pos.legend()

    # 子图3：速度时间序列（新增）
    ax2d_vel = fig2.add_subplot(223)
    ax2d_vel.plot(frames, vx_label, 'r--', label='X Velocity')
    ax2d_vel.plot(frames, vy_label, 'g--', label='Y Velocity')
    ax2d_vel.plot(frames, vz_label, 'b--', label='Z Velocity')
    ax2d_vel.set_xlabel('Frame')
    ax2d_vel.set_ylabel('Velocity')
    ax2d_vel.set_title('Velocity vs Frame')
    ax2d_vel.legend()

    # 子图4：加速度时间序列（新增）
    ax2d_acc = fig2.add_subplot(224)
    ax2d_acc.plot(frames, ax_label, 'r:', label='X Acceleration')
    ax2d_acc.plot(frames, ay_label, 'g:', label='Y Acceleration')
    ax2d_acc.plot(frames, az_label, 'b:', label='Z Acceleration')
    ax2d_acc.set_xlabel('Frame')
    ax2d_acc.set_ylabel('Acceleration')
    ax2d_acc.set_title('Acceleration vs Frame')
    ax2d_acc.legend()

    plt.tight_layout()
    plt.show()