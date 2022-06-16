from dqkf import DualQuaternionKalmanFilter
from utils import Rotation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d


def main():
    # generate data
    vert_noise = 0.002
    pos_range = 10
    ori_range = 180
    pos_g = np.random.uniform(low=-pos_range, high=pos_range, size=(3,))
    ori_g = Rotation.from_euler(np.deg2rad(np.random.uniform(low=-ori_range, high=ori_range, size=(3,))))
    pose_g = np.eye(4)
    pose_g[0:3, 0:3], pose_g[0:3, 3] = ori_g.as_mat(), pos_g
    print('ground truth quaternion: {}'.format(ori_g))
    m = o3d.io.read_triangle_mesh('assets/bunny.obj')
    pcr = np.asarray(m.sample_points_uniformly(number_of_points=500).points)
    pcb = pcr @ ori_g.as_mat().T + pos_g.reshape(1, 3)
    vs = np.asarray(m.vertices)
    pr = vs + np.random.uniform(low=-vert_noise, high=vert_noise, size=vs.shape)
    pb = vs + np.random.uniform(low=-vert_noise, high=vert_noise, size=vs.shape)
    pb = pb @ ori_g.as_mat().T + pos_g.reshape(1, 3)
    # pose estimation
    dqkf = DualQuaternionKalmanFilter()
    ts, rs = dqkf.fit(pr, pb)
    # uncomment to visualize the animation of pose estimation
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    prev_pose, curr_pose = np.eye(4), np.eye(4)
    for t, r in zip(ts, rs):
        mat = r.as_mat()
        curr_pose[0:3, 0:3], curr_pose[0:3, 3] = mat, t
        curr_inv_pose = np.linalg.inv(curr_pose)
        pcr_e = pcb @ curr_inv_pose[0:3, 0:3].T + curr_inv_pose[0:3, 3]
        ax.clear()
        ax.scatter(pcr[:, 0], pcr[:, 1], pcr[:, 2], marker='^')
        ax.scatter(pcr_e[:, 0], pcr_e[:, 1], pcr_e[:, 2], marker='s')
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.show()
    print('estimated pose: {}, {}'.format(ts[-1], rs[-1]))
    print('ground truth pose: {}, {}'.format(pos_g, ori_g))


if __name__ == '__main__':
    main()
