from dqkf import DualQuaternionKalmanFilter
from scipy.spatial.transform import Rotation
import numpy as np
import quaternion as qtn
import trimesh


def main():
    vert_noise = 0.002
    pos_range = 10
    ori_range = 180
    pos_g = np.random.uniform(low=-pos_range, high=pos_range, size=(3,))
    ori_g = qtn.from_euler_angles(np.deg2rad(np.random.uniform(low=-ori_range, high=ori_range, size=(3,))))
    print('ground truth quaternion: {}'.format(ori_g))
    obj = trimesh.load('assets/bunny.obj')
    pr = np.array(obj.vertices)
    pb = np.array(obj.vertices) + np.random.uniform(low=-vert_noise, high=vert_noise, size=pr.shape)
    pb = pb @ qtn.as_rotation_matrix(ori_g).T + pos_g.reshape(1, 3)

    dqkf = DualQuaternionKalmanFilter()
    trans, quat = dqkf.fit(pr, pb)
    print('estimated pose: {}, {}'.format(trans, quat))
    print('ground truth pose: {}, {}'.format(pos_g, ori_g))


if __name__ == '__main__':
    main()
