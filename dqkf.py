from utils import Rotation
from scipy.linalg import block_diag
import numpy as np
import quaternion as qtn


class DualQuaternionKalmanFilter(object):
    """
    Implementation of dual quaternion kalman filter based on the following paper
    https://ieeexplore.ieee.org/abstract/document/1603413
    """
    def __init__(self, max_vec=20, th_quat=1e-4, rho=1):
        self.max_vec = max_vec
        self.th_quat = th_quat
        self.rho = rho

    def fit(self, pr, pb):
        """
        Compute the relative transformation from reference frame to body frame
        :param pr: N 3-D points in reference frame
        :param pb: N 3-D points in body frame
        :return: 3-D vector for translation and 4-D vector for orientation
        """
        num_pts = pr.shape[0]
        assert num_pts > 1
        i, j = np.meshgrid(np.arange(num_pts), np.arange(num_pts), indexing='ij')
        flag = i < j
        i, j = i[flag], j[flag]
        prv = pr[i] - pr[j]
        pbv = pb[i] - pb[j]
        mpr, mpb = np.mean(pr, axis=0), np.mean(pb, axis=0)
        curr_quat = np.random.rand(4)
        curr_quat = curr_quat / np.linalg.norm(curr_quat)
        curr_cov_quat = np.eye(4) * 100
        error = 1e7
        i = 0
        ts, rs = list(), list()
        while error > self.th_quat:
            rv, bv = self.sample_vectors(prv, pbv)
            num_vec = rv.shape[0]
            h = self.get_obs_mat(rv, bv)
            xxt = curr_quat.reshape(4, 1) @ curr_quat.reshape(1, 4)
            cov_h = 0.25 * self.rho * (np.trace(xxt + curr_cov_quat) * np.eye(4) - (xxt + curr_cov_quat))
            cov_h = block_diag(*tuple([cov_h]*num_vec))
            k = curr_cov_quat @ h.T @ np.linalg.inv(h @ curr_cov_quat @ h.T + cov_h)
            next_cov_quat = (np.eye(4) - k @ h) @ curr_cov_quat
            next_quat = curr_quat - k @ (h @ curr_quat)
            norm = np.linalg.norm(next_quat)
            next_quat = next_quat / norm
            error = np.linalg.norm(next_quat - curr_quat)
            print('iter: {} | rmse: {}'.format(i, error))
            curr_quat = next_quat
            curr_cov_quat = next_cov_quat / norm
            i += 1
            r = Rotation.from_quat(curr_quat)
            rot = r.as_mat()
            trans = mpb - rot @ mpr
            rs.append(r)
            ts.append(trans)
        return ts, rs

    def sample_vectors(self, pv, qv):
        num_vec = pv.shape[0]
        ids = np.random.choice(np.arange(num_vec), min(self.max_vec, num_vec), replace=False)
        return pv[ids], qv[ids]

    def get_obs_mat(self, r, b):
        num_h = r.shape[0]
        h = np.zeros((num_h, 4, 4))
        s = 0.5 * (b + r)
        d = 0.5 * (b - r)
        h[:, 1:4, 1:4] = -self.skew(s)
        h[:, 1:4, 0] = d
        h[:, 0, 1:4] = -d
        h = np.concatenate([np.squeeze(mat) for mat in np.split(h, num_h, axis=0)], axis=0)
        return h

    @staticmethod
    def skew(v):
        zeros = np.zeros_like(v[:, 0])
        return np.stack([zeros, -v[:, 2], v[:, 1], v[:, 2], zeros, -v[:, 0], -v[:, 1], v[:, 0], zeros], axis=1).reshape(-1, 3, 3)

    @staticmethod
    def normalize(pts):
        u = np.mean(pts, axis=0)
        norm_pts = pts - u.reshape(1, 3)
        return norm_pts, u
