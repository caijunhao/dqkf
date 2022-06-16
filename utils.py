import quaternion as qtn
import copy


class Rotation(object):
    """
    A rotation class based on numpy quaternion.
    """
    def __init__(self):
        self.quat = qtn.quaternion()

    @staticmethod
    def from_quat(quat):
        q = Rotation()
        q.quat = qtn.from_float_array(quat)
        return q

    @staticmethod
    def from_mat(mat):
        q = Rotation()
        q.quat = qtn.from_rotation_matrix(mat)
        return q

    @staticmethod
    def from_euler(euler):
        q = Rotation()
        q.quat = qtn.from_euler_angles(euler)
        return q

    def as_quat(self):
        return qtn.as_float_array(self.quat)

    def as_mat(self):
        return qtn.as_rotation_matrix(self.quat)

    def conj(self):
        q = copy.deepcopy(self)
        q.quat = q.quat.conj()
        return q

    def __mul__(self, qr):
        ql = copy.deepcopy(self)
        ql.quat = ql.quat * qr.quat
        return ql

    def __str__(self):
        return str(self.quat)

    def __repr__(self):
        return str(self.quat)

