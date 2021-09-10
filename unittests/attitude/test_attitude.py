from unittest import TestCase

import numpy as np

from giant import rotations as at


class TestAttitude(TestCase):

    def check_attitude(self, attitude, quaternion, mupdate, vupdate):

        np.testing.assert_array_almost_equal(quaternion, attitude.q)
        np.testing.assert_array_almost_equal(quaternion[:3], attitude.q_vector)
        self.assertAlmostEqual(quaternion[-1], attitude.q_scalar)
        self.assertIs(attitude._mupdate, mupdate)
        self.assertIs(attitude._vupdate, vupdate)

    def test_init(self):

        att = at.Rotation()

        self.check_attitude(att, [0, 0, 0, 1], True, True)

        att = at.Rotation([0, 0, 0, 1])

        self.check_attitude(att, [0, 0, 0, 1], True, True)

        att = at.Rotation(data=[0, 0, 0, 1])

        self.check_attitude(att, [0, 0, 0, 1], True, True)

        att = at.Rotation(np.eye(3))

        self.check_attitude(att, [0, 0, 0, 1], False, True)

        att = at.Rotation([0, 0, 0])

        self.check_attitude(att, [0, 0, 0, 1], True, False)

        att = at.Rotation([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])

        self.check_attitude(att, [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)

        att = at.Rotation([np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2])

        self.check_attitude(att, [-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)

        att2 = att
        att = at.Rotation(att2)

        self.check_attitude(att, [-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)
        self.assertIs(att, att2)

        # with self.assertWarns(UserWarning):
        #
        #     at.Rotation([1, 2, 3, 4])

    def test_quaternion_setter(self):

        att = at.Rotation()

        att._mupdate = False
        att._vupdate = False

        att.quaternion = [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2]

        self.check_attitude(att, [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)

        att.quaternion = [np.sqrt(2)/2, 0, 0, -np.sqrt(2)/2]

        self.check_attitude(att, [-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)

        att2 = at.Rotation([1, 2, 3, 4])
        att.quaternion = att2

        self.check_attitude(att, np.array([1, 2, 3, 4])/np.sqrt(30), True, True)

        self.assertIsNot(att, att2)

        # with self.assertWarns(UserWarning):
        #     att.quaternion = [1, 2, 3, 4]
        #
        #     self.check_attitude(att, np.array([1, 2, 3, 4])/np.sqrt(30), True, True)

        with self.assertRaises(ValueError):

            att.quaternion = np.eye(4)

    def test_matrix_getter(self):

        att = at.Rotation([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])

        np.testing.assert_array_almost_equal([[1, 0, 0], [0, 0, -1], [0, 1, 0]], att.matrix)

        np.testing.assert_array_almost_equal([[1, 0, 0], [0, 0, -1], [0, 1, 0]], att._matrix)
        self.assertFalse(att._mupdate)

        # this is bad and you should never do this but it checks that the caching is working
        att._matrix = np.eye(3)

        np.testing.assert_array_equal(att.matrix, np.eye(3))

        self.check_attitude(att, [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], False, True)

    def test_matrix_setter(self):

        att = at.Rotation([1, 2, 3])

        att.matrix = np.eye(3)

        self.check_attitude(att, [0, 0, 0, 1], False, True)

        np.testing.assert_array_equal(att._matrix, np.eye(3))

        with self.assertRaises(ValueError):

            att.matrix = [1, 2, 3]

    def test_vector_getter(self):

        att = at.Rotation([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])

        np.testing.assert_array_almost_equal(att.vector, [np.pi/2, 0, 0])
        np.testing.assert_array_almost_equal(att._vector, [np.pi/2, 0, 0])
        self.assertFalse(att._vupdate)

        # this is bad and you should never do this but it checks that the caching is working
        att._vector = [1, 2, 3]

        np.testing.assert_array_equal(att.vector, [1, 2, 3])

        self.check_attitude(att, [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, False)

    def test_vector_setter(self):

        att = at.Rotation([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])

        att.vector = [1, 2, 3]

        self.check_attitude(att, [-0.25532186,  -0.51064372,  -0.76596558, 0.29555113], True, False)

        np.testing.assert_array_equal(att.vector, [1, 2, 3])

        with self.assertRaises(ValueError):

            att.vector = np.eye(3)

    def test_inv(self):

        att = at.Rotation([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])

        attinv = att.inv()

        self.check_attitude(attinv, [-np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)
        self.check_attitude(att, [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)

    def test_interp_attitude(self):

        att = at.Rotation()

        att.interp_attitude([1, 2, 3])
        self.check_attitude(att, [-0.25532186,  -0.51064372,  -0.76596558, 0.29555113], True, False)
        np.testing.assert_array_equal(att._vector, [1, 2, 3])

        att.interp_attitude(np.eye(3))
        self.check_attitude(att, [0, 0, 0, 1], False, True)
        np.testing.assert_array_equal(att._matrix, np.eye(3))

        att.interp_attitude([np.sqrt(2)/2, 0, 0, np.sqrt(2)/2])
        self.check_attitude(att, [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)

        att2 = at.Rotation([np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
        att.interp_attitude(att2)
        self.check_attitude(att, [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2], True, True)
        self.assertIsNot(att, att2)

        with self.assertRaises(ValueError):
            att.interp_attitude([1, 2])

    def test_eq(self):

        att = at.Rotation()

        self.assertTrue(att == at.Rotation())
        self.assertTrue(att == [0, 0, 0, 1])
        self.assertTrue(att == np.eye(3))
        self.assertTrue(att == [0, 0, 0])

    def test_mul(self):

        att = at.Rotation([1, 2, 3])

        att2 = att.inv()

        self.check_attitude(att*att2, [0, 0, 0, 1], True, True)

        with self.assertRaises(TypeError):

            _ = att*[0, 0, 0, 1]

        with self.assertRaises(TypeError):

            _ = [0, 0, 0, 1]*att

    # def test_imul(self):
    #     att = at.Rotation()
    #     with self.assertWarns(DeprecationWarning):
    #
    #         att *= [1, 0, 0, 0]
    #
    #         self.check_attitude(att, [1, 0, 0, 0], True, True)

    def test_rotate(self):

        att = at.Rotation()

        att.rotate([1, 0, 0, 0])

        self.check_attitude(att, [1, 0, 0, 0], True, True)


class TestQuaternionInverse(TestCase):

    def test_quaternion_inverse(self):

        qinv = at.quaternion_inverse([1, 2, 3, 4])

        np.testing.assert_array_equal(qinv, [-1, -2, -3, 4])

        qinv = at.quaternion_inverse(at.Rotation([1, 2, 3, 4]))

        np.testing.assert_array_almost_equal(qinv.q.flatten(), np.array([-1, -2, -3, 4])/np.sqrt(30))

        qinv = at.quaternion_inverse([[1, 2], [2, 3], [3, 4], [4, 5]])

        np.testing.assert_array_equal(qinv.T, [[-1, -2, -3, 4], [-2, -3, -4, 5]])


class TestQuaternionMultiplication(TestCase):

    def test_quaternion_multiplication(self):

        quat_1 = [1, 0, 0, 0]
        quat_2 = [0, 1, 0, 0]

        qm = at.quaternion_multiplication(quat_1, quat_2)

        np.testing.assert_array_equal(np.abs(qm), [0, 0, 1, 0])

        quat_1 = [[1], [0], [0], [0]]
        quat_2 = [[0], [1], [0], [0]]

        qm = at.quaternion_multiplication(quat_1, quat_2)

        np.testing.assert_array_equal(np.abs(qm), [[0], [0], [1], [0]])

        quat_1 = [[1, 0], [0, 1], [0, 0], [0, 0]]
        quat_2 = [[0, 0], [1, 1], [0, 0], [0, 0]]

        qm = at.quaternion_multiplication(quat_1, quat_2)

        np.testing.assert_array_equal(np.abs(qm), [[0, 0], [0, 0], [1, 0], [0, 1]])

        quat_1 = at.Rotation([1, 0, 0, 0])
        quat_2 = at.Rotation([0, 0, 1, 0])

        qm = at.quaternion_multiplication(quat_1, quat_2)

        np.testing.assert_array_equal(np.abs(qm.q), [0, 1, 0, 0])

        quat_1 = [np.sqrt(2)/2, 0, 0, np.sqrt(2)/2]  # x=x, y=z, z=-y
        quat_2 = [0, np.sqrt(2)/2, 0, np.sqrt(2)/2]  # x=-z, y=y, z=x

        qm = at.quaternion_multiplication(quat_1, quat_2)

        np.testing.assert_array_almost_equal(np.abs(qm), [0.5, 0.5, 0.5, 0.5])

        quat_1 = [0.25532186, 0.51064372, 0.76596558, -0.29555113]

        quat_2 = [-0.43199286, -0.53999107, -0.64798929, -0.31922045]

        qm = at.quaternion_multiplication(quat_1, quat_2)

        # truth comes from matrix rotations
        np.testing.assert_array_almost_equal(qm, [0.12889493, -0.16885878, 0.02972499, 0.97672373])


class TestQuaternionToRotVec(TestCase):

    def test_quaternion_to_rotvec(self):

        rvec = at.quaternion_to_rotvec([1, 0, 0, 0])

        np.testing.assert_array_almost_equal(rvec, [np.pi, 0, 0])

        rvec = at.quaternion_to_rotvec(at.Rotation([-1, 0, 0, 0]))

        np.testing.assert_array_almost_equal(rvec, [-np.pi, 0, 0])

        rvec = at.quaternion_to_rotvec([0, 1, 0, 0])

        np.testing.assert_array_almost_equal(rvec, [0, np.pi, 0])

        rvec = at.quaternion_to_rotvec([0, -1, 0, 0])

        np.testing.assert_array_almost_equal(rvec, [0, -np.pi, 0])

        rvec = at.quaternion_to_rotvec([0, 0, 1, 0])

        np.testing.assert_array_almost_equal(rvec, [0, 0, np.pi])

        rvec = at.quaternion_to_rotvec([0, 0, -1, 0])

        np.testing.assert_array_almost_equal(rvec, [0, 0, -np.pi])

        rvec = at.quaternion_to_rotvec([0, 0, 0, 1])

        np.testing.assert_array_almost_equal(rvec, [0, 0, 0])

        rvec = at.quaternion_to_rotvec([0, 0, 0, -1])

        np.testing.assert_array_almost_equal(rvec, [0, 0, 0])

        rvec = at.quaternion_to_rotvec([0.25532186,  0.51064372,  0.76596558, -0.29555113])

        np.testing.assert_array_almost_equal(rvec, [1, 2, 3])

        rvec = at.quaternion_to_rotvec([-0.25532186,  -0.51064372,  -0.76596558, 0.29555113])

        # euler axis is not unique
        np.testing.assert_array_almost_equal(rvec, np.array([1, 2, 3])*(1-2*np.pi/np.sqrt(14)))

        rvec = at.quaternion_to_rotvec([[0.25532186],  [0.51064372],  [0.76596558], [-0.29555113]])

        np.testing.assert_array_almost_equal(rvec, [[1], [2], [3]])

        rvec = at.quaternion_to_rotvec([[1, 0, 0.25532186, 0],
                                        [0, 0, 0.51064372, 0],
                                        [0, 0, 0.76596558, 0],
                                        [0, 1, -0.29555113, -1]])

        np.testing.assert_array_almost_equal(rvec, [[np.pi, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]])


class TestQuaternionToRotMat(TestCase):

    def test_quaternion_to_rotmat(self):

        rotmat = at.quaternion_to_rotmat([0, 0, 0, 1])

        np.testing.assert_array_almost_equal(rotmat, np.eye(3))

        rotmat = at.quaternion_to_rotmat([[0], [0], [0], [1]])

        np.testing.assert_array_almost_equal(rotmat, np.eye(3))

        rotmat = at.quaternion_to_rotmat([[0, 1], [0, 0], [0, 0], [1, 0]])

        np.testing.assert_array_almost_equal(rotmat, [np.eye(3), [[1, 0, 0], [0, -1, 0], [0, 0, -1]]])

        rotmat = at.quaternion_to_rotmat(at.Rotation([0, 1, 0, 0]))

        np.testing.assert_array_almost_equal(rotmat, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        rotmat = at.quaternion_to_rotmat([0, 0, np.sqrt(2)/2, np.sqrt(2)/2])

        np.testing.assert_array_almost_equal(rotmat, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        rotmat = at.quaternion_to_rotmat([-0.25532186, -0.51064372, -0.76596558, 0.29555113])

        np.testing.assert_array_almost_equal(rotmat, [[-0.69492056, 0.71352099, 0.08929286],
                                                      [-0.19200697, -0.30378504, 0.93319235],
                                                      [0.69297817, 0.6313497, 0.34810748]])


class TestQuaternionToEuler(TestCase):

    def test_quaternion_to_euler(self):
        orders = ['xyz', 'zxy', 'yxz', 'yzx', 'xzy', 'zyx', 'xyx', 'yxy', 'xzx', 'zxz', 'yzy', 'zyz']

        angles = [[np.pi/3, np.pi/3, 0], [0, np.pi/3, np.pi/3],
                  [np.pi/3, np.pi/3, np.pi/3],
                  [-np.pi/3, -np.pi/3, 0], [0, -np.pi/3, -np.pi/3],
                  [-np.pi/3, -np.pi/3, -np.pi/3],
                  [1, 2, 3], [1, -2, 3],
                  [[1, 2, 3, 1], [2, 3, 1, 2], [3, 1, 2, 3]]]

        for angle in angles:

            for order in orders:

                with self.subTest(angle=angle, order=order):

                    rmat = at.euler_to_rotmat(angle, order=order)
                    quat = at.rotmat_to_quaternion(rmat)

                    euler = at.quaternion_to_euler(quat, order=order)

                    rmat2 = at.euler_to_rotmat(euler, order=order)
                    quat2 = at.rotmat_to_quaternion(rmat2)

                    np.testing.assert_almost_equal(quat, quat2)


class TestRotVecToRotMat(TestCase):

    def test_rotvec_to_rotmat(self):
        rotmat = at.rotvec_to_rotmat([0, 0, 0])

        np.testing.assert_array_almost_equal(rotmat, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        rotmat = at.rotvec_to_rotmat([[0, 0], [0, 0], [0, 0]])

        np.testing.assert_array_almost_equal(rotmat, [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                      [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        rotmat = at.rotvec_to_rotmat([np.pi, 0, 0])

        np.testing.assert_array_almost_equal(rotmat, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        rotmat = at.rotvec_to_rotmat([0, np.pi, 0])

        np.testing.assert_array_almost_equal(rotmat, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        rotmat = at.rotvec_to_rotmat([0, 0, np.pi])

        np.testing.assert_array_almost_equal(rotmat, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        rotmat = at.rotvec_to_rotmat([[np.pi, 0, 0], [0, np.pi, 0], [0, 0, -np.pi]])

        np.testing.assert_array_almost_equal(rotmat, [[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                                                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])

        rotmat = at.rotvec_to_rotmat([[np.pi / 2, 0], [0, -np.pi / 2], [0, 0]])

        np.testing.assert_array_almost_equal(rotmat, [[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                                      [[0, 0, -1], [0, 1, 0], [1, 0, 0]]])

        rotmat = at.rotvec_to_rotmat([[np.pi / 2, 0, 0], [0, 0, -np.pi / 2], [0, 0, 0]])

        np.testing.assert_array_almost_equal(rotmat, [[[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                                      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                      [[0, 0, -1], [0, 1, 0], [1, 0, 0]]])

        rotmat = at.rotvec_to_rotmat([1, 2, 3])

        np.testing.assert_array_almost_equal(rotmat, [[-0.69492056, 0.71352099, 0.08929286],
                                                      [-0.19200697, -0.30378504, 0.93319235],
                                                      [0.69297817, 0.6313497, 0.34810748]])


class TestRotVecToQuaternion(TestCase):
    
    def test_rotvec_to_quaternion(self):
        
        q = at.rotvec_to_quaternion([0, 0, 0])
        
        np.testing.assert_array_almost_equal(q, [0, 0, 0, 1])
        
        q = at.rotvec_to_quaternion([[0], [0], [0]])

        np.testing.assert_array_almost_equal(q, [[0], [0], [0], [1]])

        q = at.rotvec_to_quaternion([np.pi, 0, 0])

        np.testing.assert_array_almost_equal(q, [1, 0, 0, 0])
        
        q = at.rotvec_to_quaternion([0, np.pi, 0])

        np.testing.assert_array_almost_equal(q, [0, 1, 0, 0])
        
        q = at.rotvec_to_quaternion([0, 0, np.pi])

        np.testing.assert_array_almost_equal(q, [0, 0, 1, 0])

        q = at.rotvec_to_quaternion([1, 2, 3])

        np.testing.assert_array_almost_equal(q, [0.25532186,  0.51064372,  0.76596558, -0.29555113])
        
        q = at.rotvec_to_quaternion([[0], [0], [np.pi]])

        np.testing.assert_array_almost_equal(q, [[0], [0], [1], [0]])
        
        q = at.rotvec_to_quaternion([[np.pi, 0, 1, 0],
                                     [0, 0, 2, 0],
                                     [0, 0, 3, 0]])
        
        np.testing.assert_array_almost_equal(q, [[1, 0, 0.25532186, 0],  
                                                 [0, 0, 0.51064372, 0],  
                                                 [0, 0, 0.76596558, 0], 
                                                 [0, 1, -0.29555113, 1]])


class TestRotMatToQuaternion(TestCase):

    def test_rotmat_to_quaternion(self):

        q = at.rotmat_to_quaternion(np.eye(3))

        np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-16)

        # figure out how to account for the fact that these can be positive or negative
        q = at.rotmat_to_quaternion(np.array([[-1., 0, 0], [0, 1, 0], [0, 0, -1]]))

        np.testing.assert_allclose(np.abs(q), [0, 1, 0, 0], atol=1e-16)

        q = at.rotmat_to_quaternion(np.array([[1., 0, 0], [0, -1, 0], [0, 0, -1]]))

        np.testing.assert_allclose(np.abs(q), [1, 0, 0, 0], atol=1e-16)

        q = at.rotmat_to_quaternion(np.array([[-1., 0, 0], [0, -1, 0], [0, 0, 1]]))

        np.testing.assert_allclose(np.abs(q), [0, 0, 1, 0], atol=1e-16)

        q = at.rotmat_to_quaternion([np.eye(3)]*2)

        np.testing.assert_allclose(q.T, [[0, 0, 0, 1]]*2, atol=1e-16)

        q = at.rotmat_to_quaternion([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        np.testing.assert_allclose(q, [0, 0, -np.sqrt(2)/2, np.sqrt(2)/2], atol=1e-16)

        q = at.rotmat_to_quaternion([[-0.69492056, -0.19200697, 0.69297817],
                                     [0.71352099, -0.30378504, 0.6313497],
                                     [0.08929286, 0.93319235, 0.34810748]])

        np.testing.assert_allclose(q, [0.25532186, 0.51064372, 0.76596558, 0.29555113], atol=1e-16)

        q = at.rotmat_to_quaternion([[[-0.69492056, -0.19200697, 0.69297817],
                                      [0.71352099, -0.30378504, 0.6313497],
                                      [0.08929286, 0.93319235, 0.34810748]],
                                     np.eye(3)])

        np.testing.assert_allclose(q.T, [[0.25532186, 0.51064372, 0.76596558, 0.29555113],
                                         [0, 0, 0, 1]], atol=1e-16)

        with self.assertRaises(ValueError):

            at.rotmat_to_quaternion([1, 2, 3])

        with self.assertRaises(ValueError):
            at.rotmat_to_quaternion([[1, 2, 3]])

        with self.assertRaises(ValueError):
            at.rotmat_to_quaternion([[1], [2], [3]])

        with self.assertRaises(ValueError):
            at.rotmat_to_quaternion([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            at.rotmat_to_quaternion([[1, 2, 3, 4]])

        with self.assertRaises(ValueError):
            at.rotmat_to_quaternion([[1], [2], [3], [4]])


class TestRotMatToEuler(TestCase):

    def test_rotmat_to_euler(self):
        orders = ['xyz', 'zxy', 'yxz', 'yzx', 'xzy', 'zyx', 'xyx', 'yxy', 'xzx', 'zxz', 'yzy', 'zyz']

        angles = [[np.pi/3, np.pi/3, 0], [0, np.pi/3, np.pi/3],
                  [np.pi/3, np.pi/3, np.pi/3],
                  [-np.pi/3, -np.pi/3, 0], [0, -np.pi/3, -np.pi/3],
                  [-np.pi/3, -np.pi/3, -np.pi/3],
                  [1, 2, 3], [1, -2, 3],
                  [[1, 2, 3, 1], [2, 3, 1, 2], [3, 1, 2, 3]]]

        for angle in angles:

            for order in orders:

                with self.subTest(angle=angle, order=order):

                    rmat = at.euler_to_rotmat(angle, order=order)

                    euler = at.rotmat_to_euler(rmat, order=order)

                    rmat2 = at.euler_to_rotmat(euler, order=order)

                    np.testing.assert_almost_equal(rmat, rmat2)


class TestEulerToRotMat(TestCase):
    def test_euler_to_rotmat(self):
        orders = ['xyz', 'zxy', 'yxz', 'yzx', 'xzy', 'zyx', 'xyx', 'yxy', 'xzx', 'zxz', 'yzy', 'zyz']

        angles = [[np.pi/3, 0, 0], [0, np.pi/3, 0], [0, 0, np.pi/3],
                  [np.pi/3, np.pi/3, 0], [0, np.pi/3, np.pi/3],
                  [np.pi/3, np.pi/3, np.pi/3],
                  [-np.pi/3, -np.pi/3, 0], [0, -np.pi/3, -np.pi/3],
                  [-np.pi/3, -np.pi/3, -np.pi/3],
                  [1, 2, 3], [1, -2, 3],
                  [[1, 2, 3, 1], [2, 3, 1, 2], [3, 1, 2, 3]]]

        for angle in angles:

            for order in orders:

                with self.subTest(angle=angle, order=order):

                    rmat = at.euler_to_rotmat(angle, order=order)

                    rmat2 = np.eye(3)

                    for an, ax in zip(angle, order):

                        if ax.upper().lower() == 'x':
                            update = at.rot_x(an)
                        elif ax.upper().lower() == 'y':
                            update = at.rot_y(an)
                        elif ax.upper().lower() == 'z':
                            update = at.rot_z(an)

                        rmat2 = update @ rmat2

                    np.testing.assert_almost_equal(rmat, rmat2)


class TestRotX(TestCase):

    def test_rot_x(self):

        angles = [3*np.pi/2, np.pi, np.pi/2, np.pi/3, 0,
                  -3*np.pi/2, -np.pi, -np.pi/2, -np.pi/3,
                  [0, np.pi/2, np.pi/3],
                  [0, -np.pi/2, -np.pi/3]]

        mats = [[[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                [[1, 0, 0], [0, 0.5, -np.sqrt(3)/2], [0, np.sqrt(3)/2, 0.5]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                [[1, 0, 0], [0, 0.5, np.sqrt(3)/2], [0, -np.sqrt(3)/2, 0.5]],
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                 [[1, 0, 0], [0, 0.5, -np.sqrt(3)/2], [0, np.sqrt(3)/2, 0.5]]],
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                 [[1, 0, 0], [0, 0.5, np.sqrt(3)/2], [0, -np.sqrt(3)/2, 0.5]]]
                ]

        for angle, solu in zip(angles, mats):

            with self.subTest(angle=angle):

                rmat = at.rot_x(angle)

                np.testing.assert_almost_equal(rmat, solu)


class TestRotY(TestCase):

    def test_rot_y(self):

        angles = [3*np.pi/2, np.pi, np.pi/2, np.pi/3, 0,
                  -3*np.pi/2, -np.pi, -np.pi/2, -np.pi/3,
                  [0, np.pi/2, np.pi/3],
                  [0, -np.pi/2, -np.pi/3]]
        srt3d2 = np.sqrt(3)/2
        mats = [[[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                [[0.5, 0, srt3d2], [0, 1, 0], [-srt3d2, 0, 0.5]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                [[0.5, 0, -srt3d2], [0, 1, 0], [srt3d2, 0, 0.5]],
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                 [[0.5, 0, srt3d2], [0, 1, 0], [-srt3d2, 0, 0.5]]],
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                 [[0.5, 0, -srt3d2], [0, 1, 0], [srt3d2, 0, 0.5]]]
                ]

        for angle, solu in zip(angles, mats):

            with self.subTest(angle=angle):

                rmat = at.rot_y(angle)

                np.testing.assert_almost_equal(rmat, solu)


class TestRotZ(TestCase):

    def test_rot_z(self):

        angles = [3*np.pi/2, np.pi, np.pi/2, np.pi/3, 0,
                  -3*np.pi/2, -np.pi, -np.pi/2, -np.pi/3,
                  [0, np.pi/2, np.pi/3],
                  [0, -np.pi/2, -np.pi/3]]
        srt3d2 = np.sqrt(3)/2
        mats = [[[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                [[0.5, -srt3d2, 0], [srt3d2, 0.5, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
                [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                [[0.5, srt3d2, 0], [-srt3d2, 0.5, 0], [0, 0, 1]],
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                 [[0.5, -srt3d2, 0], [srt3d2, 0.5, 0], [0, 0, 1]]],
                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
                 [[0.5, srt3d2, 0], [-srt3d2, 0.5, 0], [0, 0, 1]]]
                ]

        for angle, solu in zip(angles, mats):

            with self.subTest(angle=angle):

                rmat = at.rot_z(angle)

                np.testing.assert_almost_equal(rmat, solu)


class TestSkew(TestCase):
    
    def test_skew(self):
        
        skew_mat = at.skew([1, 2, 3])
        
        np.testing.assert_array_equal(skew_mat, [[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        
        skew_mat = at.skew([[1, 2], [2, 3], [3, 4]])

        np.testing.assert_array_equal(skew_mat, [[[0, -3, 2], [3, 0, -1], [-2, 1, 0]],
                                                 [[0, -4, 3], [4, 0, -2], [-3, 2, 0]]])


class TestNLERP(TestCase):

    def test_nlerp(self):

        with self.subTest(input_type=list):
            q0 = [0, 0, 0, 1]
            q1 = [0.5, 0.5, 0.5, 0.5]

            qt = at.nlerp(q0, q1, 0)

            np.testing.assert_allclose(qt, q0)

            qt = at.nlerp(q0, q1, 1)

            np.testing.assert_allclose(qt, q1)

            qt = at.nlerp(q0, q1, 0.5)

            qtrue = (np.array(q0)+np.array(q1))/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.nlerp(q0, q1, 0.25)

            qtrue = np.array(q0)*(1-0.25)+np.array(q1)*0.25
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.nlerp(q0, q1, 0.79)

            qtrue = np.array(q0)*(1-0.79) + np.array(q1)*0.79
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            q0 = np.array([0.23, 0.45, 0.67, 0.2])
            q0 /= np.linalg.norm(q0)
            q1 = np.array([-0.3, 0.2, 0.6, 0.33])
            q1 /= np.linalg.norm(q1)

            qt = at.nlerp(q0, q1, 0)

            np.testing.assert_allclose(qt, q0)

            qt = at.nlerp(q0, q1, 1)

            np.testing.assert_allclose(qt, q1)

            qt = at.nlerp(q0, q1, 0.5)

            qtrue = (q0+q1)/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.nlerp(q0, q1, 0.25)

            qtrue = q0*(1-0.25)+q1*0.25
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.nlerp(q0, q1, 0.79)

            # comes from ODTBX matlab function
            qtrue = (1-0.79)*q0+0.79*q1
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

        with self.subTest(input_type=at.Rotation):
            q0 = at.Rotation([0, 0, 0, 1])
            q1 = at.Rotation([0.5, 0.5, 0.5, 0.5])

            qt = at.nlerp(q0, q1, 0)

            np.testing.assert_allclose(qt.q, q0.q)

            qt = at.nlerp(q0, q1, 1)

            np.testing.assert_allclose(qt.q, q1.q)

            qt = at.nlerp(q0, q1, 0.5)

            qtrue = (q0.q.flatten()+q1.q.flatten())/2  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.nlerp(q0, q1, 0.25)

            qtrue = (q0.q.flatten()*(1-0.25)+q1.q.flatten()*0.25)  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.nlerp(q0, q1, 0.79)

            qtrue = q0.q.flatten()*(1-0.79)+q1.q.flatten()*0.79  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            q0 = np.array([0.23, 0.45, 0.67, 0.2])
            q0 /= np.linalg.norm(q0)
            q0 = at.Rotation(q0)
            q1 = np.array([-0.3, 0.2, 0.6, 0.33])
            q1 /= np.linalg.norm(q1)
            q1 = at.Rotation(q1)

            qt = at.nlerp(q0, q1, 0)

            np.testing.assert_allclose(qt.q, q0.q)

            qt = at.nlerp(q0, q1, 1)

            np.testing.assert_allclose(qt.q, q1.q)

            qt = at.nlerp(q0, q1, 0.5)

            qtrue = (q0.q.flatten()+q1.q.flatten())/2  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.nlerp(q0, q1, 0.25)

            qtrue = q0.q.flatten()*(1-0.25)+q1.q.flatten()*0.25  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.nlerp(q0, q1, 0.79)

            qtrue = (1-0.79)*q0.q.flatten() + 0.79*q1.q.flatten()  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)
            
            
class TestSLERP(TestCase):
    
    def test_slerp(self):

        with self.subTest(input_type=list):
            q0 = [0, 0, 0, 1]
            q1 = [0.5, 0.5, 0.5, 0.5]

            qt = at.slerp(q0, q1, 0)

            np.testing.assert_allclose(qt, q0)

            qt = at.slerp(q0, q1, 1)

            np.testing.assert_allclose(qt, q1)

            qt = at.slerp(q0, q1, 0.5)

            qtrue = (np.array(q0)+np.array(q1))/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.slerp(q0, q1, 0.25)

            qtrue = (np.array(q0)+qtrue)/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.slerp(q0, q1, 0.79)

            # comes from ODTBX matlab function
            qtrue = [0.424985851398278, 0.424985851398278, 0.424985851398278, 0.676875969682661]

            np.testing.assert_allclose(qt, qtrue)

            q0 = np.array([0.23, 0.45, 0.67, 0.2])
            q0 /= np.linalg.norm(q0)
            q1 = np.array([-0.3, 0.2, 0.6, 0.33])
            q1 /= np.linalg.norm(q1)

            qt = at.slerp(q0, q1, 0)

            np.testing.assert_allclose(qt, q0)

            qt = at.slerp(q0, q1, 1)

            np.testing.assert_allclose(qt, q1)

            qt = at.slerp(q0, q1, 0.5)

            qtrue = (q0+q1)/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.slerp(q0, q1, 0.25)

            qtrue = (q0+qtrue)/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt, qtrue)

            qt = at.slerp(q0, q1, 0.79)

            # comes from ODTBX matlab function
            qtrue = [-0.256224563175732, 0.331694624881600, 0.813762532744541, 0.402639031082742]

            np.testing.assert_allclose(qt, qtrue)
            
        with self.subTest(input_type=at.Rotation):
            q0 = at.Rotation([0, 0, 0, 1])
            q1 = at.Rotation([0.5, 0.5, 0.5, 0.5])

            qt = at.slerp(q0, q1, 0)

            np.testing.assert_allclose(qt.q, q0.q)

            qt = at.slerp(q0, q1, 1)

            np.testing.assert_allclose(qt.q, q1.q)

            qt = at.slerp(q0, q1, 0.5)

            qtrue = (q0.q.flatten()+q1.q.flatten())/2  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.slerp(q0, q1, 0.25)

            qtrue = (q0.q.flatten()+qtrue)/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.slerp(q0, q1, 0.79)

            # comes from ODTBX matlab function
            qtrue = [0.424985851398278, 0.424985851398278, 0.424985851398278, 0.676875969682661]

            np.testing.assert_allclose(qt.q.flatten(), qtrue)
            
            q0 = np.array([0.23, 0.45, 0.67, 0.2])
            q0 /= np.linalg.norm(q0)
            q0 = at.Rotation(q0)
            q1 = np.array([-0.3, 0.2, 0.6, 0.33])
            q1 /= np.linalg.norm(q1)
            q1 = at.Rotation(q1)
            
            qt = at.slerp(q0, q1, 0)
            
            np.testing.assert_allclose(qt.q, q0.q)
            
            qt = at.slerp(q0, q1, 1)

            np.testing.assert_allclose(qt.q, q1.q)

            qt = at.slerp(q0, q1, 0.5)
            
            qtrue = (q0.q.flatten()+q1.q.flatten())/2  # type: np.ndarray
            qtrue /= np.linalg.norm(qtrue)
            
            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.slerp(q0, q1, 0.25)
            
            qtrue = (q0.q.flatten()+qtrue)/2
            qtrue /= np.linalg.norm(qtrue)

            np.testing.assert_allclose(qt.q.flatten(), qtrue)

            qt = at.slerp(q0, q1, 0.79)

            # comes from ODTBX matlab function
            qtrue = [-0.256224563175732, 0.331694624881600, 0.813762532744541, 0.402639031082742]
            
            np.testing.assert_allclose(qt.q.flatten(), qtrue)
