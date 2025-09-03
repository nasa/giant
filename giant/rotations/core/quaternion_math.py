import numpy as np  

from giant._typing import ARRAY_LIKE, DOUBLE_ARRAY, DatetimeLike

from giant.rotations.core._helpers import _check_quaternion_array_and_shape

__all__ = ["quaternion_normalize", "quaternion_inverse", "quaternion_multiplication", "nlerp", "slerp"]


def quaternion_normalize(quaternion: ARRAY_LIKE) -> DOUBLE_ARRAY:
    """
    Normalizes the quaternion(s) such that the scalar term is positive and the length is 1
    
    :param quaternion: the quaternion(s) to normalize
    
    :returns: The normalized quaternions
    """
    
    work_quaternion = _check_quaternion_array_and_shape(quaternion)
    
    signs = np.sign(work_quaternion[-1])
    
    if np.shape(signs):
        signs[signs == 0] = 1
    else:
        signs = signs if signs != 0 else 1
        
    work_quaternion *= signs/np.linalg.norm(work_quaternion, axis=0, keepdims=True)
    
    return work_quaternion


def quaternion_inverse(quaternion: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function provides the inverse of a rotation quaternion of the form discussed in
    :ref:`Rotation Representations <rotation-representation-table>`.

    The inverse of a rotation quaternion is defined such that
    :math:`\mathbf{q}\otimes\mathbf{q}^{-1}=\mathbf{q}_I` where
    :math:`\mathbf{q}_I=\left[\begin{array}{cccc}0&0&0&1\end{array}\right]^T` is the identity quaternion which
    corresponds to the identity matrix (or no rotation) and :math:`\otimes` indicates quaternion multiplication.
    Mathematically this corresponds to negating the vector portion of the quaternion:

    .. math::
        \mathbf{q}=\left[\begin{array}{c}\text{sin}(\frac{\theta}{2})\hat{\mathbf{x}}\\
        \text{cos}(\frac{\theta}{2})\end{array}\right]\\
        \mathbf{q}^{-1}=\left[\begin{array}{c}-\text{sin}(\frac{\theta}{2})\hat{\mathbf{x}}\\
        \text{cos}(\frac{\theta}{2})\end{array}\right]

    This function is also vectorized, meaning that you can specify multiple rotation quaternions to be inversed by
    specifying each quaternion as a column.  Regardless of whether you are converting 1 or many
    quaternions the first axis must have a length of 4.

    This function makes the output have the same number of dimensions as the input.  Therefore, if the input is one
    dimensional, then the output is one dimensional, and if the input is two dimensional then the output will be two
    dimensional. 

    :param quaternion: The rotation quaternion(s) to be inverted
    :return: a numpy array representing the inverse quaternion corresponding to the input quaternion
    """
    
    # ensure the value is an array and break mutability
    quaternion = _check_quaternion_array_and_shape(quaternion, return_copy=True)

    # negate the vector portion
    quaternion[:3] *= -1
    
    # return the inverse quaternion(s) as an array
    return quaternion


def quaternion_multiplication(quaternion_1_in: ARRAY_LIKE,
                              quaternion_2_in: ARRAY_LIKE) -> DOUBLE_ARRAY:
    r"""
    This function performs the hamiltonian quaternion multiplication operation.

    The quaternions should be of the form as specified in
    :ref:`Rotation Representations <rotation-representation-table>`.

    The hamiltonian multiplication is defined such that
    `q_from_A_to_C = quaternion_multiplication(q_from_B_to_C, q_from_A_to_B)`

    Mathematically this is given by:

    .. math::
        \mathbf{q}_1\otimes\mathbf{q}_2=\left[\begin{array}{c}q_{s1}\mathbf{q}_{v2} + q_{s2}\mathbf{q}_{v1} +
        \mathbf{q}_{v1}\times\mathbf{q}_{v2}\\
        q_{s1}q_{s2}-\mathbf{q}_{v1}^T\mathbf{q}_{v2}\end{array}\right]

    This function is vectorized, therefore you can input multiple quaternions as a 4xn array where each column is an
    independent quaternion.

    :param quaternion_1_in: The first quaternion to multiply
    :param quaternion_2_in: The second quaternion to multiply
    :return: The non-hamiltonian product of quaternion_1 and quaternion_2
    """

    rquat = False

    quaternion_1 = _check_quaternion_array_and_shape(quaternion_1_in)
    quaternion_2 = _check_quaternion_array_and_shape(quaternion_2_in)

    qs1 = quaternion_1[-1]
    qv1 = quaternion_1[0:3]

    qs2 = quaternion_2[-1]
    qv2 = quaternion_2[0:3]

    qout = np.concatenate([qs1 * qv2 + qs2 * qv1 + np.cross(qv1, qv2, axis=0),
                           [qs1 * qs2 - (qv1 * qv2).sum(axis=0)]], axis=0)

    return qout


def nlerp(quaternion0: ARRAY_LIKE, quaternion1: ARRAY_LIKE,
          time: float | DatetimeLike,
          time0: float | DatetimeLike = 0, time1: float | DatetimeLike = 1) -> DOUBLE_ARRAY:
    r"""
    This function performs normalized linear interpolation of rotation quaternions.

    NLERP of quaternions involves first performing a linear interpolation between the two vectors, and then normalizing
    the interpolated result to have unit length.  That is:

    .. math::
        \mathbf{q}=\frac{\mathbf{q}_0(1-p)+\mathbf{q}_1p}
        {\left\|\mathbf{q}_0(1-p)+\mathbf{q}_1p\right\|}

    where :math:`\mathbf{q}` is the interpolated quaternion, :math:`\mathbf{q}_0` is the starting quaternion,
    :math:`\mathbf{q}_1` is the ending quaternion, and :math:`p` is the fractional percent of the way between
    :math:`\mathbf{q}_0` and :math:`\mathbf{q}_1` that we want to interpolate at (:math:`p\in[0, 1]`)

    When using this function you can either specify the argument `time` as the fractional percent that you want to
    interpolate at, or specify the keyword arguments `time0` and `time1` to be the times corresponding to the first and
    second quaternion respectively and the function will compute the fractional percent for you.  When using this method
    it is also possible to specify all three of `time`, `time0`, and `time1` as python datetime objects.

    .. warning::
        NLERP is a very fast and efficient interpolation method that is fine for short interpolation intervals; however,
        it does not perform a constant angular velocity interpolation (and instead performs a constant linear velocity
        interpolation), therefore it is not well suited to interpolating over long time intervals. If you need to
        interpolate over larger time intervals it is better to use the :func:`slerp` function which does perform
        constant angular velocity interpolation (but is less efficient).

    :param quaternion0: The starting quaternion(s)
    :param quaternion1: The ending quaternion(s)
    :param time: The time to interpolate the quaternions at, as a fractional percent or as the actual time between
                `time0` and `time1`
    :param time0: the time(s) corresponding to the first quaternion(s). Leave at 0 if you are specifying `time` as a
                  fractional percent
    :param time1: the time(s) corresponding to the second quaternion(s). Leave at 1 if you are specifying `time` as a
                  fractional percent
    :return: The interpolated quaternion(s)
    """
    

    # compute the fractional percent we are interpolating at
    try:
        dt = float((time - time0) / (time1 - time0)) # type: ignore
    except TypeError:
        raise TypeError('time, time0, and time1 must support subtraction resulting in a type that supports true division.'
                        'Typically this means they should all be floats or all be DatetimeLike objects')

    rquat = False

    # extract the quaternion values as arrays 
    q0 = _check_quaternion_array_and_shape(quaternion0)
    q1 = _check_quaternion_array_and_shape(quaternion1)

    # perform the linear interpolation
    q = q0 * (1 - dt) + q1 * dt

    # perform the normalization
    q /= np.linalg.norm(q, axis=0, keepdims=True)

    # return the interpolated quaternion(s)
    return q


def slerp(quaternion0: ARRAY_LIKE, quaternion1: ARRAY_LIKE,
          time: float | DatetimeLike,
          time0: float | DatetimeLike = 0, time1: float | DatetimeLike = 1) -> DOUBLE_ARRAY:
    r"""
    This function performs spherical linear interpolation of rotation quaternions.

    SLERP of quaternions involves performing a linear interpolation along the great circle arc connecting the two
    quaternions. That is:

    .. math::
        \omega = \text{cos}^{-1}(\mathbf{q}_0^T\mathbf{q}_1)\\
        \mathbf{q}=\mathbf{q}_0\text{cos}(p\omega)+
        \text{sin}(p\omega)\frac{\mathbf{q}_1-\mathbf{q}_0\text{cos}(\omega)}
        {\left\|\mathbf{q}_1-\mathbf{q}_0\text{cos}(\omega)\right\|}\\
        \mathbf{q} = \frac{\mathbf{q}}{\left\|\mathbf{q}\right\|}

    where :math:`\mathbf{q}` is the interpolated quaternion, :math:`\mathbf{q}_0` is the starting quaternion,
    :math:`\mathbf{q}_1` is the ending quaternion, :math:`\omega` is the angle between the first and second quaternion,
    and :math:`p` is the fractional percent of the way between
    :math:`\mathbf{q}_0` and :math:`\mathbf{q}_1` that we want to interpolate at (:math:`p\in[0, 1]`)

    When using this function you can either specify the argument `time` as the fractional percent that you want to
    interpolate at, or specify the keyword arguments `time0` and `time1` to be the times corresponding to the first and
    second quaternion respectively and the function will compute the fractional percent for you.  When using this method
    it is also possible to specify all three of `time`, `time0`, and `time1` as python datetime objects.

    :param quaternion0: The starting quaternion(s)
    :param quaternion1: The ending quaternion(s)
    :param time: The time to interpolate the quaternions at, as a fractional percent or as the actual time between
                `time0` and `time1`
    :param time0: the time(s) corresponding to the first quaternion(s). Leave at 0 if you are specifying `time` as a
                  fractional percent
    :param time1: the time(s) corresponding to the second quaternion(s). Leave at 1 if you are specifying `time` as a
                  fractional percent
    :return: The interpolated quaternion(s)
    """

    # compute the fractional percent we are interpolating at
    try:
        dt = float((time - time0) / (time1 - time0)) # type: ignore
    except TypeError:
        raise TypeError('time, time0, and time1 must support subtraction resulting in a type that supports true division.'
                        'Typically this means they should all be floats or all be DatetimeLike objects')

    # extract the quaternion values as arrays 
    q0 = _check_quaternion_array_and_shape(quaternion0)
    q1 = _check_quaternion_array_and_shape(quaternion1)

    # enforce unit normalization
    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)

    # get the cosine of the angle between the quaternions
    cos_angle = np.inner(q0, q1)

    if cos_angle > 0.9995:
        # if the quaternions are really close revert to nlerp
        return nlerp(q0, q1, dt)

    elif cos_angle < 0:
        # if the dot product is negative negate the second quaternion to ensure the shorter path is taken
        q1 *= -1
        cos_angle *= -1

    cos_angle = np.clip(cos_angle, -1, 1)  # ensure the domain for acos (only will leave due to numerical issues)

    angle0 = np.arccos(cos_angle)  # angle between q0 and q1
    angle = angle0 * dt  # angle between q0 and q

    # form an orthonormal basis
    qb = q1 - q0 * cos_angle

    qb /= np.linalg.norm(qb)

    # perform the interpolation
    q = q0 * np.cos(angle) + qb * np.sin(angle)
    q /= np.linalg.norm(q)

    return q
