from dataclasses import dataclass

import numpy as np

from giant.stellar_opnav.estimators.attitude_estimator import AttitudeEstimator, AttitudeEstimatorOptions
from giant.rotations import Rotation
from giant._typing import DOUBLE_ARRAY

@dataclass
class ESOQ2Options(AttitudeEstimatorOptions):
    """
    Options for the ESOQ2 attitude estimator.
    
    :param n_iter: Number of iterations for lambda computation (default=10, use 0 for lam=1)
    """
    
    n_iter: int = 10
    """
    Number of iterations for lambda computation in ESOQ2 algorithm.
    """ 


class ESOQ2(AttitudeEstimator, ESOQ2Options):
    """
    Implements the ESOQ2 (Second Estimator of the Optimal Quaternion) solution to Wahba's problem.
    
    This is a faster technique than Davenport's Q-Method solution but slightly less accurate.
    
    This class is relatively easy to use.  After you initialize the class, simply specify the
    `target_frame_directions` unit vectors (:math:`\textbf{a}_i` from the :mod:`~.stellar_opnav.estimators`
    documentation) as a 3xn array of vectors (each column is a vector) and the `base_frame_directions` unit
    vectors (:math:`\textbf{b}_i` from the :mod:`~.stellar_opnav.estimators` documentation) as a 3xn array of
    vectors  (each column is a vector) in the call to :meth:`.estimate`.  Here the `target_frame_directions` 
    unit vectors are expressed in the end frame (the frame you want to rotate to) and the 
    `base_frame_directions` unit vectors are expressed in the starting frame (the frame you want to rotate 
    from).  Each column of `target_frame_directions` and `base_frame_directions` should
    correspond to each other as a pair (i.e. column 1 in `target_frame_directions` is paired with column ` in
    `base_frame_directions`.

    Optionally, you can set the :attr:`weighted_estimation` value to `True` and then provide the
    `weights` input to :meth:`estimate` to specify whether to use weighted estimation or not, and what weights to 
    use if you are using weighted estimation.  The `weights` input should be a length n array of the weights to 
    apply to each unit vector pair.

    The :meth:`estimate` method can be called to compute the attitude quaternion that best aligns the two frames.  
    When the :meth:`estimate` method completes, the solved for rotation is returned as an :class:`.Rotation` object.  
    In addition, the formal post fit covariance matrix of the estimate can be found in the :attr:`post_fit_covariance` 
    attribute.  Note that as with all attitude quaternions, the post fit covariance matrix will be rank deficient 
    since there are only 3 true degrees of freedom.
    """
    
    def __init__(self, options: ESOQ2Options | None = None) -> None:
        """
        :param options: the options dataclass to use to configure this class
        """
        
        super().__init__(ESOQ2Options, options=options)
        
        self._attitude_prof_mat = None
        """
        The attitude profile matrix.  
        
        This is used internally and is only set after a call to :meth:`estimate`.  It is stored for reuse in the 
        computation of the post fit covariance matrix.
        """
        
        self._solved_rotation: Rotation | None = None
        """
        the solved for rotation from the base to the target frame.
        
        This gets computed in the call to estimate.  Users shouldn't use it, rely on the retunr from estimate instead.
        """
        
        
    def estimate(self, target_frame_directions: DOUBLE_ARRAY, base_frame_directions: DOUBLE_ARRAY, weights: DOUBLE_ARRAY | None = None) -> Rotation:
        """
        This function uses the ESOQ2 algorithm to determine the optimal attitude quaternion
        given sets of observed and reference unit vectors. It is based on Daniele Mortari's work.
        
        
        :param target_frame_directions: Matrix of observed unit vectors (3xN)
        :param base_frame_directions: Matrix of reference unit vectors (3xN)
        :param weights: Vector of weights for each observation (N,) or None
        :param n_iter: Number of iterations for lambda computation (default=5, use 0 for lam=1)
        
        :returns: Near optimal attitude quaternion 
            
        .. Note::
            This implementation is based on "Second Estimator of the Optimal Quaternion" by Daniele Mortari,
            Journal of Guidance, Control, and Dynamics, Vol.23, No. 5, Sep-Oct 2000, pg 885-888.
        """
        # Second Estimator of the Optimal Quaternion (Daniele Mortari, 01/05/2001)
        # Function takes two matrices of vectors and determines the optimal quaternion
        # in a least squares sense
        
        if self.weighted_estimation:
            assert weights is not None, "Weights miust not be None if doing weighted estimation"
            target_frame_directions = weights.squeeze() * target_frame_directions

        bmat = self.attitude_profile_matrix(base_frame_directions, target_frame_directions)
        self._attitude_prof_mat  = bmat.copy()
        
        # Optimal 180 deg rotation
        b_trace = np.trace(bmat)
        b_diag = np.array([bmat[0,0], bmat[1,1], bmat[2,2], b_trace])
        
        irot = np.argmin(b_diag)
        b_min = b_diag[irot]
        
        if irot < 3:
            for i in range(3):
                if i != irot:
                    bmat[:, i] *= -1.0
            b_trace = 2.0 * b_min - b_trace
        
        # Compute S matrix and z vector
        smat = bmat + bmat.T
        zvec = np.array([bmat[1,2] - bmat[2,1], 
                        bmat[2,0] - bmat[0,2], 
                        bmat[0,1] - bmat[1,0]])
        zvec2 = zvec**2
        
        # Lambda max computation
        if self.n_iter == 0:
            lam = 1.0
        else:
            tadj = (smat[1,1] * smat[2,2] - smat[1,2]**2 + 
                    smat[0,0] * smat[2,2] - smat[0,2]**2 + 
                    smat[1,1] * smat[0,0] - smat[0,1]**2)
            trB2 = b_trace**2
            aa = trB2 - tadj
            bb = trB2 + np.sum(zvec2)
            
            szvec = smat @ zvec
            c2 = -aa - bb
            
            if target_frame_directions.shape[1] == 2:
                u = 2.0 * np.sqrt(aa * bb - np.sum(szvec**2))
                lam = (np.sqrt(u - c2) + np.sqrt(-u - c2)) / 2.0
            else:
                c1 = (-(smat[0,0] * smat[1,1] * smat[2,2] + 
                    2 * smat[0,1] * smat[1,2] * smat[0,2] -
                    smat[0,0] * smat[1,2]**2 - 
                    smat[1,1] * smat[0,2]**2 - 
                    smat[2,2] * smat[0,1]**2) - 
                    np.dot(zvec, szvec))
                c0 = aa * bb - c1 * b_trace - np.sum(szvec**2)
                lam = 1.0
                
                for k in range(self.n_iter):
                    lam2 = lam**2
                    lam = (lam2 * (3.0 * lam2 + c2) - c0) / (2.0 * lam * (2.0 * lam2 + c2) + c1)
        
        tpl = b_trace + lam
        smat[np.diag_indices(3)] -= tpl
        tml = b_trace - lam
        
        # Euler axis computation
        mvec = np.array([tml * smat[0,0] - zvec2[0],
                        tml * smat[1,1] - zvec2[1], 
                        tml * smat[2,2] - zvec2[2]])
        mxvec = np.array([tml * smat[0,1] - zvec[0] * zvec[1],
                        tml * smat[0,2] - zvec[0] * zvec[2],
                        tml * smat[1,2] - zvec[1] * zvec[2]])
        
        evec = np.array([mvec[1] * mvec[2] - mxvec[2]**2,
                        mvec[0] * mvec[2] - mxvec[1]**2,
                        mvec[0] * mvec[1] - mxvec[0]**2])
        
        k = np.argmax(np.abs(evec))
        
        if k == 0:
            evec[1] = mxvec[1] * mxvec[2] - mxvec[0] * mvec[2]
            evec[2] = mxvec[0] * mxvec[2] - mxvec[1] * mvec[1]
        elif k == 1:
            evec[0] = mxvec[1] * mxvec[2] - mxvec[0] * mvec[2]
            evec[2] = mxvec[0] * mxvec[1] - mvec[0] * mxvec[2]
        elif k == 2:
            evec[0] = mxvec[0] * mxvec[2] - mxvec[1] * mvec[1]
            evec[1] = mxvec[0] * mxvec[1] - mvec[0] * mxvec[2]
        
        # Quaternion computation in rotated frame
        quat = np.zeros(4)
        quat[:3] = tml * evec
        quat[3] = -np.dot(zvec, evec)
        
        # Normalize
        quat = quat / np.linalg.norm(quat)
        
        # Undo rotation to get quaternion in input frame
        if irot == 0:
            quat = np.array([quat[3], -quat[2], quat[1], -quat[0]])
        elif irot == 1:
            quat = np.array([quat[2], quat[3], -quat[0], -quat[1]])
        elif irot == 2:
            quat = np.array([-quat[1], quat[0], quat[3], -quat[2]])
        # irot == 3: no change needed
        
        # Ensure positive scalar part
        quat = quat * np.copysign(1.0, quat[3])
        
        self._solved_rotation = Rotation(quat)
        
        return self._solved_rotation.copy()
    
    @property
    def post_fit_covariance(self) -> DOUBLE_ARRAY:
        """
        This returns the post-fit covariance after calling the ``estimate`` method as a 4x4 numpy array.

        This should be only be called after the estimate method has been called, otherwise it raises a ValueError
        
        Note that this uses the same covariance as Davenport's Q-Method solution, which is technically the lower bound.
        """

        if self._attitude_prof_mat is None or self._solved_rotation is None:
            raise ValueError('estimate method must be called before requesting the post-fit covariance')

        # compute rotation matrix times attitude profile matrix transposed
        tbt = self._solved_rotation.matrix@self._attitude_prof_mat.T

        # compute the fisher information matrix
        ftt = np.trace(tbt)*np.eye(3) - (tbt+tbt.T)/2

        # form covariance matrix
        cov = np.zeros((4, 4), dtype=np.float64)
        cov[:3, :3] = np.linalg.inv(ftt)/4

        return cov
    
