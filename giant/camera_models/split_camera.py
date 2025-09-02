"""
This module provides the SplitCamera class, which allows for combining two different camera models into a single model.

The SplitCamera class is useful when dealing with cameras that have different properties or behaviors in different
regions of the image or camera frame. It allows you to specify two different camera models and define how to split
between them based on either the camera frame coordinates or the image plane coordinates.

Use
---

To use the SplitCamera class, you need to provide two camera models and specify how to split between them:

    >>> from giant.camera_models import PinholeModel, SplitCamera
    >>> model1 = PinholeModel(...)
    >>> model2 = PinholeModel(...)
    >>> split_camera = SplitCamera(model1, model2, camera_frame_split_axis=CameraFrameSplitAxis.X,
    ...                            camera_frame_split_threshold=0.0)

The SplitCamera will then use model1 for points where the X coordinate in the camera frame is less than or equal to 0.0,
and model2 for points where the X coordinate is greater than 0.0.
"""

from importlib import import_module

from enum import IntEnum

from typing import Callable, Sequence, Literal, cast, Self
import lxml.etree as etree  # nosec

import numpy as np
from numpy.typing import NDArray, DTypeLike

from giant.camera_models.camera_model import CameraModel
from giant._typing import ARRAY_LIKE, F_SCALAR_OR_ARRAY, NONENUM, F_ARRAY_LIKE, DOUBLE_ARRAY


class CameraFrameSplitAxis(IntEnum):
    """
    An enumeration specifying the axis to use for splitting in the camera frame.

    This enum is used to define which axis in the camera frame should be used to determine
    which of the two camera models to use for a given point.
    """
    
    X = 0
    """Split along the X-axis of the camera frame"""

    Y = 1
    """Split along the Y-axis of the camera frame"""

    Z = 2
    """Split along the Z-axis of the camera frame"""
    

class ImagePlaneSplitAxis(IntEnum):
    """
    An enumeration specifying the axis to use for splitting in the image plane.

    This enum is used to define which axis in the image plane should be used to determine
    which of the two camera models to use for a given pixel location.
    """

    X = 0
    """Split along the X-axis (columns) of the image plane"""

    Y = 1
    """Split along the Y-axis (rows) of the image plane"""



class SplitCamera(CameraModel):
    """
    A camera model that combines two different camera models for different regions of the detector.

    This class allows for the creation of a hybrid camera model that uses two different camera models for different
    regions of the detector. The split is defined in both the camera frame (using
    camera_frame_split_axis and camera_frame_split_threshold) and in the image plane (using image_plane_split_axis
    and image_plane_split_threshold).

    The SplitCamera inherits from the CameraModel base class and implements all required methods, delegating to
    either model1 or model2 based on the defined split.
    """
    
    def __init__(self, model1: CameraModel, model2: CameraModel, 
                 camera_frame_split_axis: CameraFrameSplitAxis = CameraFrameSplitAxis.X, camera_frame_split_threshold: float = 0.0, 
                 image_plane_split_axis: ImagePlaneSplitAxis = ImagePlaneSplitAxis.X, image_plane_split_threshold: float = 0.0,
                 field_of_view: NONENUM = 0.0, n_rows: int = 1, n_cols: int = 1, use_a_priori: bool = False ):
        """
        Initialize a new SplitCamera instance.

        :param model1: The first camera model to use
        :param model2: The second camera model to use
        :param camera_frame_split_axis: The axis in the camera frame to use for splitting
        :param camera_frame_split_threshold: The threshold value for splitting in the camera frame
        :param image_plane_split_axis: The axis in the image plane to use for splitting
        :param image_plane_split_threshold: The threshold value for splitting in the image plane
        :param field_of_view: The field of view of the camera in degrees
        :param n_rows: The number of rows in the image
        :param n_cols: The number of columns in the image
        :param use_a_priori: Whether to use a priori information in calibration
        """
        
        super().__init__(field_of_view, n_rows, n_cols, use_a_priori)
        self._model1 = model1
        self._model2 = model2
        self._camera_frame_split_axis: int = int(camera_frame_split_axis)
        self._camera_frame_split_threshold = camera_frame_split_threshold
        self._image_plane_split_axis: int = int(image_plane_split_axis)
        self._image_plane_split_threshold = image_plane_split_threshold
        
        self.important_attributes.extend(['camera_frame_split_axis', 'camera_frame_split_threshold', 'image_plane_split_axis', 'image_plane_split_threshold'])
        
    @property
    def model1(self) -> CameraModel:
        """
        Get the first camera model.

        :return: The first camera model used in the split camera
        """
        return self._model1
    
    @property
    def model2(self) -> CameraModel:
        """
        Get the second camera model.

        :return: The second camera model used in the split camera
        """
        return self._model2
    
    @property 
    def camera_frame_split_axis(self) -> CameraFrameSplitAxis:
        """
        Get the axis used for splitting in the camera frame.

        :return: The CameraFrameSplitAxis enum value representing the split axis in the camera frame
        """
        return CameraFrameSplitAxis(self._camera_frame_split_axis)
    
    @camera_frame_split_axis.setter
    def camera_frame_split_axis(self, val: CameraFrameSplitAxis):
        """
        Set the axis used for splitting in the camera frame.

        :param val: The CameraFrameSplitAxis enum value representing the new split axis in the camera frame
        :raises ValueError: If the provided value is not a valid CameraFrameSplitAxis
        """
        # verify its a valid axis
        CameraFrameSplitAxis(val)
        self._camera_frame_split_axis = int(val)    
        
    @property 
    def image_plane_split_axis(self) -> ImagePlaneSplitAxis:
        """
        Get the axis used for splitting in the image plane.

        :return: The ImagePlaneSplitAxis enum value representing the split axis in the image plane
        """
        return ImagePlaneSplitAxis(self._image_plane_split_axis)
    
    @image_plane_split_axis.setter
    def image_plane_split_axis(self, val: ImagePlaneSplitAxis):
        """
        Set the axis used for splitting in the image plane.

        :param val: The ImagePlaneSplitAxis enum value representing the new split axis in the image plane
        :raises ValueError: If the provided value is not a valid ImagePlaneSplitAxis
        """
        # verify its a valid axis
        ImagePlaneSplitAxis(val)
        self._image_plane_split_axis = int(val)    
    
    @property 
    def estimation_parameters(self) -> list[str]:
        """
        Get the list of parameters to be estimated during calibration.

        This property combines the estimation parameters from both component camera models,
        prefixing them with 'm1_' and 'm2_' to distinguish between the two models.

        :return: A list of strings representing the parameters to be estimated
        """
        return [f"m1_{p}" for p in self._model1.estimation_parameters] + [f"m2_{p}" for p in self._model2.estimation_parameters]
    
    @estimation_parameters.setter
    def estimation_parameters(self, val: str | Sequence[str]):
        """
        Set the list of parameters to be estimated during calibration.

        This method splits the provided list into two separate lists for each component camera model,
        removing the 'm1_' and 'm2_' prefixes.
        
        Anything that doesn't have a m1_ or m2_ prefix will be applied to both models

        :param val: A list of strings representing the parameters to be estimated
        """
        if isinstance(val, str):
            val = [val]
        common = [p for p in val if not (p.startswith("m1_") or p.startswith("m2_"))]
        m1_params = [p[2:] for p in val if p.startswith("m1_")] + common
        m2_params = [p[2:] for p in val if p.startswith("m2_")] + common
        
        self._model1.estimation_parameters = m1_params
        self._model2.estimation_parameters = m2_params
    
    @property 
    def state_vector(self) -> list[float]:
        """
        Get the current state vector of the camera model.

        This property combines the state vectors from both component camera models.

        :return: A list of float numbers representing the current state of the camera model
        """
        return self._model1.state_vector+self._model2.state_vector
    
    def get_state_labels(self) -> list[str]:
        """
        Convert a list of estimation parameters into state label names.

        This method interprets the list of estimation parameters (:attr:`estimation_parameters`) into human readable
        state labels for pretty printing calibration results and for knowing the order of the state vector.
        It combines the state labels from both component camera models, prefixing them with 'm1_' and 'm2_' respectively.

        :return: The list of state names corresponding to estimation parameters in order
        """
        return [f"m1_{label}" for label in self._model1.get_state_labels()] + [f"m2_{label}" for label in self._model2.get_state_labels()]
    
    def _apply_split_logic(self, points: ARRAY_LIKE, method1: Callable, method2: Callable, *args, 
                           output_leading_shape: tuple[int,...] = (2,), output_trailing_shape: tuple[int,...] = (), 
                           frame: Literal['cf', 'if'] = 'cf', output_dtype: DTypeLike = np.float64) -> np.ndarray:
        """
        Apply the split logic to determine which component model to use for a given set of points.

        This internal method is the core of the SplitCamera's functionality. It decides whether to use model1 or model2
        based on the defined split axis and threshold, either in the camera frame or image plane. It then applies the
        appropriate method (method1 or method2) to the points that fall within each model's domain.

        :param points: The input points to be processed, either in the camera frame or image plane
        :param method1: The method to apply for points that fall within model1's domain
        :param method2: The method to apply for points that fall within model2's domain
        :param args: Additional arguments to be passed to method1 and method2
        :param output_leading_shape: The expected leading shape of the output array
        :param output_trailing_shape: The expected trailing shape of the output array
        :param frame: Specifies whether the points are in the camera frame ('cf') or image plane ('if')
        :param output_dtype: The desired data type of the output array
        :return: An array containing the results of applying method1 or method2 to the appropriate points
        """
        
        if frame == "cf":
            sa = self._camera_frame_split_axis
            st = self._camera_frame_split_threshold
        else:
            sa = self._image_plane_split_axis
            st = self._image_plane_split_threshold
            
        points = np.asanyarray(points)
        if points.ndim == 2:
            m1_check = points[sa] <= st
            m2_check = ~m1_check
            
            m1_points = points[:, m1_check]
            m2_points = points[:, m2_check]
            
            out = np.zeros(output_leading_shape + (points.shape[1],) + output_trailing_shape, output_dtype)
            
            leading_slices = [slice(None)] * len(output_leading_shape)
            
            if m1_points.size:
                out[*leading_slices, m1_check] = method1(m1_points, *args)
            if m2_points.size:
                out[*leading_slices, m2_check] = method2(m2_points, *args)
            
            return out
        
        elif points[sa] <= st:
            return method1(points, *args)
        
        return method2(points, *args)
    
    def project_onto_image(self, points_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) -> np.ndarray:
        """
        This method transforms 3D points (or directions) expressed in the camera frame into the corresponding 2D image
        locations.
        
        The points input should be either 1 or 2 dimensional, with the first axis being length 3 (each point 
        (direction) in the camera frame is specified as a column).
        
        The optional ``image`` key word argument specifies the index of the image you are projecting onto (this only 
        applies if you have a separate misalignment for each image)

        The optional ``temperature`` key word argument specifies the temperature to use when projecting the points into
        the image. This only applies when your focal length has a temperature dependence
        
        :param points_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature of the camera to use for the projection
        :return: A shape (2,) or shape (2, n) numpy array of image points (with units of pixels)
        """
        return self._apply_split_logic(points_in_camera_frame, self._model1.project_onto_image, self._model2.project_onto_image, image, temperature)
    
    def project_directions(self, directions_in_camera_frame: ARRAY_LIKE, image: int = 0) -> np.ndarray:
        """
        This method transforms 3D directions expressed in the camera frame into the corresponding 2D image
        directions.

        The direction input should be either 1 or 2 dimensional, with the first axis being length 3 (each direction
        in the camera frame is specified as a column).

        The optional ``image`` key word argument specifies the index of the image you are projecting onto (this only
        applies if you have a separate misalignment for each image)

        This method is different from method :meth:`project_onto_image` in that it only projects the direction component
        perpendicular to the optical axis of the camera (x, y axes of the camera frame) into a unit vector in the image
        plane. Therefore, you do not get a location in the image out of this, rather a unitless direction in the image.

        :param directions_in_camera_frame: a shape (3,) or shape (3, n) array of points to project
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :return: A shape (2,) or shape (2, n) numpy array of image direction unit vectors
        """
        return self._apply_split_logic(directions_in_camera_frame, self._model1.project_directions, self._model2.project_directions, image)
    
    def compute_jacobian(self, unit_vectors_in_camera_frame: Sequence[DOUBLE_ARRAY | list[list]], 
                         temperature: F_SCALAR_OR_ARRAY | Sequence[float] = 0) -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_P/\partial\mathbf{c}` where
        :math:`\mathbf{c}` is a vector of camera model parameters.
        
        The vector of camera model parameters contains things like the focal length, the pixel pitch, the distortion
        coefficients, and a misalignment vector. The ``unit_vectors_in_camera_frame`` should be a shape (m, 3, n) array
        of unit vectors expressed in the camera frame that you wish to calculate the Jacobian for where m is the number
        of images being calibrated. (These unit vectors should correspond to the pixel locations of the measurements
        when projected through the model).
        
        In general this method will not be used by the user and instead is used internally by the calibration estimators
        in :mod:`.calibration`.
        
        :param unit_vectors_in_camera_frame: A (m, 3, n) array of unit vectors expressed in the camera frame
        :param temperature: The temperature of the camera to use for computing the Jacobian matrix.
                            If temperature is an array it must be the same length as the first axis of the
                            ``unit_vectors_in_camera_frame`` input.
        :return: A (n*2, o) (where o is the length of :math:`\mathbf{c}`) array containing the Jacobian matrix
        """
            
        model_1_vectors = []
        model_2_vectors = []
        model_1_checks = []
        model_2_checks = []
        for vectors in unit_vectors_in_camera_frame:
            vectors = np.asarray(vectors)
            if vectors.size:
                m1_check = vectors[self._camera_frame_split_axis] <= self._camera_frame_split_threshold
                model_1_checks.append(m1_check)
                m2_check = ~m1_check
                model_2_checks.append(m2_check)
                model_1_vectors.append(vectors[:, m1_check])
                model_2_vectors.append(vectors[:, m2_check])
            else:
                model_1_vectors.append([[], []])
                model_2_vectors.append([[], []])
                model_1_checks.append([])
                model_2_checks.append([])
        
        m1_res = self._model1.compute_jacobian(model_1_vectors, temperature)
        m2_res = self._model2.compute_jacobian(model_2_vectors, temperature)
        
        out = np.zeros((m1_res.shape[0]+m2_res.shape[0], m1_res.shape[1] + m2_res.shape[1]), dtype=np.float64)
        
        image_start = 0
        m1_start = 0
        m2_start = 0
        
        for m1check, m2check in zip(model_1_checks, model_2_checks):
            m1_obs_indicies = np.argwhere(m1check).reshape(-1, 1)*2
            m1_indices = np.hstack([m1_obs_indicies, m1_obs_indicies+1]).ravel()+image_start
            
            m2_obs_indicies = np.argwhere(m2check).reshape(-1, 1)*2
            m2_indices = np.hstack([m2_obs_indicies, m2_obs_indicies+1]).ravel()+image_start
            
            m1_count = m1check.sum()*2
            m2_count = m2check.sum()*2
            
            out[m1_indices, :m1_res.shape[1]] = m1_res[m1_start:m1_start+m1_count]
            out[m2_indices, m1_res.shape[1]:] = m2_res[m2_start:m2_start+m2_count]
            
            m1_start += m1_count
            m2_start += m2_count
            
            image_start += m1_count + m2_count
            
        return out
    
    def compute_pixel_jacobian(self, vectors_in_camera_frame: ARRAY_LIKE, image: int = 0, temperature: float = 0) -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_P/\partial\mathbf{x}_C` where
        :math:`\mathbf{x}_C` is a vector in the camera frame that projects to :math:`\mathbf{x}_P` which is the
        pixel location.

        This method is used in the :class:`.LimbScanning` process in order to predict the change in a projected pixel
        location with respect to a change in the projected vector. The ``vectors_in_camera_frame`` input should
        be a 3xn array of vectors which the Jacobian is to be computed for.

        :param vectors_in_camera_frame: The vectors to compute the Jacobian at
        :param image: The image number to compute the the Jacobian for
        :param temperature: The temperature of the camera at the time the image was taken
        :return: The Jacobian matrix as a nx2x3 array
        """
        
        return self._apply_split_logic(vectors_in_camera_frame, self._model1.compute_pixel_jacobian, self._model2.compute_pixel_jacobian, 
                                       image, temperature, output_leading_shape=(), output_trailing_shape=(2, 3))
    
    def compute_unit_vector_jacobian(self, pixel_locations: ARRAY_LIKE, image: int = 0, temperature: float = 0) -> np.ndarray:
        r"""
        This method computes the Jacobian matrix :math:`\partial\mathbf{x}_C/\partial\mathbf{x}_P` where
        :math:`\mathbf{x}_C` is a vector in the camera frame that projects to :math:`\mathbf{x}_P` which is the
        pixel location.

        This method is used in the :class:`.LimbScanning` process in order to predict the change in the unit vector that
        projects to a pixel location with respect to a change in the pixel location. The
        ``pixel_locations`` input should be a 2xn array of vectors which the Jacobian is to be computed for.

        :param pixel_locations: The pixel locations to compute the Jacobian at
        :param image: The image number to compute the the Jacobian for
        :param temperature: The temperature of the camera at the time the image was taken
        :return: The Jacobian matrix as a nx3x2 array
        """
        
        return self._apply_split_logic(pixel_locations, self._model1.compute_unit_vector_jacobian, self._model2.compute_unit_vector_jacobian, 
                                       image, temperature, output_leading_shape=(), output_trailing_shape=(2, 3), frame="if")
        
    def apply_update(self, update_vec: F_ARRAY_LIKE):
        r"""
        This method takes in a delta update to camera parameters (:math:`\Delta\mathbf{c}`) and applies the update
        to the current instance in place.
        
        In general the delta update is calculated in the estimators in the :mod:`.calibration` subpackage and this
        method is not used by the user.
        
        The update vector is an array like object where each element corresponds to a specific camera parameter,
        corresponding to the element represented by each column coming from the :meth:`~CameraModel.compute_jacobian`
        method. For the SplitCamera, the update vector is split between the two component models and applied to each.

        :param update_vec: delta updates to the model parameters
        """
        model_1_elems = update_vec[:self._model1.state_vector_length]
        model_2_elems = update_vec[self._model1.state_vector_length:]
        
        self._model1.apply_update(model_1_elems)
        self._model2.apply_update(model_2_elems)
        
    def pixels_to_unit(self, pixels, temperature = 0, image = 0):
        """
        Convert pixel image locations to unit vectors expressed in the camera frame.

        This method delegates the conversion to either model1 or model2 based on the defined split logic.
        The pixel locations should be expressed as a shape (2,) or (2, n) array. They are converted
        to unit vectors by first going through the inverse distortion model (see :meth:`undistort_pixels`) and then 
        being converted to unit vectors in the camera frame according to the definitions of the current model (also 
        including any misalignment terms).

        :param pixels: The image points to be converted to unit vectors in the camera frame as a shape (2,) or (2, n) 
                       array
        :param temperature: The temperature to use for the undistortion
        :param image: The image index that the pixels belong to (only important if there are multiple misalignments)
        :return: The unit vectors corresponding to the image locations expressed in the camera frame as a shape (3,) or
                 (3, n) array.
        """
        return self._apply_split_logic(pixels, self._model1.pixels_to_unit, self._model2.pixels_to_unit, 
                                       image, temperature, output_leading_shape=(3,), output_trailing_shape=(), frame="if")
    
    def undistort_pixels(self, pixels, temperature = 0):
        """
        Compute undistorted pixel locations (gnomic/pinhole locations) for given distorted pixel locations.

        This method delegates the undistortion to either model1 or model2 based on the defined split logic.
        The ``pixels`` input should be specified as a shape (2,) or (2, n) array of image locations with units of 
        pixels. The return will be an array of the same shape as ``pixels`` with units of pixels but with distortion
        removed.

        :param pixels: The image points to be converted to gnomic (pinhole) locations as a shape (2,) or (2, n) array
        :param temperature: The temperature to use for the undistortion
        :return: The undistorted (gnomic) locations corresponding to the distorted pixel locations as an array of
                 the same shape as ``pixels``
        """
        return self._apply_split_logic(pixels, self._model1.undistort_pixels, self._model2.undistort_pixels, 
                                       temperature, output_leading_shape=(2,), output_trailing_shape=(), frame="if")
        
    def distort_pixels(self, pixels, temperature = 0):
        """
        Apply distortion to gnomic pixel locations.

        This method delegates the distortion application to either model1 or model2 based on the defined split logic.
        It takes gnomic pixel locations in units of pixels and applies the appropriate distortion to them.
        This method is used in the :meth:`distortion_map` method to generate the distortion values for each pixel.

        :param pixels: The pinhole location pixel locations the distortion is to be applied to as a shape (2,) or (2, n) array
        :param temperature: The temperature to use for the distortion
        :return: The distorted pixel locations in units of pixels as an array of the same shape as ``pixels``
        """
        return self._apply_split_logic(pixels, self._model1.distort_pixels, self._model2.distort_pixels, 
                                       temperature, output_leading_shape=(2,), output_trailing_shape=(), frame="if")
        
    def overwrite(self, model: 'CameraModel'):
        """
        Replace self with the properties of ``model`` in place.

        This method overwrites the properties of both component models (model1 and model2) with the corresponding
        properties from the input model. It is primarily used in the calibration classes to maintain the link between
        the internal and external camera models.

        :param model: The SplitCamera model to overwrite self with
        :raises ValueError: When ``model`` is not an instance of SplitCamera
        """
        assert isinstance(model, self.__class__), "Must be a split camera model"
        super().overwrite(model)
        self._model1.overwrite(model._model1)
        self._model2.overwrite(model._model2)
        
    def to_elem(self, elem: etree._Element, **kwargs) -> etree._Element: # type: ignore
        """
        Store this camera model in an :class:`lxml.etree.SubElement` object for saving in a GIANT xml file.

        This method extends the base :meth:`CameraModel.to_elem` method to include both component models (model1 and
        model2) in the XML structure. It creates separate subelements for each component model and stores their
        attributes using their respective :meth:`to_elem` methods.

        :param elem: The :class:`lxml.etree.SubElement` class to store this camera model in
        :param kwargs: Additional keyword arguments to pass to the component models' :meth:`to_elem` methods
        :return: The :class:`lxml.etree.SubElement` for this model
        """
        
        elem = super().to_elem(elem, **kwargs)
        
        model1_elem = elem.find("model1")

        if model1_elem is None:
            model1_elem = etree.SubElement(elem, "model1", attrib={"module": self._model1.__module__,
                                                                    "type": type(self._model1).__name__})

        self._model1.to_elem(model1_elem, **kwargs)
        
        model2_elem = elem.find("model2")

        if model2_elem is None:
            model2_elem = etree.SubElement(elem, "model2", attrib={"module": self._model2.__module__,
                                                                    "type": type(self._model2).__name__})

        self._model2.to_elem(model2_elem, **kwargs)
        
        return elem
    
    @classmethod
    def from_elem(cls, elem: etree._Element) -> 'SplitCamera':
        """
        Construct a new instance of SplitCamera from an :class:`etree._Element` object.

        This class method extends the base :meth:`CameraModel.from_elem` method to handle the reconstruction of both
        component models (model1 and model2) from the XML structure. It retrieves the necessary information for each
        component model, imports the appropriate classes, and initializes them using their respective :meth:`from_elem`
        methods.

        :param elem: The element containing the attribute information for the instance to be created
        :return: An initialized instance of SplitCamera with both component models properly set
        :raises KeyError: If the required nodes for model1 or model2 are missing in the XML element
        """
        
        inst = cast(SplitCamera, super().from_elem(elem))
        
        # find and update model1
        model1_node: etree._Element = elem.find("model1")
        if model1_node is None:
            raise KeyError("Missing node for model1")
        
        m1mod = import_module(model1_node.get('module'))

        m1cls: type[CameraModel] = getattr(m1mod, model1_node.get('type'))
        
        model1 = m1cls.from_elem(model1_node)
            
        # find and update model2
        model2_node: etree._Element = elem.find("model2")
        if model2_node is None:
            raise KeyError("Missing node for model2")
        
        m2mod = import_module(model2_node.get('module'))

        m2cls = getattr(m2mod, model2_node.get('type'))
        
        model2: type[CameraModel] = m2cls.from_elem(model2_node)
        
        inst._model1 = model1
        inst._model2 = model2
        
        return inst
    
    def check_in_fov(self, vectors: F_ARRAY_LIKE, image: int = 0, temperature: float = 0) -> NDArray[np.bool]:
        """
        Determines if any points in the array are within the field of view of the camera.
        
        This method delegates the field of view check to either model1 or model2 based on the defined split logic.
        It checks whether the given vertices are within the field of view of the appropriate camera model.


        :param vectors: Vectors to check if they are in the field of view of the camera expressed as a shape (3, n) array in the camera frame.  
        :param image: The index of the image being projected onto (only applicable with multiple misalignments)
        :param temperature: The temperature of the camera to use for the projection
        :return: A boolean array the same length as the number of columns of vectors. False by default, True if the point is in the FOV.
        """
        
        return self._apply_split_logic(vectors, self._model1.check_in_fov, self._model2.check_in_fov, image, temperature, output_leading_shape=(), output_trailing_shape=(), output_dtype=np.bool_)
            
    def __str__(self):
        """
        Return a string representation of the SplitCamera object.

        This method provides a human-readable description of the SplitCamera, including string representations
        of both component models (model1 and model2).

        :return: A string describing the SplitCamera and its component models
        """
        return f'SplitCamera:\n\nModel 1:\n{self.model1.__str__()}\n\nModel 2:\n{self.model2.__str__()}'
    
    def __repr__(self):
        """
        Return a string representation of the SplitCamera object that can be used to recreate the object.

        This method returns a string that, when passed to eval(), would create a new instance of the SplitCamera
        with the same parameters as the current instance. It includes representations of both component models
        and all the split parameters.

        :return: A string representation of the SplitCamera that can be used to recreate the object
        """
        return (f'SplitCamera({self.model1.__repr__()}, {self.model2.__repr__()}, '
                f'camera_frame_split_axis={self._camera_frame_split_axis}, camera_frame_split_threshold={self._camera_frame_split_threshold}'
                f'image_plane_split_axis={self._image_plane_split_axis}, image_plane_split_threshold={self._image_plane_split_threshold}'
                f'field_of_view={self.field_of_view}, n_rows={self.n_rows}, n_cols={self.n_cols}, use_a_prior={self.use_a_priori})')
    
        
            