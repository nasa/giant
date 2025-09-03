import numpy as np
from numpy.typing import NDArray, DTypeLike


class ImageValidationMixin:
    """
    A mixin class that provides functionality for validating and converting image data types.

    This mixin is designed to be used with image processing classes that require
    specific data types for their input images. It ensures that the input image
    is of an allowed data type before processing, and converts it if necessary.

    Attributes:
        allowed_dtypes (list[DTypeLike]): A list of allowed data types for the image.
            This must be implemented by subclasses.

    Methods:
        _validate_and_convert_image: Checks if an image is of an allowed dtype
            and converts it if not.
    """
    
    allowed_dtypes: list[DTypeLike]
    """
    A list of dtypes allowed by the detector.
    
    This must be implemented by subclasses.
    """
    
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if not self.allowed_dtypes:
            raise NotImplementedError(f'allowed_dtypes must be specified as a class attribute for {self.__class__.__name__}')

    
    def _validate_and_convert_image(self, image: NDArray) -> NDArray:
        """
        Checks if an image is of an allowed dtype and converts it if not.
        
        :param image: the image to validate
            
        :returns: The image as the rigth dtype
        """
        if image.dtype in self.allowed_dtypes:
            return image

        target_dtype = self.allowed_dtypes[0]

        # Check if the target data type is an integer type
        if np.issubdtype(target_dtype, np.integer):
            # Get the maximum possible value for the target integer type
            target_max = np.iinfo(target_dtype).max
            target_min = np.iinfo(target_dtype).min
            
            # Use float64 for precision during the scaling calculation
            scaled_image = image.astype(np.float64).copy()
            source_max = scaled_image.max()
            
            scale = 1
            
            if target_min < 0:
                
                source_min = scaled_image.min()
                
                abs_max = np.argmax([abs(source_min), abs(source_max)])
                
                if abs_max == 0 and target_min != 0:
                    scale = target_min/source_min
                elif target_max != 0:
                    scale = target_max/source_max
                    
                if scale < 0:
                    raise ValueError('Something is wrong')
                
            else:
                if source_max > 0:
                    scale = target_max/source_max

            # Safely scale the image to the target's full range
            scaled_image *= scale
                
            
            # Round to the nearest integer and cast to the final type
            return np.round(scaled_image).astype(target_dtype)
        else:
            # For floating point targets, a direct cast is sufficient
            return image.astype(target_dtype)
    