from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray, DTypeLike

from giant.image_processing.feature_matchers.feature_matcher import FeatureMatcher
from giant.image_processing.utilities.image_validation_mixin import ImageValidationMixin
from giant._typing import DOUBLE_ARRAY
from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured


try:
    from romatch.models import roma_outdoor
    import torch
    from PIL import Image
    
    def _determine_default_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    @dataclass
    class RoMaFeatureMatcherOptions(UserOptions):
        device: torch.device = field(default_factory=_determine_default_device)
        """
        What device to work on
        """
        
        coarse_res: int | tuple[int, int] = 560
        """
        The initial coarse resolution of the image (must be a multiple of 14)
        """
        
        upsample_res: tuple[int, int] = (864, 864)
        """
        The resolution to upsample the image to
        """
        
        sample_thresh: float = 0.05
        """
        Controls the thresholding used when sampling matches for estimation.
        
        In certain cases a lower or higher threshold may improve results.
        """
        

    class RoMaFeatureMatcher(UserOptionConfigured[RoMaFeatureMatcherOptions], RoMaFeatureMatcherOptions, FeatureMatcher,
                             AttributeEqualityComparison, AttributePrinting, ImageValidationMixin):
        """
        Implementation of a matcher using RoMa.
        """
        
        allowed_dtypes: list[DTypeLike] = [np.uint8]

        def __init__(self, options: RoMaFeatureMatcherOptions | None = None):
            """
            Initialize the RomaKeypointMatcher.

            
            :param options: The options to configure with
            :param romatch_checkpoint: Path to the RoMa model checkpoint.
            """
            super().__init__(RoMaFeatureMatcherOptions, options=options) # ratio_threshold is unused
            self.roma_model = roma_outdoor(device=self.device, coarse_res=self.coarse_res, upsample_res=self.upsample_res)
            self.roma_model.sample_thresh = self.sample_thresh
            print(f"RoMa model loaded on {self.device}")

        def match_images(self, image1: NDArray, image2: NDArray) -> DOUBLE_ARRAY:
            """
            Matches keypoints by overriding the base class method to use RoMa's
            end-to-end matching process.

            :param image1: The first image to match (as a NumPy array).
            :param image2: The second image to match (as a NumPy array).

            :returns: An array of the matched keypoint locations of shape (N, 2, 2).
            """
            # Convert images to PIL RGB
            # RoMa's default processing expects RGB.
            
            image1_pil = Image.fromarray(self._validate_and_convert_image(image1)).convert('RGB')
            image2_pil = Image.fromarray(self._validate_and_convert_image(image2)).convert('RGB')

            # Use RoMa to get correspondences
            warp, certainty = self.roma_model.match(image1_pil, image2_pil, device=self.device)
            
            # get the correspondences
            matches, certainty = self.roma_model.sample(warp, certainty)
            kpts1, kpts2 = self.roma_model.to_pixel_coordinates(matches, *image1.shape, *image2.shape)

            # Reshape the (N, 4) array to (N, 2, 2)
            matched_keypoints_array = np.concat([kpts1.cpu().numpy().reshape(-1, 1, 2), kpts2.cpu().numpy().reshape(-1, 1, 2)], axis=1)

            return matched_keypoints_array.astype(np.float64)
        
except ImportError:
    raise ImportError('RoMa is not installed. Please clone and follow the install instructions from https://github.com/Parskatt/RoMa/tree/main')
 