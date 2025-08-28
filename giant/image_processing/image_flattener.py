from typing import cast, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

import scipy.signal as sig

import cv2

from giant.utilities.outlier_identifier import get_outliers
from giant.utilities.options import UserOptions
from giant.utilities.mixin_classes.user_option_configured import UserOptionConfigured
from giant.utilities.mixin_classes.attribute_equality_comparison import AttributeEqualityComparison
from giant.utilities.mixin_classes.attribute_printing import AttributePrinting


class ImageFlattenerOut(NamedTuple):
    flattened: NDArray[np.float32]
    """
    The flattened image.
    
    The same shape as the input and with a float32 datatype.
    """
    
    estimated_noise_level: float | list[float]
    """
    The estimated noise level in the image as a float (if GLOBAL flattening is used) or as a list of floats (if local flatttening is used).
    
    If local flattening is used, each element of this list corresponds to the same element in the slice array
    """
    
    local_slices: list[tuple[slice, slice]] | None
    """
    The slices used for local flattening and noise approximation, or `None` if GLOBAL flattening is used
    """


class ImageFlatteningNoiseApprox(Enum):
    """
    This enumeration provides the valid options for flattening an image and determining the noise levels when
    identifying points of interest in :meth:`.ImageProcessing.find_poi_in_roi`

    You should be sure to use one of these values when setting to the :attr:`.image_flattening_noise_approximation`
    attribute of the :class:`.ImageProcessing` class.
    """

    GLOBAL = auto()
    """
    Globally flatten the image and estimate the noise level from it.
    
    In this the image in flattened by subtracting a median filtered version of the image from it and a single noise 
    level is approximated for the entire image either through sampling or through the :attr:`.dark_pixels` of the image.
    
    For most OpNav cases this is sufficient and fast.
    """

    LOCAL = auto()
    """
    Locally flatten the image and estimate the noise levels for each local region

    In this the image in flattened by splitting it into regions, estimating a linear background gradient in each region,
    and the subtracting the estimated background gradient from the region to get the flattened region.  An individual 
    noise level is estimated for each of these regions through sampling.

    This technique allows much dimmer points of interest to be extracted without overwhelming with noise, but it is 
    generally much slower and is unnecessary for all but detailed analyses.
    """



@dataclass
class ImageFlattenerOptions(UserOptions):
    
    image_flattening_noise_approximation: ImageFlatteningNoiseApprox = ImageFlatteningNoiseApprox.GLOBAL
    """
    This specifies whether to globally flatten the image and compute a single noise level or to locally do so.
    
    Generally global is sufficient for star identification purposes.  If you are trying to extract very dim stars 
    (or particles) then you may need to use the ``'LOCAL'`` option, which is much better for low SNR targets but 
    much slower.
    """
    
    local_flattening_kernel_size: int = 7
    """
    This specifies the half size of the kernel to use when locally flattening an image.  
    
    If you are using global flattening of an image this is ignored.
    
    The size of the kernel/region used in flattening the image will be ``2*local_flattening_kernel_size+1``.
    """
    
    
class ImageFlattener(UserOptionConfigured[ImageFlattenerOptions],
                     ImageFlattenerOptions,
                     AttributePrinting,
                     AttributeEqualityComparison):
    
    def __init__(self, options: ImageFlattenerOptions | None = None) -> None:
        super().__init__(ImageFlattenerOptions, options=options)
        
    @staticmethod
    def _global_flat_image_and_noise(image: np.ndarray) -> ImageFlattenerOut:
        """
        This method is used to sample the noise level of an image, as well as return a flattened version of the image.

        The image is flattened by subtracting off a median filtered copy of the image from the image itself

        The standard deviation of the noise level in the image is estimated by either calculating the standard deviation
        of flattened user defined dark pixels for the image (contained in the :attr:`.OpNavImage.dark_pixels` attribute)
        or by calculating the standard deviation of 2,000 randomly sampled differences between pixel pairs of the
        flattened image spaced 5 rows and 5 columns apart.

        This method is used by :meth:`locate_subpixel_poi_in_roi` in order to make the point of interest identification
        easier.

        :param image: The image to be flattened and have the noise level estimated for
        :return: The flattened image and the noise level as a tuple
        """

        # flatten the image by subtracting a median blurred copy of the image.  Using a blurring kernel of 5x5.
        flat_image: NDArray[np.float32] = (image.astype(np.float32) - cast(NDArray[np.float32], cv2.medianBlur(image.copy().astype(np.float32), 5)))

        dark_pixels = getattr(image, 'dark_pixels', None)
        if dark_pixels is not None:  # if there are identified dark pixels
            # flatten the dark pixels using a median filter
            flat_dark = dark_pixels.astype(np.float64) - \
                        sig.medfilt(dark_pixels.astype(np.float64))

            # compute the standard deviation of the flattened dark pixels
            standard_deviation = float(np.nanstd(flat_dark) / 2)

        else:  # otherwise, sample the image to determine the randomness
            # determine the randomness of the image by sampling at 10000 randomly selected points compared with point +5
            # rows and +5 cols from those points

            im_shape = flat_image.shape

            dist = np.minimum(np.min(im_shape) - 1, 5)

            if dist <= 0:
                raise ValueError('the input image is too small...')

            # get the total possible number of starting locations
            num_pix: float = float(np.prod(np.array(im_shape) - dist)) 

            # sample at most 1 quarter of the available starting locations
            num_choice = int(np.minimum(num_pix // 4, 2000))

            # choose a random sample of starting locations
            start_rows, start_cols = np.unravel_index(np.random.choice(np.arange(int(num_pix)), num_choice,
                                                                       replace=False),
                                                      np.array(im_shape) - dist)

            # get the other half of the sample
            next_rows = start_rows + dist
            next_cols = start_cols + dist

            # compute the standard deviation of the difference between the star points and hte next points.  This
            # measures the noise in the image and sets the threshold for identifiable stars.
            data = (flat_image[next_rows, next_cols] - flat_image[start_rows, start_cols]).ravel()

            # reject outliers from the data using MAD
            outliers = get_outliers(data)

            # compute the standard deviation
            standard_deviation = float(np.nanstd(data[~outliers]) / 2)

        return ImageFlattenerOut(flat_image, standard_deviation, None)

    # TODO: This would probably be better as a cython function where we can do parallel processing
    def _local_flat_image_and_noise(self, image) -> ImageFlattenerOut:
        """
        This method flattens the image and approximates the noise over regions of the image.

        This is not intended by the user, instead use :meth:`flatten_image_and_get_noise_level`.

        :param image: The image which is to be flattened and have noise levels estimated for
        :return: The flattened image, a list of noise values for regions of the image, and a list of tuples of slices
                 describing the regions of the image
        """

        # get the shape of the image
        img_shape = image.shape

        # make sure that the image is double, also copy it to ensure that we don't mess up the original
        flat_image = image.astype(np.float32).copy()

        # start the region center at the kernel size
        current_row = self.local_flattening_kernel_size
        current_col = self.local_flattening_kernel_size

        # initialize the lists for return
        noises, slices = [], []

        # loop rows through until we've processed the whole image
        while current_row < img_shape[0]:
            # get the row bounds and slice
            lower_row = current_row - self.local_flattening_kernel_size
            upper_row = min(current_row + self.local_flattening_kernel_size + 1, img_shape[0])
            row_slice = slice(lower_row, upper_row)

            # loop through columns until we've processed the whole image
            while current_col < img_shape[1]:
                # get the column bounds and slice
                lower_column = current_col - self.local_flattening_kernel_size
                upper_column = min(current_col + self.local_flattening_kernel_size + 1, img_shape[1])
                column_slice = slice(lower_column, upper_column)

                # get the row/column labels that we are working with
                rows, cols = np.mgrid[row_slice, column_slice]

                # get the region from the original image we are editing
                region = image[row_slice, column_slice].astype(np.float32)

                # compute the background of the region using least squares [1, x, y] @ [A, B, C] = bg
                h_matrix = np.vstack([np.ones(rows.size), cols.ravel(), rows.ravel()]).T.astype(np.float32)
                background = np.linalg.lstsq(h_matrix, region.ravel(), rcond=None)[0].ravel()

                # flatten the region by subtracting the linear background approximation
                flat_image[row_slice, column_slice] -= (h_matrix@background.reshape(3, 1)).reshape(region.shape)

                # store the slices
                slices.append((row_slice, column_slice))

                # update the current column we're centered on
                current_col += 2 * self.local_flattening_kernel_size + 1

            # update the current row/column we're centered on
            current_row += 2 * self.local_flattening_kernel_size + 1
            current_col = self.local_flattening_kernel_size

        # make sure we're extra flat by flattening the flat image with a median blur.
        flat_image: np.ndarray = (flat_image - cv2.medianBlur(flat_image.copy(), 5))

        for local_slice in slices:
            region = flat_image[local_slice[0], local_slice[1]].ravel()
            selections = np.random.choice(np.arange(int(region.size)), int(region.size//2), replace=False)

            selected_region: np.ndarray = region[selections]

            outliers = get_outliers(selected_region)

            if outliers.sum() > selections.size//2:

                local_std: float = selected_region.std()
            else:
                local_std: float = selected_region[~outliers].std()

            noises.append(local_std)

        return ImageFlattenerOut(flat_image, noises, slices)

    def __call__(self, image: NDArray) -> ImageFlattenerOut:
        """
        This method is used to sample the noise level of an image, as well as return a flattened version of the image.

        There are 2 techniques for flattening the image.

        In the first, ``GLOBAL`` technique: the image is flattened by subtracting off a median filtered copy of the
        image from the image itself

        The standard deviation of the noise level in the image is then estimated by either calculating the standard
        deviation of flattened user defined dark pixels for the image (contained in the :attr:`.OpNavImage.dark_pixels`
        attribute) or by calculating the standard deviation of 2,000 randomly sampled differences between pixel pairs of
        the flattened image spaced 5 rows and 5 columns apart.

        In the second, ``LOCAL`` technique: the image is split into regions based on :attr:`local_flattening_kernel_size`.
        For each region, a linear background gradient is estimated and subtracted from the region.  The global flattened
        image is then flattened further by subtracting off a median filtered copy of the flattened image.

        The standard deviation of the noise level is then computed for each region by sampling about half of the points
        in the flattened region and computing the standard deviation of the flattened intensity values.  In this case
        3 values are returned, the flattened image, the list of noise values for each region, and a list of slices
        defining the regions that were processed.

        :param image: The image to be flattened and have the noise level estimated for
        :return: The flattened image, the noise level(s), and a list of slices of tuples specifying the regions of 
                 the image the noise levels apply to (or None) as a NamedTuple.
        """

        if self.image_flattening_noise_approximation == ImageFlatteningNoiseApprox.GLOBAL:
            return self._global_flat_image_and_noise(image)
        else:
            return self._local_flat_image_and_noise(image)
