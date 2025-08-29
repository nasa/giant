"""
Create a mosaic image from GIANT cameras.

This experimental script is just for fun.  It will create a mosaic image for a supply of grayscale images with come
configurable parameters.  Things are definitely not finalized so use at your own risk.

.. warning::

    This script loads/saves some results from/to python pickle files.  Pickle files can be used to execute arbitrary
    code, so you should never open one from an untrusted source.
"""

from multiprocessing import Pool

from argparse import ArgumentParser

from typing import Sequence, Optional

import cv2
import numpy as np

from scipy.stats import pearsonr

from giant._typing import PATH
from giant.ray_tracer.scene import SceneObject, Scene
from giant.ray_tracer.shapes import Shape
from giant.camera import Camera

# added a warning to the documentation
import dill  # nosec


_SCAN_DIRS = np.array([[1, 0, -1, 0], [0, 1, 0, -1], [0, 0, 0, 0]])


def _get_parser() -> ArgumentParser:
    """
    Helper function for the argparse extension

    :return: A setup argument parser
    """

    warning = "WARNING: This script loads some results from python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."

    parser = ArgumentParser(description='Mosaic images', epilog=warning)
    parser.add_argument('target_camera', help='The camera file containing the target image as the first image')
    parser.add_argument('cameras', nargs='+', help='The camera files containing the mosaic images')
    parser.add_argument('-t', '--target_object_file', help='The file containing the target and light source auto '
                                                           'scene objects',
                        default=None)
    parser.add_argument('-s', '--scale', type=int, help='The scale value to scale up the target image by',
                        default=5)
    parser.add_argument('-o', '--output', help='The file to save the mosaiced to',
                        default='./mosaiced.png')
    parser.add_argument('-w', '--show', help='show the mosaiced image in real time', action='store_true')
    parser.add_argument('-c', '--crop_mosaics', help='Crop the mosaics according to the predicted location of the '
                                                     'body', action='store_true')
    parser.add_argument('-m', '--meta_kernel', help='the kernel file to load in spice', default=None)
    parser.add_argument('-a', '--mosaic_scale', type=float, default=1.0, help='the value to scale the mosaics by')
    parser.add_argument('-g', '--sub_grid_check', type=int, default=5, help='The number of sub grids to use')
    parser.add_argument('-r', '--allow_rotation', help='Allow the mosaic images to be rotated', action='store_true')

    return parser


class _MosaicMaker:
    """
    This class builds mosaics out of gray-scale images.

    This class is not intended to be used external from the mosaic_images script, so use at your own risk.
    """

    def __init__(self, target_image: np.ndarray, mosaic_camera_files: Sequence[PATH],
                 target_object: SceneObject, light_source: SceneObject,
                 target_scale: int = 5, crop_mosaics: bool = True, mosaic_scale: float = 1.0,
                 sub_grid_check: int = 5, allow_rotation: bool = True):

        self.original_target_image = target_image.astype(np.float32)
        """
        The original image that is the target for the mosaic.
        
        This will be resized according to the target_scale parameter to make the actual target image
        """

        self.target_scale = target_scale
        """
        The value to scale the target image by.  
        
        Normally this should be larger than 1.
        """

        self.target_image = cv2.resize(self.original_target_image, None, fx=target_scale, fy=target_scale)
        """
        The actual target image after scaling
        """

        self.mosaic_camera_files = mosaic_camera_files
        """
        A list of camera files containing images to use in the mosaics.  
        
        Multiple files enable parallel processing.
        """

        self.crop_mosaics = crop_mosaics
        """
        A flag specifying whether to crop the mosaics to only include the body
        
        If set to true the the mosaic images will be cropped to discard most of empty space and they will then be 
        scaled to all be the same size as the median size of the cropped images.
        """

        self.mosaic_scale = mosaic_scale
        """
        The value to scale the mosaics by.  
        """

        self.sub_grid_check = sub_grid_check
        """
        The square root of the number of subdivisions to use when searching for a good match for a portion of the target
        image.
        
        Higher values will lead to a better final result at the expense of more computation time.
        """

        self.allow_rotation = allow_rotation
        """
        A flag specifying whether to allow rotating the mosaic images to get a better fit.
        """

        self.target_object = target_object
        """
        The object that is being viewed
        """

        self.light_source = light_source
        """
        The light source that is being viewed
        """

        self.number_target_image_bins = (0, 0)
        """
        The number of bins for the target images. 
        
        This is set based off of the final size of the mosaic images.
        """

        self.target_image_bins = []
        """
        The vectors describing each bin of the target image.
        
        The vector is comprised of the mean of each sub bin in row major order. Therefore, if sub_grid_check is 3, then
        each element of this list will be a length 9 array. 
        
        The vectors are also listed in row major order
        """

        self.prepared_mosaic_images = []
        """
        The list of mosaic images after cropping and scaling
        """

        self.mosaic_image_bins = []
        """
        The vectors describing each mosaic image.
        
        The vector is comprised of the mean of each sub bin in row major order. Therefore, if sub_grid_check is 3, then
        each element of this list will be a length 9 array. 
        
        Each element of this list corresponds to
        """

        self.mosaic_image_size = 0
        """
        The size of the cropped/scaled mosaic images.  
        
        This is set after preparing the mosaic images
        """

    def prepare_target_image_bins(self):

        n_row_bins = self.target_image.shape[0] // self.mosaic_image_size
        n_col_bins = self.target_image.shape[1] // self.mosaic_image_size

        self.number_target_image_bins = (n_row_bins, n_col_bins)

        for i in range(n_row_bins):
            for j in range(n_col_bins):
                self.target_image_bins.append(
                    self.bin_image(self.target_image[i*self.mosaic_image_size:(i+1)*self.mosaic_image_size,
                                                     j*self.mosaic_image_size:(j+1)*self.mosaic_image_size])
                )

    def prepare_mosaic_images_cam(self, cam_file):

        with open(cam_file, 'rb') as ifile:
            # added a warning to the documentation
            cam: Camera = dill.load(ifile)  # nosec

        cropped_images = []

        for ind, image in cam:
            if self.crop_mosaics:
                scene = Scene([self.target_object], self.light_source)
                scene.update(image)

                if (cs := getattr(self.target_object, 'circum_sphere')) is not None:
                    limbs = cs.find_limbs(self.target_object.position.ravel() /
                                          np.linalg.norm(self.target_object.position),
                                          _SCAN_DIRS, self.target_object.position.ravel())

                    image_locs = cam.model.project_onto_image(limbs + self.target_object.position.reshape(3, 1),
                                                              temperature=image.temperature)

                else:
                    assert isinstance(self.target_object.shape, Shape)
                    image_locs = cam.model.project_onto_image(self.target_object.shape.bounding_box.vertices,
                                                              temperature=image.temperature)

                crop_start = np.floor(image_locs.min(axis=1)).astype(int)

                crop_size = np.minimum(((np.ceil(image_locs.max(axis=1)) - crop_start) + 1).max(), 10)

                crop_end = crop_start + crop_size

                cropped_images.append(cv2.resize(image[crop_start[1]:crop_end[1],
                                                 crop_start[0]:crop_end[0]].astype(np.float32),
                                                 None, fx=self.mosaic_scale, fy=self.mosaic_scale,
                                                 interpolation=cv2.INTER_AREA))
            else:
                smallest_axis = np.min(image.shape)

                cropped_images.append(cv2.resize(image[:smallest_axis, :smallest_axis], None,
                                                 fx=self.mosaic_scale, fy=self.mosaic_scale,
                                                 interpolation=cv2.INTER_AREA))

        return cropped_images

    def size_mosaics(self, img):
        if img.shape[0] < self.mosaic_image_size:
            out = cv2.resize(img, (self.mosaic_image_size, self.mosaic_image_size),
                             interpolation=cv2.INTER_LINEAR)
        elif img.shape[0] > self.mosaic_image_size:
            out = cv2.resize(img, (self.mosaic_image_size, self.mosaic_image_size),
                             interpolation=cv2.INTER_AREA)
        else:
            out = img

        return out

    def prepare_mosaic_images(self):

        if len(self.mosaic_camera_files) > 1:
            with Pool() as pool:
                cropped_images_stack = pool.map(self.prepare_mosaic_images_cam, self.mosaic_camera_files)

            cropped_images = []
            for c in cropped_images_stack:
                cropped_images.extend(c)

        else:
            cropped_images = self.prepare_mosaic_images_cam(self.mosaic_camera_files[0])

        self.mosaic_image_size = int(np.median([x.shape[0] for x in cropped_images]))

        with Pool() as pool:
            self.prepared_mosaic_images = pool.map(self.size_mosaics, cropped_images)

    def bin_image(self, image):
        return cv2.resize(image, (self.sub_grid_check, self.sub_grid_check), interpolation=cv2.INTER_AREA).ravel()

    def prepare_mosaic_bins(self):

        assert self.prepared_mosaic_images is not None, "the mosaic images must be prepared at this point"
        prepared_imgs = self.prepared_mosaic_images
        self.prepared_mosaic_images = None
        
        with Pool() as pool:
            self.mosaic_image_bins = pool.map(self.bin_image, prepared_imgs)
        self.prepared_mosaic_images = prepared_imgs

        # self.mosaic_image_bins = []
        # for img in self.prepared_mosaic_images:
        #     self.mosaic_image_bins.append(self.bin_image(img))

    def match_bin(self, bin_ind):
        image_bin = self.target_image_bins[bin_ind]

        imean = image_bin.mean()

        best_score = -1
        best_rotation = None
        best_scale = 1
        best_match = None
        for mind, mbin in enumerate(self.mosaic_image_bins):
            score = pearsonr(image_bin, mbin).correlation
            rotation = None

            if self.allow_rotation:
                for r in [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    mbinr = cv2.rotate(mbin.reshape(self.sub_grid_check, self.sub_grid_check), r).ravel()
                    scorer = pearsonr(image_bin, mbinr).correlation

                    if scorer > score:
                        score = scorer
                        rotation = r

            if score > best_score:
                best_scale = mbin.mean() / imean
                best_rotation = rotation
                best_score = score

                best_match = mind

        return best_match, best_scale, best_rotation

    def mosaic_image(self):

        out_mosaic_shape = np.array(self.number_target_image_bins)*self.mosaic_image_size
        out_mosaic = np.zeros(out_mosaic_shape.astype(int), dtype=np.float32)

        with Pool() as pool:
            matches = pool.map(self.match_bin, range(len(self.target_image_bins)))

        nbin = 0
        assert self.prepared_mosaic_images is not None
        for r in range(self.number_target_image_bins[0]):
            for c in range(self.number_target_image_bins[1]):
                match, scale, rotation = matches[nbin]
                if match is not None:
                    matched_mosaic = self.prepared_mosaic_images[match]*scale
                    if rotation is not None:
                        matched_mosaic = cv2.rotate(matched_mosaic, rotation)
                    out_mosaic[r*self.mosaic_image_size:(r+1)*self.mosaic_image_size,
                            c*self.mosaic_image_size:(c+1)*self.mosaic_image_size] = matched_mosaic

                nbin += 1

        return out_mosaic

    def make_mosaic(self):

        self.prepare_mosaic_images()
        self.prepare_mosaic_bins()
        self.prepare_target_image_bins()

        return self.mosaic_image()

    @classmethod
    def main(cls):
        parser = _get_parser()

        args = parser.parse_args()

        if args.meta_kernel is not None:
            import spiceypy as spice
            spice.furnsh(args.meta_kernel)

        with open(args.target_camera, 'rb') as ifile:
            # added a warning to the documentation
            target_image = dill.load(ifile).images[0]  # nosec

        if args.target_object_file is not None:
            with open(args.target_object_file, 'rb') as ifile:
                # added a warning to the documentation
                target_object = dill.load(ifile)  # nosec
                light_object = dill.load(ifile)  # nosec
        else:
            raise ValueError('the target must be specified')

        inst = cls(target_image, args.cameras, target_object=target_object, light_source=light_object,
                   target_scale=args.scale, crop_mosaics=args.crop_mosaics, mosaic_scale=args.mosaic_scale,
                   sub_grid_check=args.sub_grid_check, allow_rotation=args.allow_rotation)

        out = inst.make_mosaic()

        cv2.imwrite(args.output, out)

        if args.show:
            import matplotlib.pyplot as plt
            plt.imshow(out)
            plt.show()


if __name__ == '__main__':

    _MosaicMaker.main()