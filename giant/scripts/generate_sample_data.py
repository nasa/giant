# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Generate sample images and camera meta-data for doing the examples included throughout the documentation.

This script should only need to be run once per install and only if you want to work through the examples that are
included in the documentation.

This script makes a new directory called "sample_data" in the current directory and stores the sample images in that
directory.

This script can take a while to run so it is recommended that you use ``nohup`` and allow it to run in the background.

.. warning::

    This script loads/saves some results from/to python pickle files.  Pickle files can be used to execute arbitrary
    code, so you should never open one from an untrusted source.  While this script should only be opening
    pickle files created locally on your machine by GIANT, we wanted to warn you about the risk.  To be extra sure you
    can check that "./sample_data/kdtree.pickle" either doesn't exist or was created by you at a recognized time stamp.
"""

from pathlib import Path
from datetime import datetime, timedelta

from argparse import ArgumentParser

from copy import copy

# security risk addressed with warning
import pickle  # nosec

from urllib.request import urlopen

import time

import numpy as np

from giant.camera import Camera
from giant.camera_models import BrownModel
from giant.image import OpNavImage, ExposureType
from giant.rotations import Rotation, euler_to_rotmat

from giant.catalogs.giant_catalog import GIANTCatalog

from giant.stellar_opnav.star_identification import StarID

from giant.image_processing import ImageProcessing
from giant.point_spread_functions import Gaussian

from giant.ray_tracer.kdtree import KDTree
from giant.ray_tracer.shapes import Point
from giant.ray_tracer.scene import Scene, SceneObject
from giant.ray_tracer.illumination import McEwenIllumination
from giant.relative_opnav.estimators.cross_correlation import XCorrCenterFinding
from giant.utilities.stereophotoclinometry import ShapeModel
from giant.ray_tracer.utilities import ref_ellipse


# set the seed so we always get the same things
np.random.seed(52191)


MODEL = BrownModel(fx=3470, fy=3470, k1=-0.5, k2=0.3, k3=-0.2, p1=2e-5, p2=8e-5, px=1260, py=950,
                   n_rows=1944, n_cols=2592, field_of_view=25)  # type: BrownModel
"""
The camera model to translate directions in the camera frame to points on an image and vice-versa.

This is the model used to render the sample data.
"""

EPOCH = datetime(2020, 1, 1)


def _get_parser() -> ArgumentParser:
    """
    Helper function for the argparse extension

    :return: A setup argument parser
    """

    warning = "WARNING: This script loads/saves some results from/to python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."
    return ArgumentParser(description='Generate the sample data used for the examples in the GIANT documentation',
                          epilog=warning)


PSF: Gaussian = Gaussian(sigma_x=1, sigma_y=2, size=5) 
"""
The camera point spread function.

This is the PSF used to render the sample data.
"""


def _render_stars(camera_to_render: Camera) -> None:
    """
    Render stars in an image

    :param camera_to_render: The camera to render the stars into
    """

    camera_to_render.only_long_on()

    ii = 1

    psf = copy(PSF)
    for _, image_to_render in camera_to_render:
        start = time.time()
        # make the SID object to get the star locations
        sid = StarID(MODEL, catalog=GIANTCatalog(), max_magnitude=5.5,
                     a_priori_rotation_cat2camera=image_to_render.rotation_inertial_to_camera,
                     camera_position=image_to_render.position, camera_velocity=image_to_render.velocity)

        sid.project_stars(epoch=image_to_render.observation_date, temperature=image_to_render.temperature)

        drows, dcols = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10), indexing='ij')

        # set the reference magnitude to be 2.5, which corresponds to a dn of 2**14
        mref = 2.5
        inten_ref = 2 ** 14

        for ind, point in enumerate(sid.queried_catalog_image_points.T):
            rows = np.round(drows + point[1]).astype(int)
            cols = np.round(dcols + point[0]).astype(int)

            valid_check = (rows >= 0) & (rows < MODEL.n_rows) & (cols >= 0) & (cols < MODEL.n_cols)

            if valid_check.any():
                use_rows = rows[valid_check]
                use_cols = cols[valid_check]

                # compute the intensity
                inten = 10 ** (-(sid.queried_catalog_star_records.iloc[ind].mag - mref) / 2.5) * inten_ref
                psf.amplitude = inten
                psf.centroid_x = point[0]
                psf.centroid_y = point[1]

                np.add.at(image_to_render, (use_rows, use_cols), psf.evaluate(use_cols, use_rows))
        print('rendered stars for image {} of {} in {:.3f} secs'.format(ii, sum(camera_to_render.image_mask),
                                                                        time.time() - start))

        ii += 1

    camera_to_render.all_on()


def _prepare_shape(directory: Path) -> KDTree:
    start = time.time()
    print('downloading shape', flush=True)
    temp_file = directory / "shape_download.txt"
    # the url is hard coded and not user configured therefore it should generally be safe
    url = 'https://darts.isas.jaxa.jp/pub/pds3/hay-a-amica-5-itokawashape-v1.0/data/quad/quad512q.tab'
    # the file is only a text file and we only read it as such, so there isn't any security risk in it AFAIK
    with urlopen(url) as dshape:  # nosec
        if dshape.getcode() == 200:
            with temp_file.open('wb') as ofile:
                ofile.write(dshape.read())

        else:
            print("Couldn't download the shape due to code {}".format(dshape.getcode()), flush=True)
            exit(-1)

    print('downloaded shape in {:.3f} seconds'.format(time.time() - start), flush=True)
    print('building shape', flush=True)
    start = time.time()
    shape = ShapeModel(temp_file)

    tris = shape.get_triangles()

    tris.reference_ellipsoid = ref_ellipse(shape.vertices.T)

    kdt = KDTree(tris, max_depth=18)

    kdt.build(print_progress=False, force=False)
    print('shape built in {:.3f} seconds'.format(time.time()-start), flush=True)

    return kdt


def _truth_camera_position(date: datetime) -> np.ndarray:
    """
    Return the inertial position of the camera at the requested time.

    :param date: the date the position is requested
    :return: the position vector from the SSB to the camera in km in the inertial frame
    """
    # rates in km/sec
    rate_x = 0.1
    rate_y = -2.005
    rate_z = 0.05

    start = np.array([1e10, 500, 20000])

    dt = (date - EPOCH)/timedelta(seconds=1)

    return start + dt*np.array([rate_x, rate_y, rate_z])


def _truth_camera_orientation(date: datetime) -> Rotation:
    """
    Return the truth camera orientation from the inertial frame to the camera frame at the requested time

    :param date: the date the orientation is requested
    :return: the rotation from the inertial frame to the camera frame
    """

    rate_x = 5*np.pi/180
    rate_y = -2.5*np.pi/180
    rate_z = 0.2*np.pi/180

    angle_vel = np.array([rate_x, rate_y, rate_z])

    dt = (date - EPOCH)/timedelta(seconds=1)
    start = np.array([20, -10, 30.])*180/np.pi

    return Rotation(euler_to_rotmat(start+dt*angle_vel))


def camera_orientation(date: datetime) -> Rotation:
    """
    Return the a priori camera orientation from the inertial frame to the camera frame at the requested time

    :param date: the date the orientation is requested
    :return: the rotation from the inertial frame to the camera frame
    """

    return Rotation(2e-8*np.random.randn(3))*_truth_camera_orientation(date)


def camera_position(date: datetime) -> np.ndarray:
    """
    Return the inertial position of the camera at the requested time.

    :param date: the date the position is requested
    :return: the position vector from the SSB to the camera in km in the inertial frame
    """

    truth = _truth_camera_position(date)
    return np.random.randn(3)*1e-3+truth


def _truth_relative_target_position_camera(date: datetime) -> np.ndarray:
    """
    Return the relative position between Itokawa and the camera

    :param date: the date the position is requested
    :return: the position vector from the camera to itokawa in the camera frame
    """

    # rates in km/sec
    rate_x = 0.001
    rate_y = -0.005
    rate_z = 0.02

    start = np.array([0, 0, 1.5])

    dt = (date - EPOCH)/timedelta(seconds=1)

    return start + dt*np.array([rate_x, rate_y, rate_z])


def _truth_relative_sun_position_camera(date: datetime) -> np.ndarray:
    """
    Return the relative position between the sun and the camera

    :param date: the date the position is requested
    :return: the position vector from the camera to the sun in the camera frame
    """

    # rates in km/sec
    rate_x = -0.0001
    rate_y = 0.00005
    rate_z = -0.2

    start = 1e10*np.array([np.sqrt(2), 0, -np.sqrt(2)])

    dt = (date - EPOCH)/timedelta(seconds=1)

    return start + dt*np.array([rate_x, rate_y, rate_z])


def _truth_sun_position(date: datetime) -> np.ndarray:
    """
    Return the true inertial location of the sun.

    :param date: the date the position is requested
    :return: the position vector from the SSB to the sun in the inertial frame
    """

    # relative truth in inertial frame
    relative_truth = _truth_camera_orientation(date).matrix.T@_truth_relative_sun_position_camera(date)

    # inertial truth in inertial frame
    return relative_truth + _truth_camera_position(date)


def sun_position(date: datetime) -> np.ndarray:
    """
    Return the a priori inertial location of the sun.

    .. note:: This is just a simulation position, it has not real physical correspondence

    :param date: the date the position is requested
    :return: the position vector from the SSB to the sun in the inertial frame
    """

    truth = _truth_sun_position(date)

    return np.random.randn(3)+truth


def sun_orientation(date: datetime) -> Rotation:
    """
    Return the rotation from the sun fixed frame to the inertial frame.

    .. note:: This is just a simulation frame, it has not real physical correspondence

    :param date: the date the orientation is requested
    :return: the rotation from the sun fixed frome to the inertial frame
    """

    date

    return Rotation([0, 0, 0, 1])


def _truth_target_position(date: datetime) -> np.ndarray:
    """
    Return the true inertial location of Itokawa.

    :param date: the date the position is requested
    :return: the position vector from the SSB to Itokawa in the inertial frame
    """

    # relative truth in inertial frame
    relative_truth = _truth_camera_orientation(date).matrix.T@_truth_relative_target_position_camera(date)

    # inerital truth in inertial frame
    return relative_truth + _truth_camera_position(date)


def target_position(date: datetime) -> np.ndarray:
    """
    Return the inertial position of Itokawa

    .. note::
        This has nothing to do with the actual location of Itokawa

    :param date: the date the position is requested
    :return: the position vector from the SSB to Itokawa in the inertial frame
    """

    # relative truth in inertial frame
    relative_truth = _truth_camera_orientation(date).matrix.T@_truth_relative_target_position_camera(date)

    # inerital location in inertial frame
    return (np.random.randn(3)*0.05*np.linalg.norm(relative_truth)+relative_truth) + _truth_camera_position(date)


def _truth_target_orientation(date: datetime) -> Rotation:
    """
    Return the orientation of Itokawa with respect to the inertial frame

    :param date: The date the orientation is desired
    :return: the rotation from Itokawa to the inertial frame
    """

    # set the rate in seconds
    rate_z = 2.0*np.pi/180

    return Rotation([0, 0, (date - EPOCH)/timedelta(seconds=1)*rate_z])


def target_orientation(date: datetime) -> Rotation:
    """
    Return the a priori knowledge of the orientation of Itokawa with respect to the inertial frame.

    .. note::
        This has nothing to do with the actual orientation frame of Itokawa. It is purely for simulation purposes.

    :param date: The data the orientation is desired
    :return: the rotation from Itokawa to the inertial frame
    """

    return Rotation(1e-10*np.random.randn(3))*_truth_target_orientation(date)


def _render_body(camera_to_render: Camera, scene_to_render: Scene):
    """
    Render a simulated image of a target for each image in camera.

    :param camera_to_render: The camera that is being rendered
    :param scene_to_render: The scene that is being rendered
    """
    xc = XCorrCenterFinding(scene_to_render, camera_to_render, ImageProcessing(), McEwenIllumination())

    camera_to_render.all_on()
    for ind, limg in camera_to_render:

        scene_to_render.update(limg)
        start = time.time()
        if limg.exposure_type == ExposureType.LONG:
            xc.grid_size = 1
        else:
            xc.grid_size = 2

        illum, pix = xc.render(0, scene_to_render.target_objs[0], temperature=limg.temperature)

        if limg.exposure_type == ExposureType.LONG:
            illum[illum != 0] = 2**15
        else:
            illum[~np.isfinite(illum)] = 0
            # noinspection PyArgumentList
            illum *= 2**14/(xc.grid_size**2)//illum.max()

        subs = pix.round().astype(int)

        np.add.at(limg, (subs[1], subs[0]), illum.ravel())
        if limg.exposure_type == ExposureType.SHORT:
            limg[:] = PSF(limg)

        print('rendered image {} of {} in {:.3f} secs'.format(ind+1, len(camera_to_render.images),
                                                              time.time()-start), flush=True)


def main() -> None:
    """
    Parses the command line arguments and generates the images.
    """
    # need to do this to ensure that the module name is right for pickle
    # noinspection PyUnresolvedReferences
    from giant.scripts.generate_sample_data import PSF

    parser = _get_parser()
    parser.parse_args()

    np.random.seed(87239)

    # first, get the current working directory and create the sample_data directory to store the results
    cwd = Path.cwd()

    # prepare the directory
    output_dir = cwd / "sample_data"
    output_dir.mkdir(exist_ok=True)

    # load or create the shape model
    shape_file = output_dir / "kdtree.pickle"

    kd = None
    if not shape_file.exists():
        resp = input('The shape model does not exist yet.  Can we download it (~60MB)? (y/n)  ')

        if resp.lower() == 'y':
            kd = _prepare_shape(output_dir)

            with shape_file.open('wb') as ofile:
                pickle.dump(kd, ofile)

        else:
            print('Unable to proceed', flush=True)
            exit(-1)

    else:
        with shape_file.open('rb') as ifile:
            # this is a hard coded file that is created by this program.  Someone would have to specifically change out
            # the file to execute arbitrary code so I think the risk is minimal here
            kd = pickle.load(ifile)  # nosec

    # make the camera
    camera = Camera(model=MODEL, parse_data=False, psf=PSF)

    # define the scene
    target = SceneObject(kd, position_function=_truth_target_position,
                         orientation_function=_truth_target_orientation, name='Itokawa')
    sun = SceneObject(Point(np.zeros(3)), position_function=_truth_sun_position,
                      orientation_function=sun_orientation, name='Sun')
    scene = Scene(target_objs=[target], light_obj=sun)

    # make the first stellar image
    img = 100 + 10*np.random.randn(1944, 2592)

    true_att = _truth_camera_orientation(EPOCH)

    oimg = OpNavImage(img, observation_date=EPOCH, exposure=5, exposure_type='long',
                      temperature=0, rotation_inertial_to_camera=true_att, position=_truth_camera_position(EPOCH),
                      velocity=np.array([0, 0, 0.]), saturation=2 ** 16)

    camera.add_images([oimg])

    # make the first short image
    img = 100 + 10*np.random.randn(1944, 2592)

    obs_date = EPOCH+timedelta(seconds=30)

    true_att = _truth_camera_orientation(obs_date)
    true_pos = _truth_camera_position(obs_date)

    oimg = OpNavImage(img, observation_date=obs_date, exposure=0.1, exposure_type='short',
                      temperature=0, rotation_inertial_to_camera=true_att, position=true_pos,
                      velocity=np.array([0, 0, 0.]), saturation=2 ** 16)

    camera.add_images([oimg])

    # make the second stellar image
    img = 100 + 10*np.random.randn(1944, 2592)

    obs_date = EPOCH+timedelta(minutes=1)

    true_att = _truth_camera_orientation(obs_date)
    true_pos = _truth_camera_position(obs_date)

    oimg = OpNavImage(img, observation_date=obs_date, exposure=5, exposure_type='long',
                      temperature=0, rotation_inertial_to_camera=true_att, position=true_pos,
                      velocity=np.array([0, 0, 0.]), saturation=2 ** 16)

    camera.add_images([oimg])

    # make the second short image
    img = 100 + 10*np.random.randn(1944, 2592)

    obs_date = EPOCH+timedelta(seconds=90)

    true_att = _truth_camera_orientation(obs_date)
    true_pos = _truth_camera_position(obs_date)

    oimg = OpNavImage(img, observation_date=obs_date, exposure=0.1, exposure_type='short',
                      temperature=0, rotation_inertial_to_camera=true_att, position=true_pos,
                      velocity=np.array([0, 0, 0.]), saturation=2 ** 16)

    camera.add_images([oimg])

    # make the third stellar image
    img = 100 + 10*np.random.randn(1944, 2592)

    obs_date = EPOCH+timedelta(minutes=2)

    true_att = _truth_camera_orientation(obs_date)
    true_pos = _truth_camera_position(obs_date)

    oimg = OpNavImage(img, observation_date=obs_date, exposure=5, exposure_type='long',
                      temperature=0, rotation_inertial_to_camera=true_att, position=true_pos,
                      velocity=np.array([0, 0, 0.]), saturation=2 ** 16)

    camera.add_images([oimg])

    # add stars to the image
    _render_stars(camera)

    # add itokawa to the image
    _render_body(camera, scene)

    # set the perturbed attitude as the apriori
    for _, img in camera:
        img.rotation_inertial_to_camera = camera_orientation(img.observation_date)

    ofile = output_dir / 'camera.pickle'

    with ofile.open('wb') as ofpt:
        pickle.dump(camera, ofpt, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    main()
