# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
Convert SPC Autoregister results into a GIANT results pickle file format.

This script converts SPC Autoregister results into a format GIANT can understand.  This is
useful primarily for using GIANT to display the results and for comparison between GIANT SFN and SPC observations.
It makes extensive use of the :mod:`.stereophotoclinometry` module, so if you need to do work with SPC products beyond
what this scripts provides look there.

For this script to run you need to have already run the Autoregister portion of SPC on your images.  You also need to be
sure that you have a giant camera file (as a dill or pickle file) which contains images that overlap with the images in
the SPC directory.

Images are paired based on timestamp without regard for which camera was used, as SPC uses a way to identify cameras
that is difficult to generalize to the GIANT convention.  Therefore you need to be careful if you have multiple cameras
processed in your SPC directory that have image times that are very close together.

.. warning::

    This script load/saves some results from/to python pickle files.  Pickle files can be used to execute arbitrary
    code, so you should never open one from an untrusted source.
"""

from glob import glob
from argparse import ArgumentParser

# added warning to documentation
import pickle  # nosec

from os.path import join

from datetime import timedelta

from warnings import warn

# added warning to documentation
import dill  # nosec

import numpy as np

import spiceypy as spice

import cv2

from giant.relative_opnav.relnav_class import RESULTS_DTYPE
from giant.utilities.stereophotoclinometry import Nominal, Summary, Maplet
from giant.utilities.spice_interface import (et_callable_to_datetime_callable, create_callable_position,
                                             create_callable_orientation)
from giant.ray_tracer.scene import correct_light_time, correct_stellar_aberration
from giant.rotations import Rotation
from giant.camera import Camera


MAPLETS = {}


# TODO: pull down updates from orex-nav

def _get_parser() -> ArgumentParser:
    """
    Helper function for the argparse extension

    :return: A setup argument parser
    """

    warning = "WARNING: This script loads/saves some results from/to python pickle files.  " \
              "Pickle files can be used to execute arbitrary code, " \
              "so you should never open one from an untrusted source."

    parser = ArgumentParser(description='Form a GIANT results file from an SPC directory',
                            epilog=warning)

    parser.add_argument('-d', '--dir', help='The SPC directory to extract the results from',
                        type=str, default='./')
    parser.add_argument('-c', '--camera', help='The camera file containing the images that were processed',
                        type=str, default='../giant/camera.dill')
    parser.add_argument('-o', '--output', help='The file to save the results to',
                        type=str, default='./results.pickle')
    parser.add_argument('-s', '--sumfiles', help='A list of sum files to extract the results from',
                        default=None, type=str, nargs='+')
    parser.add_argument('-m', '--meta_kernel', help='The meta kernel to furnish for the predicted maplet locations',
                        default='../meta_kernel.tm', type=str)
    parser.add_argument('-t', '--target', help='The spice target that the spc landmarks are on',
                        default='BENNU', type=str)
    parser.add_argument('-f', '--tframe', help='The spice target frame that the spc landmarks are on',
                        default='IAU_BENNU', type=str)
    parser.add_argument('-l', '--lmk_displays', help='The path to the landmark display files',
                        default=None, type=str, nargs='+')

    return parser


# noinspection PyTypeChecker
def main():
    parser = _get_parser()

    args = parser.parse_args()

    spice.furnsh(args.meta_kernel)

    targ_pos = et_callable_to_datetime_callable(create_callable_position(args.target, 'J2000', 'NONE', 'SSB'))
    rot_targ2j2000 = et_callable_to_datetime_callable(create_callable_orientation(args.tframe, 'J2000'))

    with open(args.camera, 'rb') as cfile:
        # added warning to documentation
        camera: Camera = dill.load(cfile)  # nosec

    if args.sumfiles is None:
        sumfiles = sorted(glob(join(args.dir, 'SUMFILES', '*.SUM')))

    else:
        sumfiles = sorted(args.sumfiles)

    with open(join(args.dir, 'shape_info.txt'), 'r') as ifile:
        name = ifile.readline().strip()
        for line in ifile:
            if 'Pole:' in line:
                pck = line.split(':')[1].strip()
                print('Loading pck {}'.format(pck))
                spice.furnsh(pck)
                break

    if args.lmk_displays is None:
        lmk_displays = sorted(glob(join(args.dir, 'TESTFILES1', '*.pgm')))

    else:
        lmk_displays = sorted(args.lmk_displays)

    sumos: list[Summary] = []
    nomos: list[Nominal] = []
    lmkds = []

    for sumf, lmkf in zip(sumfiles, lmk_displays):

        sumo = Summary(file_name=sumf)
        nomo = Nominal(file_name=sumf.replace('SUMFILES', 'NOMINALS').replace('.SUM', '.NOM'))

        for mletname in sumo.landmarks.keys():

            if mletname not in MAPLETS.keys():

                MAPLETS[mletname] = Maplet(file_name=join(args.dir, 'MAPFILES', mletname + '.MAP'))
    
        sumos.append(sumo)
        nomos.append(nomo)
        lmkds.append(cv2.imread(lmkf, cv2.IMREAD_GRAYSCALE))

    im_times = list(map(lambda x: x.observation_date, camera.images))

    res = [None]*len(im_times)
    details = [None]*len(im_times)
    lmk_disp = [None] * len(im_times)

    bounds = timedelta(seconds=1)

    for sumo, nomo, lmkd in zip(sumos, nomos, lmkds):
        lmk_res = []

        diff_list = list(map(lambda x: abs(x - sumo.observation_date), im_times))

        # print(diff_list)

        iind = int(np.argmin(diff_list))

        if diff_list[iind] > bounds:
            
            warn('Could not pair spc sum time to time in camera for spc file sum {}'.format(sumo.image_name))

            continue

        gimage = camera.images[iind]

        lmk_details = [{'PnP Solution': True,
                        'PnP Translation': sumo.rotation_target_fixed_to_camera @ nomo.rotation_target_fixed_to_camera.T @ sumo.position_camera_to_target - nomo.position_camera_to_target,
                        'PnP Rotation': Rotation(sumo.rotation_target_fixed_to_camera @ nomo.rotation_target_fixed_to_camera.T),
                        'PnP Position': sumo.rotation_target_fixed_to_camera @ sumo.position_camera_to_target,
                        'PnP Orientation': Rotation(sumo.rotation_target_fixed_to_camera),
                        'Correlation Scores': [0]*len(sumo.landmarks.keys())}]


        gmod = camera.model

        with open(sumo.file_name.replace('SUMFILES/', '').replace('.SUM', '.OOT'), 'r') as ootfile:

            oots = ootfile.readlines()

        for ind, mletname in enumerate(sumo.landmarks.keys()):
            mlet = MAPLETS[mletname]

            # do the projection to get the predicted location
            # define the position vector from the SSB to the maplet at a given time in the inertial frame
            def lmk_pos(date):
                return targ_pos(date) + rot_targ2j2000(date).matrix @ mlet.position_objmap

            # determine the position vector from the camera to the maplet at the image time in the inertial frame
            cam2lmkpos_inertial = correct_light_time(lmk_pos, gimage.position, gimage.observation_date)
            cam2lmkpos_inertial = correct_stellar_aberration(cam2lmkpos_inertial, gimage.velocity)
        
            # determine the position vector from the camera to the maplet at the image time in the camera frame 
            cam2lmkpos_camera = gimage.rotation_inertial_to_camera.matrix @ cam2lmkpos_inertial

            # project the lmk position vector onto the image to get the predicted location
            pred = gmod.project_onto_image(cam2lmkpos_camera.ravel(), temperature=gimage.temperature)

            # get the observed lmk position vector
            # need to account for fact that SPC uses 1 based indexing for images
            obs = np.array(sumo.landmarks[mletname]) - 1

            # store the results
            lmk_res.append((np.hstack([pred, [0]]), np.hstack([obs, [0]]), 'lmk', gimage.observation_date, ind,
                            mlet.position_objmap, mletname, name))

            # get the correlation score
            for oline in reversed(oots):
                if mletname in oline:
                    soline = oline.split()
                    if len(soline) == 5:
                        lmk_details[0]['Correlation Scores'][ind] = float(soline[-1])
                    elif len(soline) == 6:
                        lmk_details[0]['Correlation Scores'][ind] = float(soline[-2])
                    else:
                        print('bad')
                    break

        res[iind] = [np.array(lmk_res, dtype=RESULTS_DTYPE)]
        lmk_disp[iind] = lmkd
        details[iind] = lmk_details

    results_dict = {'lmk': res, 'lmk displays': lmk_disp, 'lmk details': details}

    with open(args.output, 'wb') as rfile:
        # added warning to documentation
        pickle.dump(results_dict, rfile)  # nosec

    spice.kclear()


if __name__ == '__main__':

    main()
