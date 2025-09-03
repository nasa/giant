


"""
This module implements functions for viewing the results of UFO identification.

Use
---

There is currently a single user function in this module, :func:`.show_detections`.  Simply supply the detections
DataFrame and the :class:`.Camera` object to this function and it will show the results for you to examine
(alternatively it can save the plots to files).  We hope to add more functionality to this module in the future.
"""

import os

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent

import pandas as pd

import numpy as np

from giant.camera import Camera


class _InteractiveDetectionExplorer:
    """
    Helper class for interacting with the figures.

    Not to be used externally
    """

    def __init__(self, figure: Figure, ax: Axes, ufos: pd.DataFrame, camera: Camera,
                 log_scale: bool = False):

        self.current_rc_params = rcParams.copy()

        for key in rcParams.keys():
            if key.startswith('keymap'):
                if 'left' in rcParams[key]:
                    print(f'Temporarily disabling left key for {key} mapping', flush=True)
                    rcParams[key].remove("left")
                if 'right' in rcParams[key]:
                    print(f'Temporarily disabling left key for {key} mapping', flush=True)
                    rcParams[key].remove("right")

        self.figure = figure
        self.ax = ax
        self.ufos = ufos
        self.camera = camera
        self.log_scale = log_scale

        self.valid_frames = np.argwhere(self.camera.image_mask).ravel()

        self.current_frame_index = 0

        self.colorbar: Optional[Colorbar] = None

        self.canvas_event_id = self.figure.canvas.mpl_connect('key_press_event', self._on_key_press) # pyright: ignore[reportArgumentType]

        self.figure.suptitle('Left => previous frame, Right => next frame')

        self.draw_frame()


    def _on_key_press(self, event: KeyEvent):

        if event.key == 'left':
            self.previous_frame()
        if event.key == 'right':
            self.next_frame()

    def __del__(self):
        rcParams.update(self.current_rc_params)
        self.figure.canvas.mpl_disconnect(self.canvas_event_id)

    def next_frame(self):
        self.current_frame_index += 1
        if self.current_frame_index >= self.valid_frames.size:
            self.current_frame_index = 0

        self.draw_frame()

    def previous_frame(self):
        self.current_frame_index -= 1
        if self.current_frame_index < 0:
            self.current_frame_index = self.valid_frames.size - 1

        self.draw_frame()

    def draw_frame(self):

        if self.colorbar is not None:
            self.colorbar.remove()

        self.ax.cla()

        image = self.camera.images[self.valid_frames[self.current_frame_index]]

        image_mask = self.ufos.image_file == os.path.splitext(os.path.basename(image.file))[0]

        image_ufos = self.ufos.loc[image_mask]

        if self.log_scale:
            # noinspection PyArgumentList
            image = image.astype(np.float32) - image.min() + 100

        self.ax.imshow(image, cmap='gray')

        scat = self.ax.scatter(image_ufos.x_raw, image_ufos.y_raw, c=image_ufos.quality_code)

        self.colorbar = self.figure.colorbar(scat)

        self.ax.set_title(image.observation_date.isoformat())

        self.figure.canvas.draw()


def show_detections(ufos: pd.DataFrame, camera: Camera, save_frames: bool = False, interactive: bool = True,
                    frame_output: str = './{}.png', log_scale: bool = False):
    """
    This function plots possible UFO detections for each turned on image in ``camera`` over top of the image.

    For each individual image, the ufo results contained in the ``ufos`` dataframe are plotted as a scatter plot colored
    by their quality code.  The image itself is displayed as normal, or possibly using a "log" scale to bring out dimmer
    features in the image.

    There are 3 different options for displaying these figures.  The first, if ``interactive`` is ``True`` shows the
    images/ufos in a single window where you can navigate from frame to frame using the left/right arrow keys.  This is
    the recommended way to view the results.

    The second option for displaying the figures is to save them directly to a file if ``save_frames`` is ``True``.
    The file the figures are saved to is controlled by the ``frame_output`` input which should be a string giving the
    path to save the files to as well as a format specifier {} to replace with the image file.

    The final, not-recommended option, is to show all of the figures at once by setting both ``save_frames`` and
    ``interactive`` to ``False``.  This will make a single figure window for each image and will show them all
    simultaneously.  This can use a ton of memory and is not recommended if you have more than 10 images you processed.

    :param ufos: A dataframe of the UFOs to show from :attr:`.Detector.detection_data_frame`.
    :param camera: The :class:`.Camera` containing the images that the detections were extracted from
    :param save_frames: A boolean flag specifying whether to save the figures to disk instead of displaying them
    :param interactive: A boolean flag specifying whether to interactively walk through the images
    :param frame_output: A string with format specifier describing where to save the figures
    :param log_scale: A boolean flag specifying to use a logarithmic scale to display the images, allowing dimmer things
                      to stand out more
    """


    if interactive:
        fig = plt.figure()

        ax = fig.add_subplot(111)

        explorer = _InteractiveDetectionExplorer(fig, ax, ufos, camera, log_scale=log_scale)

        plt.show()

        del explorer

    elif save_frames:
        fig: Figure = plt.figure()

        ax = fig.add_subplot(111)
        colorbar = None

        for _, image in camera:

            if colorbar is not None:
                colorbar.remove()

            ax.cla()
            assert image.file is not None
            image_file = os.path.splitext(os.path.basename(image.file))[0]

            image_mask = ufos.image_file == image_file

            image_ufos = ufos.loc[image_mask]

            if log_scale:
                # noinspection PyArgumentList
                image = image.astype(np.float32) - image.min() + 100

            ax.imshow(image, cmap='gray')

            scat = ax.scatter(image_ufos.x_raw, image_ufos.y_raw, c=ufos.quality_code)

            colorbar = fig.colorbar(scat)

            ax.set_title(image.observation_date.isoformat())

            fig.savefig(frame_output.format(image_file))

        plt.close(fig)

    else:

        for _, image in camera:
            if log_scale:
                # noinspection PyArgumentList
                image = image.astype(np.float32) - image.min() + 100
            fig: Figure = plt.figure()
            ax = fig.add_subplot()
            assert image.file is not None
            image_file = os.path.splitext(os.path.basename(image.file))[0]

            image_mask = ufos.image_file == image_file

            image_ufos = ufos.loc[image_mask]

            ax.imshow(image, cmap='gray')

            scat = ax.scatter(image_ufos.x_raw, image_ufos.y_raw, c=ufos.quality_code)

            fig.colorbar(scat)

            ax.set_title(image.observation_date.isoformat())

        plt.show()
