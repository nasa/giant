


"""
UNDER CONSTRUCTION?!?!?!?

The :mod:`.visualizer` module provides...

Contents
--------
"""

import matplotlib.pyplot as plt

try:
    from matplotlib.animation import ImageMagickWriter
except UnicodeDecodeError:
    print('unable to import ImageMagickWriter')
    ImageMagickWriter = None

import numpy as np

ANGLES = np.linspace(0, 2 * np.pi, 500)
SCAN_VECTORS = np.vstack([np.cos(ANGLES), np.sin(ANGLES), np.zeros(ANGLES.size)])


def show_templates(relnav, index, target_ind, ax1=None, ax2=None, fig=None):
    # retrieve the image we are processing
    image = relnav.camera.images[index]

    # update the scene to reflect the current time
    relnav.scene.update(image)

    if (ax1 is None) or (ax2 is None) or (fig is None):
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    # set the title so we know what we're looking at
    fig.suptitle('{} {}'.format(image.observation_date.isoformat(), relnav.scene.target_objs[target_ind].name))

    # show the image
    ax1.imshow(image, cmap='gray')

    # determine the location of the template in the image (roughly)
    template_shape = np.array(relnav.saved_templates[index][target_ind].shape[::-1])
    template_size = template_shape // 2
    center = np.round(relnav.center_finding_results[index, target_ind]["measured"][:2])
    if not np.isfinite(center).all():
        print('invalid solved-for center.  using predicted.')
        center = np.round(relnav.center_finding_results[index, target_ind]["predicted"][:2])

    min_bounds = center - template_size
    max_bounds = center + template_size + (template_shape % 2)

    # crop the image, accounting for when the shape is odd using the modulo
    ax1.set_xlim(min_bounds[0], max_bounds[0])
    ax1.set_ylim(max_bounds[1], min_bounds[1])

    # label this subplot as the image
    ax1.set_title('Image')

    # show the template
    ax2.imshow(relnav.saved_templates[index][target_ind], cmap='gray')
    # label this subplot as the template
    ax2.set_title('Template')

    return ax1, ax2, fig


def show_limbs(relnav, index, ax=None):
    # retrieve the image we are processing
    image = relnav.camera.images[index]

    # retrieve the observation observation_date of the image we are processing
    date = image.observation_date

    # update the scene to reflect the current time
    relnav.scene.update(image)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.imshow(image, cmap='gray')

    # initialize variables to store the bounds of the limbs
    min_limb_bounds = [np.inf, np.inf]
    max_limb_bounds = [-np.inf, -np.inf]

    for target_ind, target in enumerate(relnav.scene.target_objs):

        # determine the a priori distance to the target
        apriori_distance = np.linalg.norm(target.position)

        # get the a priori limb points
        # define the line of sight to the body in the camera frame
        apriori_los = target.position.ravel() / apriori_distance

        # find the limb points in the camera frame
        apriori_limbs_cam = target.shape.find_limbs(apriori_los, SCAN_VECTORS)

        # project the limb points into the image
        apriori_limbs_image = relnav.camera.model.project_onto_image(apriori_limbs_cam,
                                                                     image=index,
                                                                     temperature=image.temperature)

        # plot the a priori limb points
        ax.plot(*apriori_limbs_image, linewidth=1, label='{} a priori limbs'.format(target.name))

        # adjust the target object to its observed location
        rtype = relnav.center_finding_results[index, target_ind]['type']
        if not rtype:
            rtype = relnav.relative_position_results[index, target_ind]['type']
        if rtype in [b'cof', 'cof']:
            los = relnav.camera.model.pixels_to_unit(relnav.center_finding_results[index, target_ind]['measured'][:2],
                                                     temperature=image.temperature, image=index)
            if np.isfinite(los).all():
                target.change_position(los * apriori_distance)

        elif rtype in [b'pos', 'pos']:
            los = relnav.relative_position_results[index, target_ind]['measured'].copy()
            los /= np.linalg.norm(los)
            if np.isfinite(los).all():
                target.change_position(relnav.relative_position_results[index, target_ind]['measured'])

        else:
            raise ValueError("Can't display limbs for {} type relnav".format(rtype))

        limbs_cam = target.shape.find_limbs(los, SCAN_VECTORS)

        # project the limb points into the image
        limbs_image = relnav.camera.model.project_onto_image(limbs_cam,
                                                             image=index,
                                                             temperature=image.temperature)

        # update the limb bounds
        min_limb_bounds = np.minimum(min_limb_bounds, limbs_image.min(axis=-1))
        max_limb_bounds = np.maximum(max_limb_bounds, limbs_image.max(axis=-1))

        # plot the updated limb points
        ax.plot(*limbs_image, linewidth=1, label='{} solved for limbs'.format(target.name))

        if rtype in [b'cof', 'cof']:
            # show the predicted center pixel
            ax.scatter(*relnav.center_finding_results[index, target_ind]['predicted'][:2],
                       label='{} predicted center'.format(target.name))
            # show the solved for center
            ax.scatter(*relnav.center_finding_results[index, target_ind]['measured'][:2],
                       label='{} solved-for center'.format(target.name))

        else:
            # get the apriori image location
            apriori_image_pos = relnav.camera.model.project_onto_image(
                relnav.relative_position_results[index, target_ind]['predicted'],
                image=index, temperature=image.temperature)

            # get the solved for image location
            image_pos = relnav.camera.model.project_onto_image(
                relnav.relative_position_results[index, target_ind]['measured'],
                image=index, temperature=image.temperature)

            # show the centers
            ax.scatter(*apriori_image_pos, label='{} predicted center'.format(target.name))
            ax.scatter(*image_pos, label='{} solved-for center'.format(target.name))

    # set the title so we know what image we're looking at
    ax.set_title(date.isoformat())

    return ax, max_limb_bounds, min_limb_bounds


def limb_summary_gif(relnav, fps=2, outfile='./opnavsummary.gif', dpi=100):
    # initialize the figure and axes
    fig = plt.figure()
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)

    # initialize the writer
    writer = ImageMagickWriter(fps=fps)
    writer.setup(fig=fig, outfile=outfile, dpi=dpi)

    # loop through each image and save the frame
    for ind, image in relnav.camera:

        ax.clear()

        _, max_limbs, min_limbs = show_limbs(relnav, ind, ax=ax)

        # set the limits to highlight only the portion of interest
        if np.isfinite(min_limbs).all() and np.isfinite(max_limbs).all():
            ax.set_xlim(min_limbs[0] - 10, max_limbs[0] + 10)
            ax.set_ylim(min_limbs[1] - 10, max_limbs[1] + 10)

        try:
            fig.legend().draggable()
        except AttributeError:
            fig.legend().set_draggable(True)

        writer.grab_frame()

    writer.finish()
    plt.close(fig)


def template_summary_gif(relnav, fps=2, outfile='./templatesummary.gif', dpi=100):
    # initialize the figure and axes
    fig = plt.figure()
    fig.set_tight_layout(True)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # initialize the writer
    writer = ImageMagickWriter(fps=fps)
    writer.setup(fig=fig, outfile=outfile, dpi=dpi)

    # loop through each image and save the frame
    for ind, image in relnav.camera:

        # loop through each object
        for obj_ind in range(len(relnav.scene.target_objs)):

            if relnav.saved_templates[ind] is not None:
                if relnav.saved_templates[ind][obj_ind] is not None:
                    ax1.clear()
                    ax2.clear()

                    show_templates(relnav, ind, obj_ind, ax1=ax1, ax2=ax2, fig=fig)

                    writer.grab_frame()

    writer.finish()

    plt.close(fig)


def show_center_finding_residuals(relnav):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # initialize lists to store the data
    resids = []
    dates = []

    # loop through each image
    for ind, image in relnav.camera:

        # store a list in the resids list and the datetime object in the dates list
        resids.append([])
        dates.append(image.observation_date)

        # loop through each target
        for target_ind in range(len(relnav.scene.target_objs)):

            # determine the type of results we are considering
            if relnav.center_finding_results[ind, target_ind]["type"] in [b'cof', b'lmk', 'cof', 'lmk']:

                # compute the observed minus computed residuals
                resids[-1].append((relnav.center_finding_results[ind, target_ind]['measured'] -
                                   relnav.center_finding_results[ind, target_ind]['predicted'])[:2])

            else:  # if we are considering a technique that estimates the full 3DOF position
                # project the predicted and measured positions onto the image and compute the o-c resids
                resids[-1].append(
                    relnav.camera.model.project_onto_image(relnav.center_finding_results[ind, target_ind]['measured'],
                                                           image=ind, temperature=image.temperature) -
                    relnav.camera.model.project_onto_image(relnav.center_finding_results[ind, target_ind]['predicted'],
                                                           image=ind, temperature=image.temperature))

    # stack all of the resids together
    resids = np.asarray(resids)
    dates = np.asarray(dates, dtype='datetime64[us]')

    # loop through each target again and plot the residuals vs time
    for target_ind, target in enumerate(relnav.scene.target_objs):
        ax.scatter(dates, resids[:, target_ind, 0], label='{} Columns'.format(target.name))
        ax.scatter(dates, resids[:, target_ind, 1], label='{} Rows'.format(target.name))

    # set the labels and update the x axis to display the dates better
    ax.set_xlabel('Observation Date')
    ax.set_ylabel('O-C Residuals, pix')
    fig.autofmt_xdate()

    # create a legend
    try:
        fig.legend().draggable()
    except AttributeError:
        fig.legend().set_draggable(True)


def scatter_residuals_sun_dependent(relnav):
    """
    Show observed minus computed residuals with units of pixels plotted in a frame rotated so that +x points towards the
    sun in the image.
    :param relnav:
    """

    resids = []
    # loop through each image
    for ind, image in relnav.camera:

        # loop through each target
        for target_ind in range(len(relnav.scene.target_objs)):

            # update the scene so we can get the sun direction
            relnav.scene.update(image)

            # figure out the direction to the sun in the image
            line_of_sight_sun = relnav.camera.model.project_directions(relnav.scene.light_obj.position.ravel())

            # get the rotation to make the x axis line up with this direction
            # since line_of_sight_sun is a unit vector the x component = cos(theta) and y component = sin of theta
            # so the following gives
            # [[ cos(theta), sin(theta)],
            #  [-sin(theta), cos(theta)]]
            # which rotates from the image frame to the frame with +x pointing towards the sun
            rmat = np.array([line_of_sight_sun, line_of_sight_sun[::-1] * [-1, 1]])

            # determine the type of results we are considering
            if relnav.center_finding_results[ind, target_ind]["type"] in [b'cof', b'lmk', 'cof', 'lmk']:

                # compute the observed minus computed residuals
                resids.append(rmat @ (relnav.center_finding_results[ind, target_ind]['measured'] -
                                      relnav.center_finding_results[ind, target_ind]['predicted'])[:2])

            else:  # if we are considering a technique that estimates the full 3DOF position
                # project the predicted and measured positions onto the image and compute the o-c resids
                resids.append(rmat @
                              (relnav.camera.model.project_onto_image(
                                  relnav.center_finding_results[ind, target_ind]['measured'],
                                  image=ind, temperature=image.temperature
                              ) -
                               relnav.camera.model.project_onto_image(
                                   relnav.center_finding_results[ind, target_ind]['predicted'],
                                   image=ind, temperature=image.temperature)
                               ))

    # stack all of the resids together
    resids = np.asarray(resids)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(*resids)
    ax.set_xlabel("Sun direction O-C error, pix")
    ax.set_xlabel("Anti-sun direction O-C error, pix")


def plot_residuals_sun_dependent_time(relnav):
    """
    Show observed minus computed residuals with units of pixels plotted in a frame rotated so that +x points towards the
    sun in the image.

    This is done with a time series (so the x axis of the plot is time and the y axis is residual in pixels) with 2
    different series
    :param relnav:
    """

    dates = []
    resids = []

    # loop through each image
    for ind, image in relnav.camera:

        # store a list in the resids list and the datetime object in the dates list
        resids.append([])
        dates.append(image.observation_date)

        # loop through each target
        for target_ind in range(len(relnav.scene.target_objs)):

            # update the scene so we can get the sun direction
            relnav.scene.update(image)

            # figure out the direction to the sun in the image
            line_of_sight_sun = relnav.camera.model.project_directions(relnav.scene.light_obj.position.ravel())

            # get the rotation to make the x axis line up with this direction
            # since line_of_sight_sun is a unit vector the x component = cos(theta) and y component = sin of theta
            # so the following gives
            # [[ cos(theta), sin(theta)],
            #  [-sin(theta), cos(theta)]]
            # which rotates from the image frame to the frame with +x pointing towards the sun
            rmat = np.array([line_of_sight_sun, line_of_sight_sun[::-1] * [-1, 1]])

            # determine the type of results we are considering
            if relnav.center_finding_results[ind, target_ind]["type"] in [b'cof', b'lmk', 'cof', 'lmk']:

                # compute the observed minus computed residuals
                resids[-1].append(rmat @ (relnav.center_finding_results[ind, target_ind]['measured'] -
                                      relnav.center_finding_results[ind, target_ind]['predicted'])[:2])

            else:  # if we are considering a technique that estimates the full 3DOF position
                # project the predicted and measured positions onto the image and compute the o-c resids
                resids[-1].append(rmat @
                              (relnav.camera.model.project_onto_image(
                                  relnav.center_finding_results[ind, target_ind]['measured'],
                                  image=ind, temperature=image.temperature
                              ) -
                               relnav.camera.model.project_onto_image(
                                   relnav.center_finding_results[ind, target_ind]['predicted'],
                                   image=ind, temperature=image.temperature)
                               ))

    # stack all of the resids together
    resids = np.asarray(resids)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # loop through each target again and plot the residuals vs time
    for target_ind, target in enumerate(relnav.scene.target_objs):
        ax.scatter(dates, resids[:, target_ind, 0], label='{} Sun direction'.format(target.name))
        ax.scatter(dates, resids[:, target_ind, 1], label='{} Anti sun direction'.format(target.name))

    # set the labels and update the x axis to display the dates better
    ax.set_xlabel('Observation Date')
    ax.set_ylabel('O-C Residuals, pix')
    fig.autofmt_xdate()

    # create a legend
    try:
        fig.legend().draggable()
    except AttributeError:
        fig.legend().set_draggable(True)

