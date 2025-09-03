


"""
This package provides the required routines and objects to identify UFOs in monocular images and track them from frame
to frame.

Description
___________
In GIANT, UFOs are unresolved bright spots in images that do not match to known stars or extended bodies in the scene.
In the UFO package, we attempt to autonomously identify these bright spots (through the same steps as for identifying
stars in :mod:`.stellar_opnav`) and then to autonomously track the bright spots from frame to frame using EKFs.

The :class:`.UFO` class is generally the only interface a user will require when doing UFO identification and tracking,
as it provides easy access to every component you need.  It abstracts away most of the nitty gritty details into a few
simple method calls :meth:`~.UFO.detect`, :meth:`~.UFO.track`, :meth:`~.UFO.save_results`, and
:meth:`~.UFO.visualize_detection_results` which means you can do autonomous detection and tracking without an in depth
understanding of what's going on in the background (though at least a basic understanding certainly helps).  That being
said, the substeps are also exposed throughout GIANT, so if you are doing advanced design or analysis it is easy to get
these components as well.

This package level documentation only focuses on the use of the class and some techniques for successfully performing
UFO detection and tracking (in the upcoming tuning section).  To see behind the scenes of what's going on, refer to the
submodule documentations from this package.

Tuning for Successful Detection and Tracking
____________________________________________
The process of tuning the UFO detection routines is very similar to tuning the Stellar OpNav routines except we are
looking for unmatched bright spots instead of trying to minimize these things.  There are actually 2 passes for doing
detection.  The first pass is to correct the attitude for th images and thus should be tuned conservatively as you
normally would for star identification (see :mod:`.stellar_opnav` for details).  The second pass, after the attitude has
been estimated is generally attempting to identify both more stars and more unmatched image points, therefore we
generally turn off the RANSAC (doing essentially a nearest neighboor search) and allow many more stars and many more
points to be considered (while not going over the top).

Once you have a good set of UFO detections (not missing real detections and not having excessive amounts of spurious
points) then you can move on to tracking.  Tuning for tracking can be somewhat more involved.  It involves setting up
the :class:`.Dynamics` model to appropriately model the dynamics that are working on the UFOs which is too general a
topic to cover here.  It also requires figuring out a good intial guess for the initial state for the UFOs based off of
the first observation of them, which again is too broad of a topic to cover here (though you can see an example of in
:mod:`.state_initializer`).  Then you need to specify the search distance function, which specifies the euclidean search
distance to use when pairing points in subsequent frames to the predicted locations of the points based off of the
previous frames, which you generally want to be fairly open (depending on the dynamics of the UFO's/detectors) but not
so open that you overwhelm the tracker with too many paths forward. Beyond that, most of the rest of the tuning
parameters are fairly straight forward and are primarily focused on memory management issues.  Because of the range
of different cases that you may be tracking in, the best way to figure out a tuning is through understanding what the
tracker is doing (there is a paper at https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019EA000843 which goes
into details) and understanding the environment you are dealing with.
"""
