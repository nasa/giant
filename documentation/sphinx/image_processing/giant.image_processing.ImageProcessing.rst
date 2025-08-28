ImageProcessing
===============

.. currentmodule:: giant.image_processing

:mod:`giant.image_processing`\:

.. autoclass:: ImageProcessing
    :no-members:
    :members: centroiding, save_psf, image_denoising, subpixel_method, zernike_edge_width, denoising_args,
              denoising_kwargs, denoise_flag, correlator, correlator_kwargs, pae_threshold, pae_order, centroid_size,
              poi_threshold, poi_min_size, poi_max_size, reject_saturation, return_stats,
              image_flattening_noise_approximation, flattening_kernel_size

.. rubric:: Summary of Methods

.. autosummary::
    :nosignatures:
    :toctree:

    ~ImageProcessing.flatten_image_and_get_noise_level
    ~ImageProcessing.corners_to_roi
    ~ImageProcessing.find_poi_in_roi
    ~ImageProcessing.refine_locations
    ~ImageProcessing.locate_subpixel_poi_in_roi
    ~ImageProcessing.denoise_image
    ~ImageProcessing.correlate
    ~ImageProcessing.identify_subpixel_limbs
    ~ImageProcessing.identify_pixel_edges
    ~ImageProcessing.pae_edges
    ~ImageProcessing.refine_edges_pae
    ~ImageProcessing.refine_edges_zernike_ramp

|
