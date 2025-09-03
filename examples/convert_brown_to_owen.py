import numpy as np

from giant.camera_models import load, BrownModel, OwenModel, save

from giant.calibration.calibration_class import Calibration

from giant.camera import Camera
from giant.image import OpNavImage

from giant.rotations import Rotation


if __name__ == "__main__":

    existing_model_file = "/path/to/camera_model.xml"
    existing_model_name = "NameOfModel"

    brown: BrownModel = load(existing_model_file, existing_model_name)

    # brown: BrownModel = BrownModel(
    #     fx=1000, fy=1000, px=512.5, py=512.5, k1=1.2e-3, n_cols=1024, n_rows=1024,)

    owen: OwenModel = OwenModel(focal_length=1, kx=brown.fx, ky=brown.fy, px=brown.px, py=brown.py, estimation_parameters=[
                                'focal_length', 'ky', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'])

    calib_points_x, calib_points_y = np.meshgrid(
        np.arange(0, brown.n_cols, 5), np.arange(0, brown.n_rows, 5))

    calib_points = np.vstack((calib_points_x.ravel(), calib_points_y.ravel())).astype(np.float64)

    calib_units = brown.pixels_to_unit(calib_points)

    image = OpNavImage(np.zeros((brown.n_rows, brown.n_cols)), temperature=0.0,
                       rotation_inertial_to_camera=Rotation([0, 0, 0, 1.0]))

    camera = Camera([image], owen)

    calib = Calibration(camera)

    calib._matched_extracted_image_points = [calib_points]

    calib._matched_catalog_unit_vectors_camera = [calib_units]
    calib._matched_catalog_unit_vectors_inertial = [calib_units]
    calib._matched_catalog_image_points = [
        owen.project_onto_image(calib_units)]
    calib._queried_catalog_unit_vectors = [calib_units]
    calib._queried_catalog_image_points = [
        owen.project_onto_image(calib_units)]
    calib._unmatched_catalog_unit_vectors = [np.array([0, 0, 0])]

    print(calib.matched_star_residuals(0).mean(axis=1),
          calib.matched_star_residuals(0).std(axis=1))

    calib.estimate_geometric_calibration()

    print(calib.matched_star_residuals(0).mean(axis=1),
          calib.matched_star_residuals(0).std(axis=1))

    print(calib.model)

    save(existing_model_file, existing_model_name+"Owen", calib.model)
