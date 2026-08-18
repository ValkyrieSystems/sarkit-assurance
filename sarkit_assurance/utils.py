import numpy as np
import shapely


def get_samples_in_poly(poly: shapely.Polygon, grid_size: int = 11) -> np.ndarray:
    """Return samples that intersect a polygon."""
    bounds = np.asarray(poly.bounds).reshape(2, 2)  # [[xmin, ymin], [xmax, ymax]]
    mesh = np.stack(
        np.meshgrid(
            np.linspace(bounds[0, 0], bounds[1, 0], grid_size),
            np.linspace(bounds[0, 1], bounds[1, 1], grid_size),
        ),
        axis=-1,
    )
    inner_mesh = shapely.get_coordinates(poly.intersection(shapely.multipoints(mesh)))
    poly_vertices = shapely.get_coordinates(poly.exterior)[:-1]
    return np.concatenate(
        [inner_mesh, poly_vertices],
        axis=0,
    )


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)
