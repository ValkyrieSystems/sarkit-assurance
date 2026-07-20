import sarkit.sicd as sksicd
import shapely


def get_validdata_polygon(sicd_ew: sksicd.ElementWrapper) -> shapely.Polygon:
    """Return a polygon describing a SICD's valid data in global row/column coordinates"""
    vertices = sicd_ew["ImageData"].get("ValidData", None)
    if vertices is None:
        # Use edges of full image
        nrows = sicd_ew["ImageData"]["FullImage"]["NumRows"]
        ncols = sicd_ew["ImageData"]["FullImage"]["NumCols"]

        vertices = [(0, 0), (0, ncols - 1), (nrows - 1, ncols - 1), (nrows - 1, 0)]

    return shapely.Polygon(vertices)
