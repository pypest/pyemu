import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, solve
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib as mpl

mpl.use('Tkagg')
import matplotlib.pyplot as plt
import os


# ============================================================================
# UTILITY FUNCTIONS FOR CONSISTENT 2D ARRAY HANDLING
# ============================================================================

def standardize_input_arrays(zones_raw, grid_coords, shape):
    """
    Standardize all input arrays to consistent 2D format for internal processing.

    Parameters
    ----------
    zones_raw : np.ndarray or None
        Raw zone array, shape (n_points,) or (ny, nx) or None
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    shape : tuple, actaully required here
        Grid shape (ny, nx) for single layer or (nz, ny, nx) for multilayer

    Returns
    -------
    tuple
        (zones, grid_coords_2d, original_zones_shape) where:
        - zones: Always 2D array (ny, nx) or None
        - grid_coords_2d: Grid coordinates for 2D processing
        - original_zones_shape: Original shape for output conversion
    """
    if len(shape) == 3:
        nz, ny, nx = shape
        layer_shape = (ny, nx)
    else:
        layer_shape = shape

    # Handle zones
    zones = None
    original_zones_shape = None

    if zones_raw is not None:
        original_zones_shape = zones_raw.shape
        ny, nx = layer_shape

        if zones_raw.shape == layer_shape:
            zones = zones_raw.copy()
        elif zones_raw.shape == (ny * nx,):
            zones = zones_raw.reshape(layer_shape)
        else:
            raise ValueError(f"Zone array shape {zones_raw.shape} incompatible with grid shape {layer_shape}")

    return zones, grid_coords, original_zones_shape


def convert_output_to_original_format(field_2d, original_zones_shape):
    """
    Convert 2D field back to original input format.

    Parameters
    ----------
    field_2d : np.ndarray
        Field in 2D format, shape (ny, nx)
    original_zones_shape : tuple or None
        Original input array shape

    Returns
    -------
    np.ndarray
        Field in original format (1D or 2D)
    """
    if original_zones_shape is None:
        return field_2d

    if len(original_zones_shape) == 1:
        # Original was 1D, return flattened
        return field_2d.flatten()
    else:
        # Original was 2D, return as-is
        return field_2d


def ensure_2d_for_processing(arr, shape):
    """
    Ensure array is in 2D format for internal processing.
    Simpler version focused on internal consistency.

    Parameters
    ----------
    arr : np.ndarray
        Input array
    shape : tuple
        Target 2D shape (ny, nx)

    Returns
    -------
    np.ndarray
        2D array with shape (ny, nx)
    """
    ny, nx = shape

    if arr.shape == shape:
        return arr
    elif arr.shape == (ny * nx,):
        return arr.reshape(shape)
    else:
        raise ValueError(f"Array shape {arr.shape} cannot be converted to 2D shape {shape}")


def validate_grid_coordinates(grid_coords, shape):
    """
    Validate grid coordinates match expected shape.

    Parameters
    ----------
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    shape : tuple
        Expected grid shape (nz, ny, nx) or (ny, nx)

    Returns
    -------
    bool
        True if valid
    """
    if len(shape) == 3:
        nz, ny, nx = shape
        expected_points = nz * ny * nx
    else:
        ny, nx = shape
        expected_points = ny * nx

    return grid_coords.shape[0] == expected_points


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(conceptual_points_file, grid_file, zone_file=None, iids_file=None,
         field_name=['field'], layer_mode=True, save_path='.'):
    """
    Generate spatially correlated fields using tensor-based geostatistics.

    This is the main entry point for tensor-based field generation. It reads
    conceptual points (control points) with anisotropy parameters and generates
    correlated fields on a specified grid.

    Parameters
    ----------
    conceptual_points_file : str
        Path to CSV file containing conceptual points with columns:
        x, y, major, anisotropy, bearing, mean_{field_name}, sd_{field_name}
        Optional: layer (for layer-by-layer processing)
    grid_file : str
        Path to whitespace-delimited file with grid coordinates (columns: x, y, z)
    zone_file : str, optional
        Path to whitespace-delimited array file with zones
    field_name : list of str, default ['field']
        Name(s) of field(s) to generate. Must match column prefixes in conceptual_points
    layer_mode : bool, default True
        If True, process each layer independently (recommended for 3D grids)

    Returns
    -------
    None
        Saves results to output/ directory as layer files and plots

    Examples
    --------
    >>> main('my_points.csv', 'my_grid.dat', field_name=['kh'], layer_mode=True)
    """
    print("=== Tensor-Based Non-Stationary Field Generation ===")

    # Load data
    conceptual_points = pd.read_csv(conceptual_points_file)
    grid = pd.read_csv(grid_file, sep=r'\s+')
    grid_coords = grid[['x', 'y', 'z']].values

    print(f"Loaded {len(conceptual_points)} conceptual points")
    print(f"Loaded {len(grid_coords)} grid points")

    # Infer grid structure FIRST
    shape = infer_grid_shape(grid_coords)
    print(f"Grid shape: {shape} (nz, ny, nx)")

    if iids_file is not None:
        os.path.join(os.path.dirname(conceptual_points_file), 'iids.arr')
        iids = np.loadtxt(iids_file)
    else:
        print("  Generating correlated noise...")
        np.random.seed(42)
        iids = np.random.randn(*shape)
        for i in range(iids.shape[0]):
            iids_file = os.path.join(os.path.dirname(conceptual_points_file), f'iids_layer{i+1}.arr')
            np.savetxt(iids_file, iids[i])

    # Validate grid coordinates
    if not validate_grid_coordinates(grid_coords, shape):
        raise ValueError(f"Grid coordinates ({len(grid_coords)} points) don't match inferred shape {shape}")

    # NOW we can standardize zones using the shape
    zones = None

    if zone_file:
        zones_raw = np.loadtxt(zone_file)
        zones, _, original_zones_shape = standardize_input_arrays(zones_raw, grid_coords, shape)
        print(
            f"Loaded zones: original shape {original_zones_shape}, standardized to 2D {zones.shape if zones is not None else None}")

    for fn in field_name:
        cols = ['name', 'x', 'y', 'major', 'anisotropy', 'bearing']
        cols = cols + [_ for _ in conceptual_points.columns if f'_{fn}' in _]
        if layer_mode and 'layer' in conceptual_points.columns:
            # Process by layer
            cols = cols + ['layer']
            field = generate_field_by_layer(conceptual_points[cols], grid_coords, iids,
                                            shape, zones=zones, save_path=save_path)
        else:
            # Process full 3D
            cols = cols + ['z']
            field = generate_field_3d(conceptual_points[cols], grid_coords, shape)

        # Save results - field is now array or original shape
        for i in conceptual_points.layer.unique():
            save_layer(field[i-1], layer=i, field_name=fn, save_path=save_path)
        # Plot based on layer in conceptual points
        for i in conceptual_points.layer.unique():
            plot_layer(np.where(zones==0,np.nan, field[i-1]), layer=i, field_name=fn, save_path=save_path)
        print("\n=== Complete ===")
        print(f"Field shape: {field.shape}")  # Now shows (nz, ny, nx)
        print(f"Field statistics: mean={np.mean(field):.3f}, std={np.std(field):.3f}")


def generate_field_by_layer(conceptual_points, grid_coords, iids,
                            shape, zones=None, save_path='.'):
    """
    Generate field processing each layer independently.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with tensor and statistical parameters
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    shape : tuple
        Grid shape as (nz, ny, nx)

    Returns
    -------
    np.ndarray
        Flattened field values
    """
    nz, ny, nx = shape
    field_3d = np.zeros(shape)
    grid_3d = grid_coords.reshape(shape + (3,))

    # Get unique layers
    if 'layer' in conceptual_points.columns:
        unique_layers = sorted([int(x-1) for x in conceptual_points['layer'].unique()])
        print(f"Processing layers: {unique_layers}")

    for layer_idx in unique_layers:
        print(f"\nProcessing layer {layer_idx}/{len(unique_layers)}")

        # Get grid points and iids for this layer
        layer_grid = grid_3d[layer_idx].reshape(-1, 3)
        layer_coords = layer_grid[:, :2]  # Just x, y
        layer_shape = (ny, nx)  # Shape for this layer
        layer_iids = iids[layer_idx]

        if 'layer' in conceptual_points.columns:
            layer_cp = conceptual_points[conceptual_points['layer']-1 == layer_idx]
            if len(layer_cp) == 0:
                print(f"  No conceptual points for layer {layer_idx}, using all")
                layer_cp = conceptual_points
        else:
            layer_cp = conceptual_points

        # Prepare zones for this layer (all zones now consistently 2D)
        layer_zones = None
        if zones is not None:
            if len(zones.shape) == 3:  # 3D zones - not supported yet, but keeping logic
                layer_zones = zones[layer_idx]
            else:  # 2D zones (standard case), use for all layers
                layer_zones = zones
            # zones are already guaranteed to be 2D from standardize_input_arrays()

        # Generate field for this layer - now returns 2D
        layer_field_2d, geological_tensors = generate_single_layer_zone_based(layer_cp, layer_coords,
                                                          layer_iids, zones=layer_zones, shape=layer_shape,
                                                          save_path=save_path)
        visualize_geological_tensors(geological_tensors, layer_coords, layer_shape, zones=layer_zones,
                                     conceptual_points=layer_cp, subsample=4,
                                     save_path=save_path, title_suf=f'layer {layer_idx+1}')

        # layer_field_2d is already 2D
        field_3d[layer_idx] = layer_field_2d

    return field_3d  # Return as 3D array, NOT flattened


def generate_single_layer_zone_based(conceptual_points, grid_coords_2d, iids,
                                     zones=None, shape=None, save_path='.'):
    """
    Modified generate_single_layer with zone-based tensor interpolation.
    """
    cp_coords = conceptual_points[['x', 'y']].values
    cp_means = conceptual_points.filter(like='mean').values.flatten()

    sd_cols = [col for col in conceptual_points.columns if 'sd_' in col.lower()]
    if sd_cols:
        cp_sd = conceptual_points[sd_cols[0]].values
    else:
        raise ValueError("No sd columns found (expected format: 'sd_fieldname')")

    if not shape:
        shape = infer_grid_shape(grid_coords_2d)

    # Step 1: Create geological tensors from conceptual points
    print("  Creating geological correlation tensors...")
    major = conceptual_points['major'].values
    minor = major / conceptual_points['anisotropy'].values

    # Better normalization that preserves direction
    def normalize_bearing(bearing):
        # Convert to 0-360 first
        bearing = bearing % 360
        # Don't automatically fold to 0-180, preserve the original direction choice
        return bearing

    bearings_normalized = [normalize_bearing(b) for b in conceptual_points['bearing'].values]
    bearing_rad = np.radians(bearings_normalized)
    cp_tensors = create_2d_tensors(bearing_rad, major, minor)

    # Debug: Test tensor creation with first few points
    print("  Tensor creation verification:")
    for i in range(min(3, len(cp_tensors))):
        original_bearing = conceptual_points['bearing'].values[i]
        extracted_bearing = get_tensor_bearing(cp_tensors[i], original_bearing)
        print(f"    Point {i}: Input bearing={original_bearing:.1f}°, Extracted={extracted_bearing:.1f}°")

    # Step 2: Zone-based tensor interpolation
    if zones is not None:
        print("  Interpolating geological tensors by zone...")
        geological_tensors = interpolate_tensors_by_zone_nearest(
            cp_coords, cp_tensors, grid_coords_2d, zones
        )
    else:
        print("  Interpolating geological tensors globally...")
        geological_tensors = interpolate_tensors_2d(cp_coords, cp_tensors, grid_coords_2d)

    # Step 3: Interpolate means using zone-aware kriging - returns 2D
    print("  Interpolating means with geological structure...")
    interp_means_2d = tensor_aware_kriging(
        cp_coords, cp_means, grid_coords_2d, geological_tensors,
        variogram_model='exponential', sill=1.0, nugget=0.1,
        background_value=np.mean(cp_means), max_search_radius=1e20,
        min_points=3, transform='log', min_value=1e-8, max_neighbors=8,
        zones=zones
    )
    if zones is not None:
        print("  Smoothing values at geological boundaries...")
        interp_means_2d = create_boundary_modified_scalar(
            interp_means_2d, zones,
            transition_cells=3, mode='smooth'
        )

    # Step 5: Interpolate sd using GEOLOGICAL tensors - returns 2D
    print("  Interpolating standard deviation...")
    interp_sd_2d = tensor_aware_kriging(
        cp_coords, cp_sd, grid_coords_2d, geological_tensors,  # Use geological tensors
        variogram_model='exponential', sill=1.0, nugget=0.1,
        background_value=np.mean(cp_sd), max_search_radius=1e20,
        min_points=3, transform=None, min_value=1e-8, max_neighbors=8,
        zones=zones
    )
    if zones is not None:
        print("  Enhancing variance at geological boundaries...")
        interp_sd_2d = create_boundary_modified_scalar(
            interp_sd_2d, zones,
            peak_increase=1, transition_cells=3, mode='enhance'
        )

    # Step 6: Generate correlated noise - work with 1D for noise generation, then reshape
    print("  Generating correlated noise...")
    correlated_noise_1d = generate_correlated_noise_2d(grid_coords_2d, geological_tensors, iids)
    correlated_noise_2d = correlated_noise_1d.reshape(shape)

    # Step 7: Combine into lognormal field - all operations in 2D
    field_2d = 10 ** (np.log10(interp_means_2d) + correlated_noise_2d * interp_sd_2d)

    return field_2d, geological_tensors


def interpolate_tensors_by_zone_nearest(cp_coords, cp_tensors, grid_coords, zones):
    """Zone-based nearest neighbor tensor assignment - no interpolation.

    Parameters
    ----------
    zones : np.ndarray
        Zone IDs, shape (ny, nx) - GUARANTEED to be 2D from entry point
    """
    shape = zones.shape
    n_grid = len(grid_coords)
    ny, nx = shape

    # Get zone for each conceptual point
    cp_zones = assign_conceptual_points_to_zones(cp_coords, grid_coords, zones)

    # Initialize output tensors
    geological_tensors = np.zeros((n_grid, 2, 2))

    # Get unique zones
    unique_zones = np.unique(zones)

    for zone_id in unique_zones:
        print(f"    Assigning nearest neighbor tensors for zone {zone_id}")

        # Find conceptual points in this zone
        zone_cp_mask = cp_zones == zone_id
        zone_cp_coords = cp_coords[zone_cp_mask]
        zone_cp_tensors = cp_tensors[zone_cp_mask]

        # Find grid points in this zone
        zone_grid_indices = []
        for i in range(n_grid):
            row, col = divmod(i, nx)
            if zones[row, col] == zone_id:
                zone_grid_indices.append(i)

        zone_grid_indices = np.array(zone_grid_indices)

        if len(zone_cp_coords) > 0:
            # Assign nearest neighbor tensor (no interpolation)
            for grid_idx in zone_grid_indices:
                distances = np.linalg.norm(zone_cp_coords - grid_coords[grid_idx], axis=1)
                nearest_cp = np.argmin(distances)
                geological_tensors[grid_idx] = zone_cp_tensors[nearest_cp]
        else:
            # No conceptual points in zone - use default tensor
            print(f"      Warning: No conceptual points in zone {zone_id}, using default tensor")
            default_tensor = np.eye(2) * 1000 ** 2
            geological_tensors[zone_grid_indices] = default_tensor

    return geological_tensors


def create_boundary_modified_scalar(base_field_2d, zones,
                                    peak_increase=0.3, transition_cells=5, mode="enhance"):
    """
    Modify scalar field values near geological zone boundaries with either enhancement or smoothing.

    Parameters
    ----------
    base_field_2d : np.ndarray
        Base scalar values
    zones : np.ndarray
        Zone IDs
    peak_increase : float
        Max enhancement or smoothing strength
    transition_cells : int
        Width of transition region
    mode : str
        'enhance' or 'smooth'

    Returns
    -------
    np.ndarray
        Modified scalar field
    """
    from scipy.ndimage import distance_transform_edt

    if mode not in ("enhance", "smooth"):
        raise ValueError("mode must be 'enhance' or 'smooth'")

    boundary_mask, _ = detect_zone_boundaries(zones)
    distance = distance_transform_edt(~boundary_mask)
    transition_mask = distance <= transition_cells

    modified = base_field_2d.copy()

    if mode == "enhance":
        # Linear enhancement near boundaries
        factor = 1 - distance[transition_mask] / transition_cells
        enhancement = peak_increase * factor
        modified[transition_mask] += enhancement

    elif mode == "smooth":
        # Smooth field with Gaussian filter as the target blending value
        smoothed_field = gaussian_filter(base_field_2d, sigma=transition_cells / 2)

        # Blend with original based on distance to boundary
        weight = 1 - distance[transition_mask] / transition_cells
        modified[transition_mask] = (
            weight * smoothed_field[transition_mask] +
            (1 - weight) * base_field_2d[transition_mask]
        )

    print(f"    {'Enhanced' if mode == 'enhance' else 'Smoothed'} {np.count_nonzero(transition_mask)} points near boundaries")
    return modified



def create_boundary_enhanced_tensors(geological_tensors, zones,
                                     boundary_anisotropy=10.0, boundary_major_scale=2.0,
                                     transition_cells=3):
    """
    Enhance geological tensors at zone boundaries with increased anisotropy.
    Uses distance-based transition from geological to boundary behavior.

    Parameters
    ----------
    geological_tensors : np.ndarray
        Base geological tensors, shape (n_points, 2, 2)
    zones : np.ndarray
        Zone IDs, shape (ny, nx) - GUARANTEED to be 2D from entry point
    boundary_anisotropy : float
        Anisotropy ratio at boundaries (major/minor)
    boundary_major_scale : float
        Scale factor for major axis at boundaries
    transition_cells : int
        Number of cells over which to transition from geological to boundary

    Returns
    -------
    np.ndarray
        Enhanced tensors, shape (n_points, 2, 2)
    """
    shape = zones.shape
    ny, nx = shape

    # Detect boundary points and orientations
    boundary_mask, boundary_directions = detect_zone_boundaries(zones)

    enhanced_tensors = geological_tensors.copy()
    boundary_count = 0

    for i in range(len(geological_tensors)):
        row, col = divmod(i, nx)

        # Calculate minimum distance to any boundary within transition zone
        min_dist_to_boundary = transition_cells + 1  # Start beyond transition

        for dr in range(-transition_cells, transition_cells + 1):
            for dc in range(-transition_cells, transition_cells + 1):
                check_row, check_col = row + dr, col + dc
                if (0 <= check_row < ny and 0 <= check_col < nx and
                        boundary_mask[check_row, check_col]):
                    dist = np.sqrt(dr ** 2 + dc ** 2)
                    min_dist_to_boundary = min(min_dist_to_boundary, dist)

        # Skip if far from boundaries
        if min_dist_to_boundary > transition_cells:
            continue

        boundary_count += 1

        # Get boundary direction (use closest boundary point's direction)
        boundary_dir = boundary_directions[row, col]
        if np.linalg.norm(boundary_dir) < 1e-10:
            # If current cell isn't boundary, find closest boundary direction
            for dr in range(-transition_cells, transition_cells + 1):
                for dc in range(-transition_cells, transition_cells + 1):
                    check_row, check_col = row + dr, col + dc
                    if (0 <= check_row < ny and 0 <= check_col < nx and
                            boundary_mask[check_row, check_col]):
                        boundary_dir = boundary_directions[check_row, check_col]
                        if np.linalg.norm(boundary_dir) > 1e-10:
                            break
                if np.linalg.norm(boundary_dir) > 1e-10:
                    break

        if np.linalg.norm(boundary_dir) < 1e-10:
            continue

        # Normalize direction
        boundary_dir = boundary_dir / np.linalg.norm(boundary_dir)

        # Create boundary tensor
        cos_theta = boundary_dir[0]
        sin_theta = boundary_dir[1]

        # rotate clockwise
        R = np.array([[cos_theta, sin_theta],
                      [-sin_theta, cos_theta]])

        eigenvals = np.linalg.eigvals(geological_tensors[i])
        dominant_length = np.sqrt(np.max(eigenvals))

        major_length = dominant_length * boundary_major_scale
        minor_length = major_length / boundary_anisotropy

        S_boundary = np.diag([minor_length ** 2, major_length ** 2])
        boundary_tensor = R @ S_boundary @ R.T

        # Distance-based blending with eigenvalue interpolation
        if min_dist_to_boundary == 0:
            # At boundary - pure boundary tensor
            enhanced_tensors[i] = boundary_tensor
        else:
            # Smooth transition - blend eigenvalues to preserve positive definiteness
            geo_weight = min_dist_to_boundary / transition_cells

            geo_vals, geo_vecs = np.linalg.eigh(geological_tensors[i])
            bound_vals, bound_vecs = np.linalg.eigh(boundary_tensor)

            # Interpolate eigenvalues, use boundary orientation
            mixed_vals = geo_weight * geo_vals + (1 - geo_weight) * bound_vals
            enhanced_tensors[i] = bound_vecs @ np.diag(mixed_vals) @ bound_vecs.T

    print(f"    Enhanced {boundary_count} tensors at geological boundaries")
    return enhanced_tensors


def detect_zone_boundaries(zones):
    """
    Detect boundaries between geological zones and estimate local orientations.

    Parameters
    ----------
    zones : np.ndarray
        Zone IDs, shape (ny, nx) - GUARANTEED to be 2D

    Returns
    -------
    tuple
        (boundary_mask, boundary_directions) where:
        - boundary_mask: bool array, shape (ny, nx)
        - boundary_directions: direction vectors, shape (ny, nx, 2)
    """
    ny, nx = zones.shape
    boundary_mask = np.zeros((ny, nx), dtype=bool)
    boundary_directions = np.zeros((ny, nx, 2))

    # Find boundary points (any point with different-zone neighbors)
    for i in range(ny):
        for j in range(nx):
            current_zone = zones[i, j]

            # Check 8-connected neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < ny and 0 <= nj < nx:
                        if zones[ni, nj] != current_zone:
                            boundary_mask[i, j] = True
                            break
                if boundary_mask[i, j]:
                    break

    # Estimate boundary orientations
    for i in range(ny):
        for j in range(nx):
            if not boundary_mask[i, j]:
                continue

            boundary_directions[i, j] = estimate_boundary_direction(zones, i, j)

    return boundary_mask, boundary_directions


def estimate_boundary_direction(zones, center_i, center_j, radius=2):
    """
    Estimate local boundary direction at a point.
    Returns direction vector in geological coordinates (North = 0°).

    Parameters
    ----------
    zones : np.ndarray
        Zone IDs - GUARANTEED to be 2D, shape (ny, nx)
    center_i, center_j : int
        Center point coordinates
    radius : int
        Search radius

    Returns
    -------
    np.ndarray
        Boundary direction vector [east_component, north_component]
        where North = [0, 1], East = [1, 0] (geological convention)
    """
    ny, nx = zones.shape
    center_zone = zones[center_i, center_j]

    same_zone_points = []
    diff_zone_points = []

    # Collect points within radius
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue

            ni, nj = center_i + di, center_j + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                if zones[ni, nj] == center_zone:
                    same_zone_points.append([dj, di])  # x=dj (east), y=di (south in array coords)
                else:
                    diff_zone_points.append([dj, di])

    if len(same_zone_points) < 1 or len(diff_zone_points) < 1:
        return np.array([0.0, 1.0])  # Default north (geological N=0°)

    same_zone_points = np.array(same_zone_points)
    diff_zone_points = np.array(diff_zone_points)

    # Find direction separating the groups
    same_centroid = np.mean(same_zone_points, axis=0)
    diff_centroid = np.mean(diff_zone_points, axis=0)

    separation_vector = diff_centroid - same_centroid
    if np.linalg.norm(separation_vector) < 1e-10:
        return np.array([0.0, 1.0])  # Default north

    # Convert to geological coordinates and get boundary direction
    # Array coordinates: x=east, y=south; Geological: x=east, y=north
    sep_east = separation_vector[0]
    sep_south = separation_vector[1]
    sep_north = -sep_south  # Flip y-axis: array south becomes geological north

    # Normalize separation vector in geological coordinates
    sep_geo = np.array([sep_east, sep_north])
    sep_geo_normalized = sep_geo / np.linalg.norm(sep_geo)

    # Boundary runs perpendicular to separation (rotate 90° counterclockwise)
    # For vector [east, north], perpendicular is [-north, east]
    boundary_direction = np.array([-sep_geo_normalized[1], sep_geo_normalized[0]])

    return boundary_direction


def tensor_aware_kriging(cp_coords, cp_values, grid_coords, interp_tensors,
                         variogram_model='exponential', sill=1.0, nugget=0.01,
                         background_value=0.0, max_search_radius=1e20, min_points=1,
                         max_neighbors=4, transform=None, min_value=1e-8,
                         zones=None):
    """
    Ordinary kriging using tensor-aware anisotropic distances.

    Parameters
    ----------
    zones : np.ndarray, optional
        Zone IDs, shape (ny, nx) - 2D array

    Returns
    -------
    np.ndarray
        Interpolated values, shape (ny, nx) - ALWAYS 2D
    """
    n_grid = len(grid_coords)

    # Get shape from zones if provided, or infer from grid
    if zones is not None:
        shape = zones.shape
        ny, nx = shape
    else:
        # Infer shape from grid coordinates
        x_unique = np.unique(grid_coords[:, 0])
        y_unique = np.unique(grid_coords[:, 1])
        ny, nx = len(y_unique), len(x_unique)
        shape = (ny, nx)

    interp_values_1d = np.full(n_grid, background_value)

    # Calculate zone for each conceptual point (once, outside the main loop)
    if zones is not None:
        cp_zones = []
        for coord in cp_coords:
            # Find closest grid point to get zone
            distances = np.sum((grid_coords - coord) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            row, col = divmod(closest_idx, nx)
            cp_zones.append(zones[row, col])
        cp_zones = np.array(cp_zones)

    for i in range(n_grid):
        local_tensor = interp_tensors[i]
        if zones is not None:
            # Get current zone and filter conceptual points
            row, col = divmod(i, nx)
            current_zone = zones[row, col]

            # Filter to same zone, fallback to all if insufficient points found
            zone_mask = cp_zones == current_zone
            if np.sum(zone_mask) >= min_points:
                cp_coords_filtered = cp_coords[zone_mask]
                cp_values_filtered = cp_values[zone_mask]
            else:
                cp_coords_filtered = cp_coords
                cp_values_filtered = cp_values
        else:
            cp_coords_filtered = cp_coords
            cp_values_filtered = cp_values

        try:
            tensor_inv = np.linalg.inv(local_tensor)
            aniso_distances = []

            for j in range(len(cp_coords_filtered)):
                dx = cp_coords_filtered[j] - grid_coords[i]
                aniso_dist = np.sqrt(dx.T @ tensor_inv @ dx)
                aniso_distances.append(aniso_dist)

            aniso_distances = np.array(aniso_distances)

        except np.linalg.LinAlgError:
            aniso_distances = np.linalg.norm(cp_coords_filtered - grid_coords[i], axis=1)

        sorted_indices = np.argsort(aniso_distances)
        n_candidates = min(max_neighbors or len(aniso_distances),
                           np.sum(aniso_distances <= max_search_radius))

        if n_candidates < min_points:
            continue

        closest_indices = sorted_indices[:n_candidates]
        nearby_values = cp_values_filtered[closest_indices]
        nearby_coords = cp_coords_filtered[closest_indices]
        nearby_distances = aniso_distances[closest_indices]

        if transform == 'log':
            if min_value is None:
                positive_values = nearby_values[nearby_values > 0]
                min_value = np.min(positive_values) * 0.01
            nearby_values_transformed = np.log10(np.maximum(nearby_values, min_value))
        else:
            nearby_values_transformed = nearby_values.copy()

        def variogram(h):
            if variogram_model == 'exponential':
                return nugget + (sill - nugget) * (1 - np.exp(-h))
            elif variogram_model == 'gaussian':
                return nugget + (sill - nugget) * (1 - np.exp(-(h ** 2)))
            else:  # spherical
                return np.where(h <= 1.0,
                                nugget + (sill - nugget) * (1.5 * h - 0.5 * h ** 3),
                                sill)

        C = np.zeros((n_candidates + 1, n_candidates + 1))
        c = np.zeros(n_candidates + 1)

        for j in range(n_candidates):
            for k in range(n_candidates):
                dx = nearby_coords[j] - nearby_coords[k]
                try:
                    h = np.sqrt(dx.T @ tensor_inv @ dx)
                except:
                    h = np.linalg.norm(dx)
                C[j, k] = variogram(h)

            C[j, -1] = 1
            C[-1, j] = 1
            c[j] = variogram(nearby_distances[j])

        c[-1] = 1

        try:
            cond_num = np.linalg.cond(C)
            if cond_num > 1e12:  # Matrix is poorly conditioned
                # Fall back to inverse distance weighting
                weights = np.exp(-nearby_distances)
                weights = weights / np.sum(weights)
                interp_values_1d[i] = np.sum(weights * nearby_values_transformed)
            else:
                weights = np.linalg.solve(C, c)[:-1]
                interp_values_1d[i] = np.sum(weights * nearby_values_transformed)
        except:
            # Existing fallback for any other errors
            weights = np.exp(-nearby_distances)
            interp_values_1d[i] = np.sum(weights * nearby_values_transformed) / np.sum(weights)

    if transform == 'log':
        interp_values_1d = 10 ** interp_values_1d

    # ALWAYS convert to 2D - no exceptions
    return interp_values_1d.reshape(shape)


def create_2d_tensors(theta, major, minor):
    """Create 2x2 anisotropy tensors from geostatistical parameters."""
    theta = np.atleast_1d(theta)
    major = np.atleast_1d(major)
    minor = np.atleast_1d(minor)

    n = len(theta)
    tensors = np.zeros((n, 2, 2))

    for i in range(n):
        # For geological bearing (clockwise from north):
        # North = [0, 1], East = [1, 0]
        # Bearing 5° should give direction [x, y] = [sin(5°), cos(5°)] = [0.087, 0.996]

        bearing_rad = theta[i]

        # Direction vector for this bearing
        east_comp = np.sin(bearing_rad)  # X component
        north_comp = np.cos(bearing_rad)  # Y component

        # Create rotation matrix to align [1,0] with bearing direction
        cos_t = north_comp
        sin_t = east_comp

        # clockwise rotation by angle theta
        R = np.array([[cos_t, sin_t],
                      [-sin_t, cos_t]])
        S = np.diag([minor[i] ** 2, major[i] ** 2])
        tensors[i] = R @ S @ R.T

    return tensors.squeeze() if n == 1 else tensors


def get_tensor_bearing(tensor, expected_bearing=None):
    """Extract geological bearing from tensor."""
    eigenvals, eigenvecs = np.linalg.eigh(tensor)
    max_idx = np.argmax(eigenvals)
    major_eigenvec = eigenvecs[:, max_idx]

    # Swap components to match geological convention: [north, east] instead of [east, north]
    north_component = major_eigenvec[1]  # y-component is north
    east_component = major_eigenvec[0]  # x-component is east

    # Calculate bearing from north (geological convention)
    angle1 = np.degrees(np.arctan2(east_component, north_component))
    if angle1 < 0:
        angle1 += 360
    angle2 = (angle1 + 180) % 360

    # Choose correct direction if expected bearing provided
    if expected_bearing is not None:
        diff1 = min(abs(angle1 - expected_bearing), abs(angle1 - expected_bearing + 360),
                    abs(angle1 - expected_bearing - 360))
        diff2 = min(abs(angle2 - expected_bearing), abs(angle2 - expected_bearing + 360),
                    abs(angle2 - expected_bearing - 360))
        return angle1 if diff1 < diff2 else angle2

    return angle1


def assign_conceptual_points_to_zones(cp_coords, grid_coords, zones):
    """
    Extract zone assignment logic from tensor_aware_kriging into reusable function.

    Parameters
    ----------
    zones : np.ndarray
        Zone IDs, shape (ny, nx) - GUARANTEED to be 2D from entry point
    """
    ny, nx = zones.shape

    cp_zones = []
    for coord in cp_coords:
        # Find closest grid point to get zone
        distances = np.sum((grid_coords - coord) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        row, col = divmod(closest_idx, nx)
        cp_zones.append(zones[row, col])

    return np.array(cp_zones)


def interpolate_tensors_by_zone(cp_coords, cp_tensors, grid_coords, zones):
    """
    Zone-aware tensor interpolation - interpolate tensors within each zone separately.

    Parameters
    ----------
    cp_coords : np.ndarray
        Conceptual point coordinates
    cp_tensors : np.ndarray
        Conceptual point tensors
    grid_coords : np.ndarray
        Grid coordinates
    zones : np.ndarray
        Zone array, shape (ny, nx) - GUARANTEED to be 2D from entry point

    Returns
    -------
    np.ndarray
        Interpolated tensors, shape (n_grid, 2, 2)
    """
    n_grid = len(grid_coords)
    ny, nx = zones.shape

    # Get zone for each conceptual point
    cp_zones = assign_conceptual_points_to_zones(cp_coords, grid_coords, zones)

    # Initialize output tensors
    geological_tensors = np.zeros((n_grid, 2, 2))

    # Get unique zones
    unique_zones = np.unique(zones)

    for zone_id in unique_zones:
        print(f"    Interpolating tensors for zone {zone_id}")

        # Find conceptual points in this zone
        zone_cp_mask = cp_zones == zone_id
        zone_cp_coords = cp_coords[zone_cp_mask]
        zone_cp_tensors = cp_tensors[zone_cp_mask]

        # Find grid points in this zone
        zone_grid_indices = []
        for i in range(n_grid):
            row, col = divmod(i, nx)
            if zones[row, col] == zone_id:
                zone_grid_indices.append(i)

        zone_grid_indices = np.array(zone_grid_indices)
        zone_grid_coords = grid_coords[zone_grid_indices]

        if len(zone_cp_coords) > 0:
            # Interpolate tensors within zone using only zone's conceptual points
            zone_tensors = interpolate_tensors_2d(zone_cp_coords, zone_cp_tensors, zone_grid_coords,
                                                  (len(zone_grid_coords), 2))
            geological_tensors[zone_grid_indices] = zone_tensors
        else:
            # No conceptual points in zone - use default tensor or nearest zone
            print(f"      Warning: No conceptual points in zone {zone_id}, using default tensor")
            default_tensor = np.eye(2) * 1000 ** 2  # Default 1000m correlation length
            geological_tensors[zone_grid_indices] = default_tensor

    return geological_tensors


def interpolate_tensors_2d(cp_coords, cp_tensors, grid_coords):
    """
    Interpolate 2x2 tensors using log-Euclidean approach.
    """
    shape = infer_grid_shape(grid_coords)
    n_grid = len(grid_coords)
    interp_tensors = np.zeros((n_grid, 2, 2))

    log_tensors = np.array([logm(t) for t in cp_tensors])

    for i in range(2):
        for j in range(i, 2):
            values = log_tensors[:, i, j].real
            # Use tensor_aware_kriging with shape for 2D output, but we need 1D for tensor interpolation
            # This is a special case - convert back to 1D for tensor component interpolation
            interp_values_2d = tensor_aware_kriging_1d_output(cp_coords, values, grid_coords, shape)
            interp_tensors[:, i, j] = interp_values_2d.flatten()
            if i != j:
                interp_tensors[:, j, i] = interp_values_2d.flatten()

    for i in range(n_grid):
        interp_tensors[i] = expm(interp_tensors[i])

    return interp_tensors


def tensor_aware_kriging_1d_output(cp_coords, cp_values, grid_coords):
    """
    Special version of kriging for tensor interpolation that returns 1D output.
    This is only used internally for tensor component interpolation.
    """
    shape = infer_grid_shape(grid_coords)
    ny, nx = shape
    n_grid = len(grid_coords)
    interp_values = np.full(n_grid, 0.0)

    # Simple ordinary kriging for tensor components
    for i in range(n_grid):
        distances = np.linalg.norm(cp_coords - grid_coords[i], axis=1)
        weights = np.exp(-distances / 1000)  # Simple exponential weights
        if np.sum(weights) > 1e-10:
            interp_values[i] = np.sum(weights * cp_values) / np.sum(weights)

    return interp_values


def generate_correlated_noise_2d(grid_coords, tensors, iids,
                                 n_neighbors=50, anisotropy_strength=1.0):
    """Generate spatially correlated noise respecting tensor anisotropy."""
    n_points = len(grid_coords)
    correlated_noise = np.zeros(n_points)

    for i in range(n_points):
        evals, evecs = np.linalg.eigh(tensors[i])
        if evals[1] < evals[0]:
            evals, evecs = evals[::-1], evecs[:, ::-1]

        major_corr_length = np.sqrt(evals[1])
        minor_corr_length = np.sqrt(evals[0])

        scale_for_minor_axis = minor_corr_length / anisotropy_strength
        scale_for_major_axis = major_corr_length * anisotropy_strength

        transform = evecs @ np.diag([scale_for_minor_axis, scale_for_major_axis])

        all_dx = grid_coords - grid_coords[i]
        inv_transform = np.linalg.inv(transform)
        dx_transformed = all_dx @ inv_transform.T
        aniso_distances = np.linalg.norm(dx_transformed, axis=1)

        neighbor_indices = np.argsort(aniso_distances)[:min(n_neighbors, n_points)]

        weighted_sum = 0.0
        sum_weights = 0.0

        for j in neighbor_indices:
            # Use anisotropic distance instead of actual distance for weight calculation
            # This respects the tensor-defined correlation structure
            weight = np.exp(-(aniso_distances[j]) ** 2)
            weighted_sum += weight * iids.flatten()[j]
            sum_weights += weight

        if sum_weights > 1e-10:
            correlated_noise[i] = weighted_sum / sum_weights
        else:
            correlated_noise[i] = iids[i]

    return correlated_noise


def ordinary_kriging(cp_coords, cp_values, grid_coords,
                     variogram_model='exponential', range_param=10000,
                     sill=1.0, nugget=0.1, background_value=0.0,
                     max_search_radius=1e20, min_points=1,
                     transform=None, min_value=1e-8, max_neighbors=5):
    """
    Standard ordinary kriging with variogram models and search controls.

    Returns
    -------
    np.ndarray
        Interpolated values (1D array) - kept as 1D since this is a utility function
    """
    n_grid = len(grid_coords)
    interp_values = np.full(n_grid, background_value)

    def variogram(h):
        if variogram_model == 'exponential':
            return nugget + (sill - nugget) * (1 - np.exp(-h / range_param))
        elif variogram_model == 'gaussian':
            return nugget + (sill - nugget) * (1 - np.exp(-(h ** 2) / (range_param ** 2)))
        else:  # spherical
            return np.where(h <= range_param,
                            nugget + (sill - nugget) * (1.5 * h / range_param - 0.5 * (h / range_param) ** 3),
                            sill)

    for i in range(n_grid):
        distances = np.linalg.norm(cp_coords - grid_coords[i], axis=1)
        sorted_indices = np.argsort(distances)
        n_candidates = min(max_neighbors or len(distances),
                           np.sum(distances <= max_search_radius))

        if n_candidates < min_points:
            continue

        closest_indices = sorted_indices[:n_candidates]
        nearby_values = cp_values[closest_indices]
        nearby_coords = cp_coords[closest_indices]
        nearby_distances = distances[closest_indices]

        if transform == 'log':
            if min_value is None:
                positive_values = nearby_values[nearby_values > 0]
                min_value = np.min(positive_values) * 0.01
            nearby_values_transformed = np.log10(np.maximum(nearby_values, min_value))
        else:
            nearby_values_transformed = nearby_values.copy()

        C = np.zeros((n_candidates + 1, n_candidates + 1))
        c = np.zeros(n_candidates + 1)

        for j in range(n_candidates):
            for k in range(n_candidates):
                h = np.linalg.norm(nearby_coords[j] - nearby_coords[k])
                C[j, k] = variogram(h)
            C[j, -1] = 1
            C[-1, j] = 1
            c[j] = variogram(nearby_distances[j])
        c[-1] = 1

        try:
            weights = solve(C, c)[:-1]
            interp_values[i] = np.sum(weights * nearby_values_transformed)
        except:
            weights = np.exp(-nearby_distances / range_param)
            interp_values[i] = np.sum(weights * nearby_values_transformed) / np.sum(weights)

    if transform == 'log':
        interp_values = 10 ** interp_values

    return interp_values


def generate_field_3d(conceptual_points, grid_coords, shape):
    """Placeholder for full 3D field generation (not implemented)."""
    raise NotImplementedError("Use layer_mode=True for now")


def infer_grid_shape(grid_coords):
    """Infer grid dimensions from coordinate array."""
    x_unique = np.unique(grid_coords[:, 0])
    y_unique = np.unique(grid_coords[:, 1])
    z_unique = np.unique(grid_coords[:, 2])

    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(grid_coords) // (nx * ny)

    return (nz, ny, nx)


def save_layer(field, layer, field_name, save_path='.'):
    """
    Save field as individual layer files.

    Parameters
    ----------
    field : np.ndarray
        Field array, shape (nz, ny, nx) for 3D or (ny, nx) for 2D
    shape : tuple
        Grid shape (nz, ny, nx) or (ny, nx)
    field_name : str or list
        Field name(s) used for filename
    save_path : str
        Directory to save output files
    """
    if isinstance(field_name, list):
        field_name = field_name[0]

    os.makedirs(save_path, exist_ok=True)

    filename = os.path.join(save_path, f'{field_name}_layer_{layer + 1:02d}.arr')
    np.savetxt(filename, field, fmt='%.6f')
    print(f"Saved layer {layer} to '{filename}'")

def plot_layer(field, layer=0, field_name='field', save_path='.', transform='log'):
    import matplotlib.colors as mcolors
    if transform=='log':
        field = np.log10(field)

    colors = [
        (0.00, 'ghostwhite'),
        (0.05, 'lightblue'),
        (0.15, 'skyblue'),
        (0.25, 'blue'),
        (0.35, 'teal'),
        (0.45, 'green'),
        (0.50, 'limegreen'),
        (0.55, 'yellowgreen'),
        (0.65, 'yellow'),
        (0.75, 'gold'),
        (0.85, 'orange'),
        (0.90, 'darkorange'),
        (0.95, 'peru'),
        (1.00, 'sienna')
    ]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=256)

    plt.figure(figsize=(10, 8))
    plt.imshow(field, cmap=custom_cmap)
    plt.colorbar(label='Parameter Value')
    plt.title(f'{field_name} for layer {layer}')
    plt.xlabel('X')
    plt.ylabel('Y')

    os.makedirs('output', exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{field_name}_layer_{layer:02d}.png'), dpi=150)
    plt.close()

def add_grid_indices_to_conceptual_points(conceptual_points, grid_coords, shape):
    """
    Add i,j grid indices to conceptual points dataframe.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with x,y coordinates
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    shape : tuple
        Grid shape (ny, nx)

    Returns
    -------
    pd.DataFrame
        Conceptual points with added 'grid_i' and 'grid_j' columns
    """
    ny, nx = shape
    cp_coords = conceptual_points[['x', 'y']].values

    grid_i_list = []
    grid_j_list = []

    for coord in cp_coords:
        # Find closest grid point
        distances = np.sum((grid_coords[:, :2] - coord) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        row, col = divmod(closest_idx, nx)

        grid_i_list.append(row)
        grid_j_list.append(col)

    # Add to dataframe
    conceptual_points_copy = conceptual_points.copy()
    conceptual_points_copy['grid_i'] = grid_i_list
    conceptual_points_copy['grid_j'] = grid_j_list

    return conceptual_points_copy


def plot_zones_with_conceptual_points(zones, conceptual_points, grid_coords,
                                      figsize=(12, 10), save_path='.'):
    """
    Plot geology zones with conceptual points overlaid and labeled.

    Parameters
    ----------
    zones : np.ndarray
        Zone array, shape (n_points,) or (ny, nx)
    conceptual_points : pd.DataFrame
        Must have columns: x, y, grid_i, grid_j
    grid_coords : np.ndarray
        Grid coordinates for coordinate transformation
    shape : tuple
        Grid shape (ny, nx)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    shape = zones.shape
    ny, nx = shape

    # Get grid bounds for coordinate transformation
    x_coords = grid_coords[:, 0].reshape(shape[0], shape[1])
    y_coords = grid_coords[:, 1].reshape(shape[0], shape[1])

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot zones with better color handling
    unique_zones = np.unique(zones)
    n_zones = len(unique_zones)

    # Create a mapping from zone values to sequential colors
    zone_to_color = {zone: i for i, zone in enumerate(unique_zones)}
    zones_remapped = np.zeros_like(zones)
    for zone_val, color_idx in zone_to_color.items():
        zones_remapped[zones == zone_val] = color_idx

    # Plot zones - flip the array itself to match real-world orientation
    zones_flipped = np.flipud(zones_remapped)

    im = ax.imshow(zones_flipped, extent=[x_min, x_max, y_min, y_max],
                   origin='lower', cmap='tab10', alpha=0.7,
                   vmin=0, vmax=n_zones - 1)

    # Add colorbar with actual zone values
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Zone ID')
    cbar.set_ticks(range(n_zones))
    cbar.set_ticklabels([f'{int(z)}' for z in unique_zones])

    # Plot conceptual points
    for idx, row in conceptual_points.iterrows():
        x, y = row['x'], row['y']
        grid_i, grid_j = row['grid_i'], row['grid_j']
        name = conceptual_points.loc[idx, 'name']

        # Plot point
        ax.plot(x, y, 'ro', markersize=8, markeredgecolor='black', markeredgewidth=1)

        # Add label with index and grid position
        ax.annotate(f'{name}\n({grid_i},{grid_j})',
                    (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Draw line to corresponding grid cell center
        grid_i_int, grid_j_int = int(grid_i), int(grid_j)
        if 0 <= grid_i_int < ny and 0 <= grid_j_int < nx:
            grid_x = x_coords[grid_i_int, grid_j_int]
            grid_y = y_coords[grid_i_int, grid_j_int]
            ax.plot([x, grid_x], [y, grid_y], 'r--', alpha=0.5, linewidth=1)

            # Mark grid cell center
            ax.plot(grid_x, grid_y, 'r+', markersize=10, markeredgewidth=2)

    ax.set_xlabel('X (NZTM)')
    ax.set_ylabel('Y (NZTM)')
    ax.set_title('Geology Zones with Conceptual Points\n(Red dots = CP locations, + = assigned grid cells)')
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r',
                   markersize=8, label='Conceptual Points'),
        plt.Line2D([0], [0], marker='+', color='r', markersize=10,
                   label='Assigned Grid Cells'),
        plt.Line2D([0], [0], color='r', linestyle='--',
                   label='CP to Grid Assignment')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f'conceptual_points.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {os.path.join(save_path, f'{save_path}/conceptual_points.png')}")

    plt.close()


def debug_zone_assignments(conceptual_points, grid_coords, zones, shape, save_path='.'):
    """
    Complete workflow to debug zone assignments.
    """
    # Add grid indices
    cp_with_indices = add_grid_indices_to_conceptual_points(
        conceptual_points, grid_coords, shape
    )

    # Print summary
    print("Conceptual Point Zone Assignments:")
    print("=" * 50)
    for idx, row in cp_with_indices.iterrows():
        ny, nx = shape
        grid_i, grid_j = int(row['grid_i']), int(row['grid_j'])

        # zones are guaranteed to be 2D from entry point
        zone_id = zones[grid_i, grid_j]

        print(f"CP {idx}: ({row['x']:.1f}, {row['y']:.1f}) -> "
              f"grid[{grid_i},{grid_j}] -> zone {zone_id}")

    # Create plot
    plot_zones_with_conceptual_points(
        zones, cp_with_indices, grid_coords, shape,
        save_path=os.path.join(save_path, f'zone_debug.png'))

    return cp_with_indices


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_geological_tensors(geological_tensors, grid_coords, shape, zones=None,
                                 conceptual_points=None, subsample=4, scale_factor=20,
                                 figsize=(12, 10), title_suf=None, save_path='.'):
    """
    Visualize geological tensors as ellipses overlaid on zones, with conceptual points as oriented lines.

    Parameters
    ----------
    geological_tensors : np.ndarray
        Tensor field, shape (n_points, 2, 2)
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 2) or (n_points, 3)
    shape : tuple
        Grid shape (ny, nx)
    zones : np.ndarray, optional
        Zone array for background
    conceptual_points : pd.DataFrame, optional
        Conceptual points with x, y, bearing, major columns
    subsample : int
        Show every Nth tensor (for readability)
    scale_factor : float
        Scale ellipses for visibility
    """
    ny, nx = shape

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare zone data but don't plot yet
    if zones is not None:
        # zones are guaranteed to be 2D from entry point

        # Create zone color mapping
        unique_zones = np.unique(zones)
        zone_to_color = {zone: i for i, zone in enumerate(unique_zones)}
        zones_remapped = np.zeros_like(zones)
        for zone_val, color_idx in zone_to_color.items():
            zones_remapped[zones == zone_val] = color_idx

        # Get coordinate bounds
        x_coords = grid_coords[:, 0].reshape(shape)
        y_coords = grid_coords[:, 1].reshape(shape)
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        zones_flipped = np.flipud(zones_remapped)
        # Plot zones as background with low alpha
        im = ax.imshow(zones_flipped, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='tab10', alpha=0.2, zorder=0,
                       vmin=0, vmax=len(unique_zones) - 1)

    # Plot conceptual points as oriented lines
    if conceptual_points is not None:
        for idx, row in conceptual_points.iterrows():
            x, y = row['x'], row['y']
            bearing = row['bearing']
            major = row['major']

            # Scale line length for visibility (similar to ellipse scaling)
            line_length = major / (scale_factor * 2)  # Adjust scaling as needed

            # Convert geological bearing (CW from N) to math angle for plotting
            # Geological: 0°=N, 90°=E; Math: 0°=E, 90°=N
            math_angle_rad = np.radians(90 - bearing)

            # Calculate line endpoints
            dx = line_length * np.cos(math_angle_rad) / 2
            dy = line_length * np.sin(math_angle_rad) / 2

            # Plot line centered on point
            ax.plot([x - dx, x + dx], [y - dy, y + dy],
                    'r-', linewidth=3, alpha=0.9, zorder=4)

            # Plot center point
            ax.plot(x, y, 'ro', markersize=4, markeredgecolor='black',
                    markeredgewidth=1, alpha=0.9, zorder=5)

            # Add label with bearing
            ax.annotate(f'{bearing:.0f}°', (x, y), xytext=(8, 8),
                        textcoords='offset points', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                        zorder=6)

    # Subsample points for visualization
    indices = np.arange(0, len(geological_tensors), subsample)

    print(f"Plotting {len(indices)} tensors out of {len(geological_tensors)}")

    default_count = 0
    ellipse_count = 0

    for i in indices:
        tensor = geological_tensors[i]
        x, y = grid_coords[i, 0], grid_coords[i, 1]

        # Check for default tensor (indicates no interpolation happened)
        if np.allclose(tensor, np.eye(2) * 1000000):
            # Plot as red circle for default tensors with high zorder
            ax.plot(x, y, 'ko', markersize=2, alpha=0.3, zorder=1)  # Make less prominent
            default_count += 1
            continue

        try:
            # Eigendecomposition to get ellipse parameters
            eigenvals, eigenvecs = np.linalg.eigh(tensor)

            # Sort by eigenvalue magnitude
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Ellipse dimensions (correlation lengths) - make them more visible
            major_axis = 2 * np.sqrt(eigenvals[0]) / scale_factor
            minor_axis = 2 * np.sqrt(eigenvals[1]) / scale_factor

            # Make sure ellipses are visible
            if major_axis < 30:  # Smaller minimum to avoid overwhelming CP lines
                major_axis = 30
            if minor_axis < 15:
                minor_axis = 15

            # Ellipse orientation (angle of major axis)
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

            # Create ellipse with lower prominence than CP lines
            ellipse = patches.Ellipse((x, y), major_axis, minor_axis,
                                      angle=angle, linewidth=1,
                                      edgecolor='blue', facecolor='none', alpha=0.6, zorder=2)
            ax.add_patch(ellipse)
            ellipse_count += 1

        except:
            # If tensor is problematic, plot as yellow point with high zorder
            ax.plot(x, y, 'yo', markersize=2, zorder=3)

    print(f"Plotted {default_count} default tensors (black circles) and {ellipse_count} ellipses")

    ax.set_xlabel('X (NZTM)')
    ax.set_ylabel('Y (NZTM)')

    # Update title and legend
    title_parts = [f'Geological Tensors (subsampled 1:{subsample})\n']
    if title_suf is not None:
        title_parts += [f'{title_suf}']
    # more like legend?
    if conceptual_points is not None:
        title_parts.append('Red lines = conceptual points (bearing)')
    title_parts.append('Blue ellipses = interpolated tensors')
    ax.set_title('\n'.join(title_parts))

    ax.set_aspect('equal', adjustable='box')

    # Add colorbar for zones if present
    if zones is not None:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Zone ID')
        cbar.set_ticks(range(len(unique_zones)))
        cbar.set_ticklabels([f'{int(z)}' for z in unique_zones])

    # Add legend
    legend_elements = []
    if conceptual_points is not None:
        legend_elements.extend([
            plt.Line2D([0], [0], color='red', linewidth=3, label='Conceptual Point Bearings'),
            plt.Line2D([0], [0], marker='o', color='red', markersize=6,
                       markeredgecolor='black', label='Conceptual Points', linestyle='None')
        ])
    legend_elements.extend([
        patches.Patch(facecolor='none', edgecolor='blue', label='Interpolated Tensors'),
        plt.Line2D([0], [0], marker='o', color='black', markersize=4,
                   label='Default Tensors', linestyle='None', alpha=0.3)
    ])

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'intrpolated_tensors_{title_suf}.png'), dpi=150, bbox_inches='tight')
    print(f"Saved tensor visualization to {os.path.join(save_path, f'intrpolated_tensors_{title_suf}.png')}")
    plt.close()


def analyze_cp_tensor_alignment(conceptual_points, geological_tensors, grid_coords, zones):
    """
    Compare conceptual point bearings with actual tensor orientations at those locations.

    Parameters
    ----------
    zones : np.ndarray
        Zone IDs, shape (ny, nx) - GUARANTEED to be 2D from entry point
    """
    shape = zones.shape
    ny, nx = shape

    print("Conceptual Point vs Tensor Orientation Analysis:")
    print("=" * 70)
    print(f"{'CP#':<4} {'Name':<12} {'Zone':<6} {'Input°':<8} {'Tensor°':<8} {'Diff°':<8} {'Status'}")
    print("-" * 70)

    for idx, row in conceptual_points.iterrows():
        cp_x, cp_y = row['x'], row['y']
        input_bearing = row['bearing']
        cp_name = row.get('name', f'CP_{idx}')

        # Find closest grid point
        distances = np.sum((grid_coords[:, :2] - np.array([cp_x, cp_y])) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        grid_row, grid_col = divmod(closest_idx, nx)

        # Get zone
        zone_id = zones[grid_row, grid_col]

        # Get tensor at that location
        tensor = geological_tensors[closest_idx]

        # Extract orientation
        if np.allclose(tensor, np.eye(2) * 1000000):
            tensor_bearing = "DEFAULT"
            diff = "N/A"
            status = "No interp"
        else:
            tensor_bearing = get_tensor_bearing(tensor, input_bearing)

            # Calculate difference (handling wraparound)
            diff = abs(input_bearing - tensor_bearing)
            if diff > 180:
                diff = 360 - diff

            # Status based on difference
            if diff < 10:
                status = "✓ Good"
            elif diff < 30:
                status = "~ Close"
            else:
                status = "✗ Bad"

        print(f"{idx:<4} {cp_name:<12} {zone_id:<6.0f} {input_bearing:<8.0f} {tensor_bearing:<8} {diff:<8} {status}")

    print("-" * 70)


def analyze_tensor_statistics(geological_tensors, grid_coords, zones):
    """
    Print statistics about the tensor field by zone.

    Parameters
    ----------
    zones : np.ndarray
        Zone IDs, shape (ny, nx) - GUARANTEED to be 2D from entry point
    """
    shape = zones.shape
    ny, nx = shape

    print("Tensor Analysis by Zone:")
    print("=" * 50)

    for zone_id in np.unique(zones):
        # Find grid points in this zone
        zone_indices = []
        for i in range(len(grid_coords)):
            row, col = divmod(i, nx)
            if zones[row, col] == zone_id:
                zone_indices.append(i)

        if len(zone_indices) == 0:
            continue

        zone_tensors = geological_tensors[zone_indices]

        # Count default tensors
        default_count = 0
        anisotropies = []
        orientations = []

        for tensor in zone_tensors:
            if np.allclose(tensor, np.eye(2) * 1000000):
                default_count += 1
            else:
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(tensor)
                    idx = np.argsort(eigenvals)[::-1]
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]

                    aniso_ratio = np.sqrt(eigenvals[0] / eigenvals[1])
                    orientation = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

                    anisotropies.append(aniso_ratio)
                    orientations.append(orientation)
                except:
                    pass

        print(f"Zone {zone_id}:")
        print(f"  Total points: {len(zone_indices)}")
        print(f"  Default tensors: {default_count} ({100 * default_count / len(zone_indices):.1f}%)")
        if anisotropies:
            print(f"  Anisotropy ratio: {np.mean(anisotropies):.2f} ± {np.std(anisotropies):.2f}")
            print(f"  Orientation: {np.mean(orientations):.1f}° ± {np.std(orientations):.1f}°")
        print()


# ============================================================================
# EXAMPLE USAGE - UNCHANGED FUNCTIONALITY
# ============================================================================

# Example usage
if __name__ == "__main__":
    """
    Example usage of the tensor-based field generator.

    Required file formats:
    - conceptual_points.csv: CSV with columns x,y,major,anisotropy,bearing,mean_kh,sd_kh
    - grid.dat: Whitespace-delimited with columns x,y,z

    Optional:
    - zones: Can be passed to generate_single_layer() as numpy array
      with integer zone IDs, shape (n_points,) or (ny, nx)
    """
    main(
        conceptual_points_file=r'C:\modelling\sva_ds\data\conceptual_points.csv',
        grid_file=r'C:\modelling\sva_ds\data\grid.dat',
        zone_file=r'C:\modelling\sva_ds\data\waq_arr.geoclass_simple.arr',
        field_name=['kh'],
        layer_mode=True,
        save_path=r'C:\modelling\sva_ds\data\output'
    )