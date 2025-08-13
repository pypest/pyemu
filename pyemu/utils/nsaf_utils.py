import numpy as np
import pandas as pd
from scipy.linalg import logm, expm
import os
import matplotlib as mpl

mpl.use('Tkagg')
import matplotlib.pyplot as plt


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


def _ensure_positive_definite(tensor, min_eigenval=1e-8):
    """Ensure tensor is positive definite."""
    eigenvals, eigenvecs = np.linalg.eigh(tensor)
    eigenvals_reg = np.maximum(eigenvals, min_eigenval)
    return eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T


def create_2d_tensors(theta, major, minor):
    """Create 2x2 anisotropy tensors from geostatistical parameters.

    Parameters
    ----------
    theta : array-like
        Bearing angles in radians (0 = North, clockwise positive)
    major : array-like
        Major axis correlation lengths
    minor : array-like
        Minor axis correlation lengths

    Returns
    -------
    np.ndarray
        Tensors with shape (n, 2, 2)
    """
    theta = np.atleast_1d(theta)
    major = np.atleast_1d(major)
    minor = np.atleast_1d(minor)

    n = len(theta)
    tensors = np.zeros((n, 2, 2))

    for i in range(n):
        bearing_rad = theta[i]

        # Direction vector for this bearing (geological convention)
        east_comp = np.sin(bearing_rad)  # X component
        north_comp = np.cos(bearing_rad)  # Y component

        # Create rotation matrix to align [1,0] with bearing direction
        cos_t = north_comp
        sin_t = east_comp

        # Clockwise rotation by angle theta
        R = np.array([[cos_t, sin_t],
                      [-sin_t, cos_t]])
        S = np.diag([minor[i] ** 2, major[i] ** 2])
        tensors[i] = R @ S @ R.T

    return tensors.squeeze() if n == 1 else tensors


def load_conceptual_points(cp_file):
    """Load conceptual points from CSV file or DataFrame.

    Parameters
    ----------
    cp_file : str or pd.DataFrame
        Path to CSV file or DataFrame with columns: name,x,y,z,mean_kh,sd_kh,major,anisotropy,bearing,layer

    Returns
    -------
    pd.DataFrame
        Conceptual points with added 'transverse' column if missing
    """
    if isinstance(cp_file, str):
        cp_df = pd.read_csv(cp_file)
    elif isinstance(cp_file, pd.DataFrame):
        cp_df = cp_file.copy()
    else:
        raise TypeError(f"cp_file must be string path or pandas DataFrame, got {type(cp_file)}")

    #  Validate required columns (excluding anisotropy/transverse for now)
    required_cols = ['x', 'y', 'major', 'bearing']
    missing_cols = [col for col in required_cols if col not in cp_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for anisotropy OR transverse (at least one must exist)
    has_anisotropy = 'anisotropy' in cp_df.columns
    has_transverse = 'transverse' in cp_df.columns

    if not has_anisotropy and not has_transverse:
        raise ValueError("Must have either 'anisotropy' OR 'transverse' column")

    # Add transverse column if missing (derive from anisotropy)
    if not has_transverse and has_anisotropy:
        cp_df['transverse'] = cp_df['major'] / cp_df['anisotropy']
        print(f"  Added 'transverse' column derived from anisotropy")

    # Add anisotropy column if missing (derive from transverse)
    elif not has_anisotropy and has_transverse:
        cp_df['anisotropy'] = cp_df['major'] / cp_df['transverse']
        print(f"  Added 'anisotropy' column derived from transverse")

    # If both exist, validate they're consistent
    elif has_anisotropy and has_transverse:
        expected_transverse = cp_df['major'] / cp_df['anisotropy']
        if not np.allclose(cp_df['transverse'], expected_transverse, rtol=1e-3):
            print("  Warning: 'transverse' and 'anisotropy' columns are inconsistent")
            print("  Using 'transverse' values and recalculating 'anisotropy'")
            cp_df['anisotropy'] = cp_df['major'] / cp_df['transverse']

    return cp_df


def interpolate_tensors(cp_file, xcentergrid, ycentergrid, zones=None,
                        method='krig', layer=0, config_dict=None):
    """
    Tensor field interpolation using NSAF log-Euclidean mathematics
    with pypestutils kriging engine.

    Parameters
    ----------
    cp_file : str or pd.DataFrame
        Conceptual points file path or DataFrame
    xcentergrid : np.ndarray
        2D array of x-coordinates from pyemu SpatialReference (ny, nx)
    ycentergrid : np.ndarray
        2D array of y-coordinates from pyemu SpatialReference (ny, nx)
    zones : np.ndarray, optional
        2D zone array (ny, nx)
    method : str, default 'krig'
        Interpolation method: 'krig', 'idw', or 'nearest'
    layer : int, default 0
        Layer number to process
    config_dict : dict, optional
        Configuration parameters for pypestutils

    Returns
    -------
    np.ndarray
        Interpolated tensors with shape (ny*nx, 2, 2)
    """
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    # Load and filter conceptual points
    cp_df = load_conceptual_points(cp_file)

    # Filter to current layer
    if 'layer' in cp_df.columns:
        layer_cp = cp_df[cp_df['layer'] == layer].copy()
    else:
        layer_cp = cp_df.copy()

    if len(layer_cp) == 0:
        raise ValueError(f"No conceptual points found for layer {layer}")

    # Set default config
    if config_dict is None:
        config_dict = {
            'vartype': 2,  # Exponential variogram
            'krigtype': 1,  # Ordinary kriging
            'search_radius': 1e10,
            'maxpts_interp': 20,
            'minpts_interp': 1
        }

    # Prepare grid coordinates
    ny, nx = xcentergrid.shape
    grid_x = xcentergrid.flatten()
    grid_y = ycentergrid.flatten()

    # Prepare conceptual point data
    cp_coords = np.column_stack([layer_cp['x'].values, layer_cp['y'].values])

    # Convert bearing from degrees to radians (maintaining geological convention)
    bearing_rad = np.radians(layer_cp['bearing'].values)

    # Create tensors at conceptual points
    cp_tensors = create_2d_tensors(
        bearing_rad,
        layer_cp['major'].values,
        layer_cp['transverse'].values  # This is minor axis
    )

    # Handle zones
    if zones is not None:
        zones_flat = zones.flatten().astype(int)
        cp_zones = _assign_cp_to_zones(cp_coords, grid_x, grid_y, zones_flat, nx)
    else:
        zones_flat = np.ones(len(grid_x), dtype=int)
        cp_zones = np.ones(len(layer_cp), dtype=int)

    # Interpolate tensors using log-Euclidean approach
    if method == 'krig':
        interp_tensors = _interpolate_tensors_kriging(
            cp_coords, cp_tensors, cp_zones,
            grid_x, grid_y, zones_flat,
            config_dict, nx, ny
        )
    elif method == 'idw':
        interp_tensors = _interpolate_tensors_idw(
            cp_coords, cp_tensors, cp_zones,
            grid_x, grid_y, zones_flat,
            nx, ny
        )
    elif method == 'nearest':
        interp_tensors = _interpolate_tensors_nearest(
            cp_coords, cp_tensors, cp_zones,
            grid_x, grid_y, zones_flat,
            nx, ny
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return interp_tensors


def _assign_cp_to_zones(cp_coords, grid_x, grid_y, zones_flat, nx):
    """Assign conceptual points to zones based on nearest grid point."""
    grid_coords = np.column_stack([grid_x, grid_y])
    cp_zones = np.zeros(len(cp_coords), dtype=int)

    for i, cp_coord in enumerate(cp_coords):
        distances = np.sum((grid_coords - cp_coord) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        cp_zones[i] = zones_flat[closest_idx]

    return cp_zones


def _interpolate_tensors_kriging(cp_coords, cp_tensors, cp_zones,
                                 grid_x, grid_y, zones_flat, config_dict, nx, ny):
    """Interpolate tensors using pypestutils kriging with log-Euclidean approach."""
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    n_grid = len(grid_x)
    result_tensors = np.zeros((n_grid, 2, 2))

    # Convert tensors to log space
    log_tensors = np.array([logm(_ensure_positive_definite(tensor))
                            for tensor in cp_tensors])

    # Process each zone separately
    unique_zones = np.unique(zones_flat)

    for zone_id in unique_zones:
        zone_mask = zones_flat == zone_id
        zone_indices = np.where(zone_mask)[0]

        if len(zone_indices) == 0:
            continue

        # Filter conceptual points for this zone
        cp_mask = cp_zones == zone_id
        if np.sum(cp_mask) == 0:
            # No conceptual points in this zone - use default tensor
            default_tensor = np.eye(2) * 1000 ** 2
            result_tensors[zone_indices] = default_tensor
            continue

        zone_cp_coords = cp_coords[cp_mask]
        zone_log_tensors = log_tensors[cp_mask]
        zone_grid_x = grid_x[zone_indices]
        zone_grid_y = grid_y[zone_indices]

        # Interpolate each tensor component separately
        zone_tensors = np.zeros((len(zone_indices), 2, 2))

        for i in range(2):
            for j in range(i, 2):  # Only upper triangle, then mirror
                component_values = zone_log_tensors[:, i, j].real

                # Use pypestutils for kriging
                interp_component = _krig_component_pypestutils(
                    zone_cp_coords[:, 0], zone_cp_coords[:, 1],
                    component_values, zone_grid_x, zone_grid_y,
                    config_dict
                )

                zone_tensors[:, i, j] = interp_component
                if i != j:
                    zone_tensors[:, j, i] = interp_component

        # Convert back from log space and ensure positive definiteness
        for k in range(len(zone_indices)):
            try:
                tensor_exp = expm(zone_tensors[k])
                result_tensors[zone_indices[k]] = _ensure_positive_definite(tensor_exp)
            except:
                # Fallback to default tensor
                result_tensors[zone_indices[k]] = np.eye(2) * 1000 ** 2

    return result_tensors


def _krig_component_pypestutils(cp_x, cp_y, cp_values, grid_x, grid_y, config_dict):
    """Use pypestutils to krig a single tensor component."""
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    lib = PestUtilsLib()

    # Create temporary factor file
    fac_fname = "temp_tensor_component.fac"
    if os.path.exists(fac_fname):
        os.remove(fac_fname)
    fac_ftype = "text"

    # All points in same zone for component interpolation
    cp_zones = np.ones(len(cp_x), dtype=int)
    grid_zones = np.ones(len(grid_x), dtype=int)

    # Use isotropic parameters for component interpolation
    corrlen = np.full(len(grid_x), 1000.0)  # Default correlation length
    aniso = np.ones(len(grid_x))  # Isotropic
    bearing = np.zeros(len(grid_x))  # No rotation

    try:
        npts = lib.calc_kriging_factors_2d(
            cp_x, cp_y, cp_zones,
            grid_x, grid_y, grid_zones,
            int(config_dict.get("vartype", 2)),  # Exponential
            int(config_dict.get("krigtype", 1)),  # Ordinary kriging
            corrlen, aniso, bearing,
            config_dict.get("search_radius", 1e10),
            config_dict.get("maxpts_interp", 20),
            config_dict.get("minpts_interp", 1),
            fac_fname, fac_ftype
        )

        # Apply kriging using the component values
        result = lib.krige_using_file(
            fac_fname, fac_ftype,
            len(grid_x),
            int(config_dict.get("krigtype", 1)),
            "none",  # No transformation for individual components
            cp_values,
            np.mean(cp_values),  # Fill value
            np.mean(cp_values)  # No-interpolation value
        )

        if os.path.exists(fac_fname):
            os.remove(fac_fname)

        lib.free_all_memory()

        return result["targval"]

    except Exception as e:
        # Cleanup and fallback
        if os.path.exists(fac_fname):
            os.remove(fac_fname)
        lib.free_all_memory()

        # Simple IDW fallback
        return _idw_component(cp_x, cp_y, cp_values, grid_x, grid_y)


def _idw_component(cp_x, cp_y, cp_values, grid_x, grid_y, power=2):
    """Inverse distance weighting fallback for component interpolation."""
    result = np.zeros(len(grid_x))

    for i in range(len(grid_x)):
        distances = np.sqrt((cp_x - grid_x[i]) ** 2 + (cp_y - grid_y[i]) ** 2)
        distances = np.maximum(distances, 1e-10)  # Avoid division by zero

        weights = 1.0 / (distances ** power)
        result[i] = np.sum(weights * cp_values) / np.sum(weights)

    return result


def _interpolate_tensors_idw(cp_coords, cp_tensors, cp_zones,
                             grid_x, grid_y, zones_flat, nx, ny):
    """IDW interpolation using log-Euclidean approach."""
    n_grid = len(grid_x)
    result_tensors = np.zeros((n_grid, 2, 2))

    # Convert to log space
    log_tensors = np.array([logm(_ensure_positive_definite(tensor))
                            for tensor in cp_tensors])

    # Process each zone
    unique_zones = np.unique(zones_flat)

    for zone_id in unique_zones:
        zone_mask = zones_flat == zone_id
        zone_indices = np.where(zone_mask)[0]

        if len(zone_indices) == 0:
            continue

        cp_mask = cp_zones == zone_id
        if np.sum(cp_mask) == 0:
            default_tensor = np.eye(2) * 1000 ** 2
            result_tensors[zone_indices] = default_tensor
            continue

        zone_cp_coords = cp_coords[cp_mask]
        zone_log_tensors = log_tensors[cp_mask]

        for idx in zone_indices:
            gx, gy = grid_x[idx], grid_y[idx]
            distances = np.sqrt((zone_cp_coords[:, 0] - gx) ** 2 +
                                (zone_cp_coords[:, 1] - gy) ** 2)
            distances = np.maximum(distances, 1e-10)

            weights = 1.0 / (distances ** 2)
            weights /= np.sum(weights)

            # Weighted average in log space
            log_result = np.sum(weights[:, np.newaxis, np.newaxis] * zone_log_tensors, axis=0)

            try:
                tensor_exp = expm(log_result)
                result_tensors[idx] = _ensure_positive_definite(tensor_exp)
            except:
                result_tensors[idx] = np.eye(2) * 1000 ** 2

    return result_tensors


def _interpolate_tensors_nearest(cp_coords, cp_tensors, cp_zones,
                                 grid_x, grid_y, zones_flat, nx, ny):
    """Nearest neighbor interpolation."""
    n_grid = len(grid_x)
    result_tensors = np.zeros((n_grid, 2, 2))

    # Process each zone
    unique_zones = np.unique(zones_flat)

    for zone_id in unique_zones:
        zone_mask = zones_flat == zone_id
        zone_indices = np.where(zone_mask)[0]

        if len(zone_indices) == 0:
            continue

        cp_mask = cp_zones == zone_id
        if np.sum(cp_mask) == 0:
            default_tensor = np.eye(2) * 1000 ** 2
            result_tensors[zone_indices] = default_tensor
            continue

        zone_cp_coords = cp_coords[cp_mask]
        zone_cp_tensors = cp_tensors[cp_mask]

        for idx in zone_indices:
            gx, gy = grid_x[idx], grid_y[idx]
            distances = np.sqrt((zone_cp_coords[:, 0] - gx) ** 2 +
                                (zone_cp_coords[:, 1] - gy) ** 2)
            nearest_idx = np.argmin(distances)
            result_tensors[idx] = zone_cp_tensors[nearest_idx]

    return result_tensors


def apply_ppu_hyperpars(cp_file, xcentergrid, ycentergrid, zones=None,
                        out_filename=None, n_realizations=1, layer=0,
                        vartransform="none", boundary_smooth=True,
                        boundary_enhance=True, tensor_method='krig',
                        config_dict=None, iids=None, area=None, **kwargs):
    """
    Parameter interpolation and field generation combining NSAF tensor mathematics
    with pypestutils FIELDGEN2D_SVA for spatially correlated noise.

    Parameters
    ----------
    cp_file : str or pd.DataFrame
        Conceptual points file or DataFrame
    xcentergrid : np.ndarray
        X-coordinates from pyemu SpatialReference, shape (ny, nx)
    ycentergrid : np.ndarray
        Y-coordinates from pyemu SpatialReference, shape (ny, nx)
    zones : np.ndarray, optional
        Zone array, shape (ny, nx)
    out_filename : str, optional
        Base filename for output (if None, returns arrays only)
    n_realizations : int, default 1
        Number of stochastic realizations to generate
    layer : int, default 0
        Layer number to process
    vartransform : str, default "none"
        Transformation: "none" or "log"
    boundary_smooth : bool, default True
        Apply smoothing to mean at zone boundaries
    boundary_enhance : bool, default True
        Apply variance enhancement at zone boundaries
    tensor_method : str, default 'krig'
        Tensor interpolation method: 'krig', 'idw', or 'nearest'
    config_dict : dict, optional
        Configuration parameters for pypestutils
    iids : np.ndarray, optional
        Pre-generated IIDs with shape (ny*nx, n_realizations)

    Returns
    -------
    dict
        Dictionary containing:
        - 'fields': Generated parameter fields, shape (n_realizations, ny, nx)
        - 'mean': Interpolated mean field, shape (ny, nx)
        - 'sd': Interpolated standard deviation field, shape (ny, nx)
        - 'tensors': Interpolated tensors, shape (ny*nx, 2, 2)
    """

    print("=== Parameter Interpolation and Field Generation ===")

    # Load conceptual points
    cp_df = load_conceptual_points(cp_file)
    if 'layer' in cp_df.columns:
        layer_cp = cp_df[cp_df['layer'] == layer].copy()
    else:
        layer_cp = cp_df.copy()

    if len(layer_cp) == 0:
        raise ValueError(f"No conceptual points found for layer {layer}")

    # Get grid dimensions and coordinates
    ny, nx = xcentergrid.shape
    grid_coords_2d = np.column_stack([xcentergrid.flatten(), ycentergrid.flatten()])

    print(f"Processing layer {layer} with {len(layer_cp)} conceptual points")
    print(f"Grid dimensions: {ny} x {nx} = {ny * nx} cells")

    # Step 1: Interpolate tensors
    print("\nStep 1: Interpolating tensor field...")
    tensors = interpolate_tensors(
        layer_cp, xcentergrid, ycentergrid,
        zones=zones, method=tensor_method,
        layer=layer, config_dict=config_dict
    )

    # Step 2: Convert tensors to geostatistical parameters
    print("\nStep 2: Converting tensors to geostatistical parameters...")
    bearing_deg, anisotropy, corrlen_major = _tensors_to_geostat_params(tensors)

    # Step 3: Prepare conceptual point data
    cp_coords = np.column_stack([layer_cp['x'].values, layer_cp['y'].values])
    cp_means = layer_cp['mean_kh'].values
    cp_sd = layer_cp['sd_kh'].values

    # Step 4: Interpolate mean and sd using tensor-aware kriging
    print("\nStep 4: Interpolating mean and standard deviation fields...")

    # Configure kriging parameters
    if config_dict is None:
        config_dict = {
            'vartype': 2,  # Exponential variogram
            'krigtype': 1,  # Ordinary kriging
            'search_radius': 1e10,
            'maxpts_interp': 20,
            'minpts_interp': 1
        }

    # Transform for log-domain if needed
    transform = 'log' if vartransform == 'log' else None
    min_value = 1e-8 if vartransform == 'log' else None

    # Interpolate mean
    print("  Interpolating mean...")
    interp_means_2d = tensor_aware_kriging(
        cp_coords, cp_means, grid_coords_2d, tensors,
        variogram_model='exponential', sill=1.0, nugget=0.1,
        background_value=np.mean(cp_means), max_search_radius=1e20,
        min_points=3, transform=transform, min_value=min_value,
        max_neighbors=4, zones=zones
    )

    # Apply boundary smoothing to mean
    if boundary_smooth and zones is not None:
        print("  Smoothing mean at geological boundaries...")
        interp_means_2d = create_boundary_modified_scalar(
            interp_means_2d, zones,
            transition_cells=3, mode='smooth'
        )

    # Interpolate standard deviation
    print("  Interpolating standard deviation...")
    interp_sd_2d = tensor_aware_kriging(
        cp_coords, cp_sd, grid_coords_2d, tensors,
        variogram_model='exponential', sill=1.0, nugget=0.1,
        background_value=np.mean(cp_sd), max_search_radius=1e20,
        min_points=3, transform=transform, min_value=min_value,
        max_neighbors=4, zones=zones
    )

    # Apply boundary enhancement to sd
    if boundary_enhance and zones is not None:
        print("  Enhancing variance at geological boundaries...")
        interp_sd_2d = create_boundary_modified_scalar(
            interp_sd_2d, zones,
            peak_increase=1.0, transition_cells=3, mode='enhance'
        )

    # Step 5: Generate stochastic fields using FIELDGEN2D_SVA approach
    print(f"\nStep 5: Generating {n_realizations} stochastic realizations...")

    if n_realizations > 0:
        fields = _generate_stochastic_fields(
            grid_coords=grid_coords_2d,
            mean_field=interp_means_2d,
            sd_field=interp_sd_2d,
            bearing=bearing_deg,
            anisotropy=anisotropy,
            corrlen=corrlen_major,
            zones=zones,
            n_realizations=n_realizations,
            vartransform=vartransform,
            config_dict=config_dict,
            ny=ny,
            nx=nx,
            area=area,
            iids=iids
        )
    else:
        fields = None

    # Prepare output
    results = {
        'fields': fields,
        'mean': interp_means_2d,
        'sd': interp_sd_2d,
        'tensors': tensors,
        'bearing': bearing_deg.reshape(ny, nx),
        'anisotropy': anisotropy.reshape(ny, nx),
        'corrlen': corrlen_major.reshape(ny, nx)
    }

    # Save to files if requested
    if out_filename is not None:
        _save_results(results, out_filename, vartransform)

    print("=== Interpolation complete ===")
    return results


def _tensors_to_geostat_params(tensors):
    """Convert tensor field to geostatistical parameters."""
    n_points = len(tensors)
    bearing_deg = np.zeros(n_points)
    anisotropy = np.zeros(n_points)
    corrlen_major = np.zeros(n_points)

    for i, tensor in enumerate(tensors):
        # Skip default tensors
        if np.allclose(tensor, np.eye(2) * 1000000):
            bearing_deg[i] = 0.0
            anisotropy[i] = 1.0
            corrlen_major[i] = 1000.0
            continue

        try:
            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(tensor)

            # Sort by eigenvalue magnitude
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]

            # Extract parameters
            major_length = np.sqrt(eigenvals[0])
            minor_length = np.sqrt(eigenvals[1])

            corrlen_major[i] = major_length
            anisotropy[i] = major_length / minor_length

            # Calculate bearing from major eigenvector
            major_vec = eigenvecs[:, 0]
            # Convert to geological bearing (0° = North, clockwise positive)
            bearing_rad = np.arctan2(major_vec[0], major_vec[1])
            bearing_deg[i] = np.degrees(bearing_rad) % 360

        except:
            # Fallback values
            bearing_deg[i] = 0.0
            anisotropy[i] = 1.0
            corrlen_major[i] = 1000.0

    return bearing_deg, anisotropy, corrlen_major


def tensor_aware_kriging(cp_coords, cp_values, grid_coords, interp_tensors,
                         variogram_model='exponential', sill=1.0, nugget=0.01,
                         background_value=0.0, max_search_radius=1e20, min_points=1,
                         max_neighbors=4, transform=None, min_value=1e-8,
                         zones=None):
    """
    Tensor-aware kriging using anisotropic distances.
    (Implementation from the NSAF code provided earlier)
    """
    # Get grid shape
    n_grid = len(grid_coords)
    n_dims = grid_coords.shape[1]

    # Handle zones shape inference
    if zones is not None:
        shape = zones.shape
        if n_dims == 2:
            ny, nx = shape
        else:
            nz, ny, nx = shape
    else:
        # Simple rectangular grid assumption
        grid_span_x = grid_coords[:, 0].max() - grid_coords[:, 0].min()
        grid_span_y = grid_coords[:, 1].max() - grid_coords[:, 1].min()

        # Estimate grid dimensions
        unique_x = len(np.unique(grid_coords[:, 0]))
        unique_y = len(np.unique(grid_coords[:, 1]))

        if unique_x * unique_y == n_grid:
            nx, ny = unique_x, unique_y
        else:
            # Fallback to square root
            nx = ny = int(np.sqrt(n_grid))

        shape = (ny, nx)

    # Define index conversion functions
    if n_dims == 3:
        def get_indices(idx):
            layer = idx // (ny * nx)
            remainder = idx % (ny * nx)
            row = remainder // nx
            col = remainder % nx
            return layer, row, col
    else:
        def get_indices(idx):
            row = idx // nx
            col = idx % nx
            return row, col

    # Variogram function
    def variogram(h):
        if variogram_model == 'exponential':
            return nugget + (sill - nugget) * (1 - np.exp(-h))
        elif variogram_model == 'gaussian':
            return nugget + (sill - nugget) * (1 - np.exp(-(h ** 2)))
        else:  # spherical
            return np.where(h <= 1.0,
                            nugget + (sill - nugget) * (1.5 * h - 0.5 * h ** 3),
                            sill)

    interp_values_1d = np.full(n_grid, background_value)

    # Calculate zone for each conceptual point if zones provided
    if zones is not None:
        cp_zones = []
        for coord in cp_coords:
            distances = np.sum((grid_coords - coord) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            if n_dims == 3:
                layer, row, col = get_indices(closest_idx)
                cp_zones.append(zones[layer, row, col])
            else:
                row, col = get_indices(closest_idx)
                cp_zones.append(zones[row, col])
        cp_zones = np.array(cp_zones)

    # Pre-compute log transformation if needed
    if transform == 'log':
        if min_value is None:
            positive_values = cp_values[cp_values > 0]
            min_value = np.min(positive_values) * 0.01 if len(positive_values) > 0 else 1e-8

    # Main kriging loop
    for i in range(n_grid):
        local_tensor = interp_tensors[i]

        if zones is not None:
            # Get current zone and filter conceptual points
            if n_dims == 3:
                layer, row, col = get_indices(i)
                current_zone = zones[layer, row, col]
            else:
                row, col = get_indices(i)
                current_zone = zones[row, col]

            # Filter to same zone
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

        # Compute anisotropic distances
        try:
            tensor_inv = np.linalg.inv(local_tensor)
            dx = cp_coords_filtered - grid_coords[i]
            aniso_distances = np.sqrt(np.sum(dx @ tensor_inv * dx, axis=1))
        except np.linalg.LinAlgError:
            aniso_distances = np.linalg.norm(cp_coords_filtered - grid_coords[i], axis=1)

        # Get neighbors within search radius
        valid_mask = aniso_distances <= max_search_radius
        if np.sum(valid_mask) < min_points:
            continue

        # Sort and limit to max_neighbors
        sorted_indices = np.argsort(aniso_distances[valid_mask])
        n_candidates = min(max_neighbors or len(sorted_indices), len(sorted_indices))

        if n_candidates < min_points:
            continue

        closest_indices = np.where(valid_mask)[0][sorted_indices[:n_candidates]]
        nearby_values = cp_values_filtered[closest_indices]
        nearby_coords = cp_coords_filtered[closest_indices]
        nearby_distances = aniso_distances[closest_indices]

        # Apply log transform if needed
        if transform == 'log':
            nearby_values_transformed = np.log10(np.maximum(nearby_values, min_value))
        else:
            nearby_values_transformed = nearby_values.copy()

        # Set up and solve kriging system
        n_pts = len(nearby_values_transformed)
        C = np.zeros((n_pts + 1, n_pts + 1))
        c = np.zeros(n_pts + 1)

        # Build covariance matrix
        for j in range(n_pts):
            for k in range(n_pts):
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

        # Solve kriging system
        try:
            cond_num = np.linalg.cond(C)
            if cond_num > 1e12:
                # Fall back to IDW
                weights = np.exp(-nearby_distances)
                weights = weights / np.sum(weights)
                interp_values_1d[i] = np.sum(weights * nearby_values_transformed)
            else:
                weights = np.linalg.solve(C, c)[:-1]
                interp_values_1d[i] = np.sum(weights * nearby_values_transformed)
        except:
            # IDW fallback
            weights = np.exp(-nearby_distances)
            interp_values_1d[i] = np.sum(weights * nearby_values_transformed) / np.sum(weights)

    # Back-transform if needed
    if transform == 'log':
        interp_values_1d = 10 ** interp_values_1d

    return interp_values_1d.reshape(shape)


def _generate_stochastic_fields(grid_coords, mean_field, sd_field,
                                bearing, anisotropy, corrlen,
                                zones, n_realizations, vartransform,
                                config_dict, ny, nx, iids=None, area=None):
    """
    Generate stochastic fields using pypestutils FIELDGEN2D_SVA.

    This creates temporary files as required by FIELDGEN2D_SVA and then calls
    the pypestutils wrapper.
    """
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    print("  Generating stochastic fields using FIELDGEN2D_SVA...")

    # Prepare grid data
    n_points = ny * nx
    x_coords = grid_coords[:, 0]
    y_coords = grid_coords[:, 1]

    # Flatten fields
    mean_flat = mean_field.flatten()
    variance_flat = (sd_field.flatten()) ** 2  # Convert SD to variance

    # Geostatistical parameters from tensors
    aa_values = corrlen.flatten()
    aniso_values = anisotropy.flatten()
    bearing_values = bearing.flatten()

    # domain based on zones
    if zones is None:
        domain = np.ones((ny,nx))
    else:
        domain = np.where(zones>1,1,0)
    if area is None:
        area = domain.copy()
    domain = domain.flatten().astype(int)
    area = area.flatten()

    # Generate or use provided IIDs
    if iids is None:
        print("  Generating new IIDs...")
        iids = np.random.normal(0, 1, size=(n_points, n_realizations))
    else:
        print("  Using provided IIDs...")
        if iids.shape != (n_points, n_realizations):
            raise ValueError(f"IIDs shape {iids.shape} doesn't match expected {(n_points, n_realizations)}")

    # Create temporary files for FIELDGEN2D_SVA
    node_file = "temp_nodes_2d.dat"
    avgfunc_file = "temp_avgfunc_2d.dat"
    output_file = "temp_output_2d.csv"

    try:
        # Create 2D node specifications file
        # _create_node_spec_file(node_file, x_coords, y_coords, zones_flat)
        #
        # # Create 2D averaging function specification file
        # _create_avgfunc_spec_file(avgfunc_file, mean_flat, variance_flat,
        #                           aa_values, aniso_values, bearing_values)

        # Set up pypestutils parameters
        lib = PestUtilsLib()
        lib.initialize_randgen(11235813)

        # Determine transformation type
        if vartransform == 'log':
            trans_flag = 'log'  # Log domain
        else:
            trans_flag = 'none'  # Natural domain

        # Averaging function type
        avg_func = config_dict.get('averaging_function', 'exponential')
        if isinstance(avg_func, dict):
            if avg_func == 'pow':
                avg_flag, power_value = next(iter(avg_func.items()))
        else:
            power_value = 0
        if avg_func == 'exponential':
            avg_flag = 'exp'
        elif avg_func == 'gaussian':
            avg_flag = 'gauss'
        elif avg_func == 'spherical':
            avg_flag = 'spher'
        else:
            avg_flag = 'exp'  # Default to exponential

        # Call FIELDGEN2D_SVA through pypestutils interface
        print(f"    Calling FIELDGEN2D_SVA for {n_realizations} realizations...")

        # This is the correct way to call FIELDGEN2D_SVA in pypestutils
        success = lib.fieldgen2d_sva(
                    x_coords, y_coords,
                    area=area,
                    active=domain,
                    mean=mean_flat,
                    var=variance_flat,
                    aa=corrlen,
                    anis=anisotropy,
                    bearing=bearing,
                    transtype=trans_flag,  # e.g., 'n' or 'l'
                    avetype=avg_flag,      # 's', 'x', 'g', or 'p'
                    power=power_value,     # only needed if avetype='p'
                    nreal=n_realizations,

                )

        if not success:
            raise Exception("FIELDGEN2D_SVA execution failed")

        # Read results from CSV file
        print("    Reading FIELDGEN2D_SVA results...")
        results_df = pd.read_csv(output_file)

        # Extract field values - assume columns are named 'real_001', 'real_002', etc.
        fields = np.zeros((n_realizations, ny, nx))

        for real in range(n_realizations):
            col_name = f"real_{real + 1:03d}"
            if col_name in results_df.columns:
                field_values = results_df[col_name].values
                fields[real] = field_values.reshape(ny, nx)
            else:
                # Fallback column naming
                field_values = results_df.iloc[:, real].values
                fields[real] = field_values.reshape(ny, nx)

        # Clean up
        lib.free_all_memory()

    except Exception as e:
        print(f"    Warning: FIELDGEN2D_SVA failed: {e}")
        print(f"    Using fallback method...")

        # Fallback to simple field generation
        fields = np.zeros((n_realizations, ny, nx))
        for real in range(n_realizations):
            current_iids = iids[:, real]
            field_values = _generate_simple_field(
                grid_coords, mean_flat, variance_flat,
                aa_values, aniso_values, bearing_values,
                vartransform, current_iids, ny, nx
            )
            fields[real] = field_values.reshape(ny, nx)

    finally:
        # Clean up temporary files
        for temp_file in [node_file, avgfunc_file, output_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return fields


# def _create_node_spec_file(filename, x_coords, y_coords, zones):
#     """Create 2D node specifications file for FIELDGEN2D_SVA."""
#     n_points = len(x_coords)
#
#     with open(filename, 'w') as f:
#         # Header
#         f.write(f"{n_points}\n")  # Number of nodes
#
#         # Node data: node_id, x, y, zone, active_flag
#         for i in range(n_points):
#             node_id = i + 1  # 1-based indexing
#             x = x_coords[i]
#             y = y_coords[i]
#             zone = zones[i]
#             active = 1  # All nodes active
#             f.write(f"{node_id:8d} {x:15.6f} {y:15.6f} {zone:6d} {active:2d}\n")
#
#
# def _create_avgfunc_spec_file(filename, mean_vals, variance_vals, aa_vals, aniso_vals, bearing_vals):
#     """Create 2D averaging function specification file for FIELDGEN2D_SVA."""
#     n_points = len(mean_vals)
#
#     with open(filename, 'w') as f:
#         # Header
#         f.write(f"{n_points}\n")  # Number of nodes
#
#         # Averaging function data: node_id, mean, variance, aa, anisotropy, bearing
#         for i in range(n_points):
#             node_id = i + 1  # 1-based indexing
#             mean = mean_vals[i]
#             variance = variance_vals[i]
#             aa = aa_vals[i]  # Correlation length (major axis)
#             aniso = aniso_vals[i]  # Anisotropy ratio
#             bearing = bearing_vals[i]  # Bearing in degrees
#
#             f.write(f"{node_id:8d} {mean:15.6f} {variance:15.6f} {aa:15.6f} {aniso:10.4f} {bearing:10.2f}\n")


def _generate_simple_field(grid_coords, mean_flat, variance_flat,
                           aa_values, aniso_values, bearing_values,
                           vartransform, iids, ny, nx):
    """
    Simple fallback field generation if FIELDGEN2D_SVA fails.
    """
    print("    Using simple correlated field generation fallback...")

    n_points = len(grid_coords)

    # Scale by local standard deviation
    stochastic_component = iids * np.sqrt(variance_flat)

    # Apply simple spatial correlation using distance-based averaging
    smoothed_component = np.zeros_like(stochastic_component)

    for i in range(n_points):
        local_aa = aa_values[i]

        # Simple correlation based on distance
        distances = np.linalg.norm(grid_coords - grid_coords[i], axis=1)
        weights = np.exp(-distances / (local_aa + 1e-10))  # Avoid division by zero
        weights = weights / np.sum(weights)

        # Apply weighted average
        smoothed_component[i] = np.sum(weights * stochastic_component)

    # Combine with mean
    if vartransform == 'log':
        # Log domain: mean * exp(stochastic_component)
        field_values = mean_flat * np.exp(smoothed_component)
    else:
        # Normal domain: mean + stochastic_component
        field_values = mean_flat + smoothed_component

    return field_values


def create_boundary_modified_scalar(base_field, zones,
                                    peak_increase=0.3, transition_cells=5, mode="enhance"):
    """
    Modify scalar field values near geological zone boundaries.
    (Implementation from the provided NSAF code)
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if mode not in ("enhance", "smooth"):
        raise ValueError("mode must be 'enhance' or 'smooth'")

    # 2D case only for now
    if zones.shape != base_field.shape:
        raise ValueError(f"Zones shape {zones.shape} must match field shape {base_field.shape}")

    boundary_mask, _ = detect_zone_boundaries(zones)
    distance = distance_transform_edt(~boundary_mask)
    transition_mask = distance <= transition_cells

    modified = base_field.copy()

    if mode == "enhance":
        # Linear enhancement near boundaries
        factor = 1 - distance[transition_mask] / transition_cells
        enhancement = peak_increase * factor
        modified[transition_mask] += enhancement

    elif mode == "smooth":
        # Smooth field with Gaussian filter
        smoothed_field = gaussian_filter(base_field, sigma=transition_cells / 2)

        # Blend with original based on distance to boundary
        weight = 1 - distance[transition_mask] / transition_cells
        modified[transition_mask] = (
                weight * smoothed_field[transition_mask] +
                (1 - weight) * base_field[transition_mask]
        )

    print(
        f"    {'Enhanced' if mode == 'enhance' else 'Smoothed'} {np.count_nonzero(transition_mask)} points near boundaries")

    return modified


def detect_zone_boundaries(zones):
    """Detect boundaries between geological zones."""
    ny, nx = zones.shape
    boundary_mask = np.zeros((ny, nx)).astype(bool)
    boundary_directions = np.zeros((ny, nx, 2))

    # Find boundary points
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

    return boundary_mask, boundary_directions


def _save_results(results, out_filename, vartransform):
    """Save results to files in PyEMU-compatible format."""

    # Save mean field
    mean_file = f"{out_filename}_mean.txt"
    np.savetxt(mean_file, results['mean'], fmt="%20.8E")
    print(f"  Saved mean field to {mean_file}")

    # Save standard deviation field
    sd_file = f"{out_filename}_sd.txt"
    np.savetxt(sd_file, results['sd'], fmt="%20.8E")
    print(f"  Saved SD field to {sd_file}")

    # Save geostatistical parameter fields
    for param in ['bearing', 'anisotropy', 'corrlen']:
        param_file = f"{out_filename}_{param}.txt"
        np.savetxt(param_file, results[param], fmt="%20.8E")
        print(f"  Saved {param} field to {param_file}")

    # Save stochastic realizations
    if results['fields'] is not None:
        for i, field in enumerate(results['fields']):
            field_file = f"{out_filename}_real_{i + 1:03d}.txt"
            np.savetxt(field_file, field, fmt="%20.8E")

        print(f"  Saved {len(results['fields'])} realizations")


# Example usage function
def example_usage():
    """Example of how to use the tensor interpolation."""

    # Create example conceptual points DataFrame
    cp_data = {
        'name': ['cp1', 'cp2', 'cp3', 'cp4'],
        'x': [1000, 3000, 1000, 3000],
        'y': [1000, 1000, 3000, 3000],
        'z': [0, 0, 0, 0],
        'mean_kh': [1.0, 1.0, 1.0, 1.0],
        'sd_kh': [0.1, 0.1, 0.1, 0.1],
        'major': [2000, 1500, 1800, 2200],
        'anisotropy': [3.0, 2.5, 4.0, 2.0],
        'bearing': [45, 90, 135, 0],  # Degrees
        'layer': [0, 0, 0, 0]
    }
    cp_df = pd.DataFrame(cp_data)

    # Create example grid (pyemu style)
    nx, ny = 50, 40
    x = np.linspace(0, 5000, nx)
    y = np.linspace(0, 4000, ny)
    xcentergrid, ycentergrid = np.meshgrid(x, y)

    # Example zones
    zones = np.ones((ny, nx), dtype=int)
    zones[:ny // 2, :] = 1
    zones[ny // 2:, :] = 2

    # Configuration
    config = {
        'vartype': 2,  # Exponential
        'krigtype': 1,  # Ordinary kriging
        'search_radius': 10000,
        'maxpts_interp': 10,
        'minpts_interp': 1
    }

    # Interpolate tensors
    tensors = interpolate_tensors(
        cp_df, xcentergrid, ycentergrid,
        zones=zones, method='krig',
        layer=0, config_dict=config
    )

    print(f"Interpolated tensor field shape: {tensors.shape}")
    print(f"Sample tensor at grid point 0:\n{tensors[0]}")

    return tensors


def example_workflow():
    """Example of complete workflow."""
    import matplotlib.pyplot as plt

    # Create example data
    nx, ny = 50, 40
    x = np.linspace(0, 5000, nx)
    y = np.linspace(0, 4000, ny)
    xcentergrid, ycentergrid = np.meshgrid(x, y, indexing='xy')
    xcentergrid = xcentergrid.T
    ycentergrid = ycentergrid.T

    # Example conceptual points
    cp_data = {
        'name': ['cp1', 'cp2', 'cp3', 'cp4'],
        'x': [1000, 3000, 1000, 3000],
        'y': [1000, 1000, 3000, 3000],
        'z': [0, 0, 0, 0],
        'mean_kh': [1.5, 2.0, 1.8, 1.2],
        'sd_kh': [0.3, 0.4, 0.2, 0.5],
        'major': [2000, 1500, 1800, 2200],
        'anisotropy': [3.0, 2.5, 4.0, 2.0],
        'bearing': [45, 90, 135, 0],
        'layer': [0, 0, 0, 0]
    }
    cp_df = pd.DataFrame(cp_data)

    # Example zones
    zones = np.ones((ny, nx), dtype=int)
    zones[:ny // 2, :] = 1
    zones[ny // 2:, :] = 2

    # Run workflow
    results = apply_ppu_hyperpars(
        cp_df, xcentergrid, ycentergrid,
        zones=zones, n_realizations=3,
        out_filename="example",
        vartransform="none"
    )

    # Quick visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Mean field
    im1 = axes[0, 0].imshow(results['mean'], origin='lower')
    axes[0, 0].set_title('Mean Field')
    plt.colorbar(im1, ax=axes[0, 0])

    # SD field
    im2 = axes[0, 1].imshow(results['sd'], origin='lower')
    axes[0, 1].set_title('Standard Deviation')
    plt.colorbar(im2, ax=axes[0, 1])

    # Anisotropy field
    im3 = axes[0, 2].imshow(results['anisotropy'], origin='lower')
    axes[0, 2].set_title('Anisotropy Ratio')
    plt.colorbar(im3, ax=axes[0, 2])

    # First three realizations
    for i in range(3):
        im = axes[1, i].imshow(results['fields'][i], origin='lower')
        axes[1, i].set_title(f'Realization {i + 1}')
        plt.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig('workflow_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Example workflow complete!")
    return results