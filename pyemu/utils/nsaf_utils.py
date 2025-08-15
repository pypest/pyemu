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
                        method='idw', layer=0, config_dict=None):
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
        layer_cp = cp_df[cp_df['layer'] - 1 == layer].copy()
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
    x_extent = np.max(grid_x) - np.min(grid_x)
    y_extent = np.max(grid_y) - np.min(grid_y)
    domain_extent = max(x_extent, y_extent)
    corrlen = np.full(len(grid_x), np.max([10000.0, domain_extent/3]))
    aniso = np.ones(len(grid_x))  # Isotropic
    bearing = np.zeros(len(grid_x))  # No rotation

    try:
        npts = lib.calc_kriging_factors_2d(
            cp_x, cp_y, cp_zones,
            grid_x, grid_y, grid_zones,
            int(config_dict.get("vartype", 2)),  # Exponential
            int(config_dict.get("krigtype", 1)),  # Ordinary kriging
            corrlen, aniso, bearing,
            config_dict.get("search_radius", 1e20),
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
                        config_dict=None, iids=None, area=None,
                        mean_col='mean', sd_col='sd', **kwargs):
    """
    Parameter interpolation and field generation combining NSAF tensor mathematics
    with pypestutils FIELDGEN2D_SVA for spatially correlated noise.

    FLEXIBLE version that works with any property by specifying column names.

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
    area : np.ndarray, optional
        Cell areas with shape (ny, nx)
    mean_col : str, default 'mean'
        Name of the mean column in conceptual points
    sd_col : str, default 'sd'
        Name of the standard deviation column in conceptual points

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
        layer_cp = cp_df[cp_df['layer'] - 1 == layer].copy()
    else:
        layer_cp = cp_df.copy()

    if len(layer_cp) == 0:
        raise ValueError(f"No conceptual points found for layer {layer}")

    # Validate required columns
    required_base_cols = ['x', 'y', 'major', 'bearing']
    missing_cols = [col for col in required_base_cols if col not in layer_cp.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for transverse/anisotropy
    if 'transverse' not in layer_cp.columns:
        if 'anisotropy' in layer_cp.columns:
            layer_cp['transverse'] = layer_cp['major'] / layer_cp['anisotropy']
        else:
            raise ValueError("Must have either 'transverse' or 'anisotropy' column")

    # Validate mean and sd columns
    if mean_col not in layer_cp.columns:
        raise ValueError(f"Mean column '{mean_col}' not found in conceptual points")
    if sd_col not in layer_cp.columns:
        raise ValueError(f"Standard deviation column '{sd_col}' not found in conceptual points")

    # Get grid dimensions and coordinates
    ny, nx = xcentergrid.shape
    grid_coords_2d = np.column_stack([xcentergrid.flatten(), ycentergrid.flatten()])

    print(f"Processing layer {layer} with {len(layer_cp)} conceptual points")
    print(f"Grid dimensions: {ny} x {nx} = {ny * nx} cells")
    print(f"Using columns: mean='{mean_col}', sd='{sd_col}'")

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
    cp_means = layer_cp[mean_col].values
    cp_sd = layer_cp[sd_col].values.clip(1e-8,None)

    print(f"Conceptual point data ranges:")
    print(f"  {mean_col}: [{cp_means.min():.3f}, {cp_means.max():.3f}]")
    print(f"  {sd_col}: [{cp_sd.min():.3f}, {cp_sd.max():.3f}]")

    # Step 4: Interpolate mean and sd using tensor-aware kriging
    print("\nStep 4: Interpolating mean and standard deviation fields...")

    # Configure kriging parameters
    if config_dict is None:
        config_dict = {
            'vartype': 'gaussian',
            'krigtype': 1,
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

    print(f"Interpolated field ranges:")
    print(f"  Mean: [{interp_means_2d.min():.3f}, {interp_means_2d.max():.3f}]")
    print(f"  SD: [{interp_sd_2d.min():.3f}, {interp_sd_2d.max():.3f}]")

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
            vartransform='n',
            config_dict=config_dict,
            ny=ny,
            nx=nx,
            area=area,
            iids=iids,
            active=kwargs.get('active', None)
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
    Tensor-aware kriging using pypestutils with anisotropic correlation structures.
    Clean version without automatic transform detection.
    """
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    # Get grid shape
    n_grid = len(grid_coords)
    n_dims = grid_coords.shape[1]

    print(f"    Tensor-aware kriging: {len(cp_coords)} conceptual points -> {n_grid} grid points")
    print(
        f"    Conceptual points range: X=[{cp_coords[:, 0].min():.1f}, {cp_coords[:, 0].max():.1f}], Y=[{cp_coords[:, 1].min():.1f}, {cp_coords[:, 1].max():.1f}]")
    print(
        f"    Grid points range: X=[{grid_coords[:, 0].min():.1f}, {grid_coords[:, 0].max():.1f}], Y=[{grid_coords[:, 1].min():.1f}, {grid_coords[:, 1].max():.1f}]")
    print(f"    Value range: [{cp_values.min():.6f}, {cp_values.max():.6f}], background: {background_value:.6f}")
    print(f"    Transform requested: {transform}")

    # Handle zones shape inference
    if zones is not None:
        shape = zones.shape
        if n_dims == 2:
            ny, nx = shape
        else:
            nz, ny, nx = shape
        print(f"    Using zones with shape: {shape}")
    else:
        # Infer grid dimensions
        x_coords = grid_coords[:, 0]
        y_coords = grid_coords[:, 1]

        unique_x = np.unique(x_coords)
        unique_y = np.unique(y_coords)
        nx = len(unique_x)
        ny = len(unique_y)

        if nx * ny != n_grid:
            side_length = int(np.sqrt(n_grid))
            nx = ny = side_length
            print(f"    Warning: Could not infer grid dimensions, using square grid {nx}x{ny}")
        else:
            print(f"    Inferred grid dimensions: ny={ny}, nx={nx}")

        shape = (ny, nx)

    # Convert tensors to geostatistical parameters for pypestutils
    print("    Converting tensors to geostatistical parameters...")
    bearing_deg, anisotropy_ratios, corrlen_major = _tensors_to_geostat_params(interp_tensors)

    # Prepare grid coordinates
    grid_x = grid_coords[:, 0]
    grid_y = grid_coords[:, 1]

    # Handle zones
    if zones is not None:
        zones_flat = zones.flatten().astype(int)
        cp_zones = _assign_cp_to_zones(cp_coords, grid_x, grid_y, zones_flat, nx)
    else:
        zones_flat = np.ones(len(grid_x), dtype=int)
        cp_zones = np.ones(len(cp_coords), dtype=int)

    # Handle transformation - NO automatic detection, just pass through
    cp_values_transformed = cp_values.copy()  # Always use values as-is

    # Set pypestutils transform flag
    if transform == 'log':
        transform_flag = "log"
        print(f"    Using log transform in pypestutils")
    else:
        transform_flag = "none"
        print(f"    Using no transform in pypestutils")

    print(f"    Input values to pypestutils: [{cp_values_transformed.min():.6f}, {cp_values_transformed.max():.6f}]")

    # Use pypestutils for kriging
    print("    Calling pypestutils for tensor-aware kriging...")

    def fix_zones_for_pypestutils(zones_flat, cp_zones, min_points):
        """
        Ensure every zone in the grid either:
        1) Has >= min_points control points, OR
        2) Gets merged with a viable neighboring zone
        """
        zones_fixed = zones_flat.copy()
        cp_zones_fixed = cp_zones.copy()

        unique_grid_zones = np.unique(zones_flat)

        zones_to_eliminate = []

        for zone in unique_grid_zones:
            cp_count = np.sum(cp_zones == zone)
            grid_count = np.sum(zones_flat == zone)

            if cp_count < min_points:
                print(f"    Zone {zone}: {cp_count} CPs, {grid_count} grid cells - ELIMINATING")
                zones_to_eliminate.append(zone)

        if zones_to_eliminate:
            # Find the zone with the most control points to use as the "receiver"
            viable_zones = [z for z in unique_grid_zones if z not in zones_to_eliminate]
            if viable_zones:
                # Pick the zone with most control points
                best_zone = max(viable_zones, key=lambda z: np.sum(cp_zones == z))
                print(f"    Consolidating {len(zones_to_eliminate)} bad zones into zone {best_zone}")

                for bad_zone in zones_to_eliminate:
                    # Move all grid cells from bad zone to best zone
                    zones_fixed[zones_flat == bad_zone] = best_zone
                    # Move any control points (though there shouldn't be many)
                    cp_zones_fixed[cp_zones == bad_zone] = best_zone
            else:
                # All zones are bad - just make everything zone 1
                print("    All zones insufficient - making single zone")
                zones_fixed = np.ones_like(zones_flat)
                cp_zones_fixed = np.ones_like(cp_zones)

        # Final verification
        final_zones = np.unique(zones_fixed)
        print(f"    Final zones: {final_zones}")
        for zone in final_zones:
            cp_count = np.sum(cp_zones_fixed == zone)
            grid_count = np.sum(zones_fixed == zone)
            print(f"      Zone {zone}: {cp_count} CPs, {grid_count} grid cells")

            if cp_count < min_points:
                print(f"      WARNING: Zone {zone} still has insufficient points!")

        return zones_fixed, cp_zones_fixed

    # Use it:
    zones_for_kriging, cp_zones_for_kriging = fix_zones_for_pypestutils(
        zones_flat, cp_zones, min_points
    )

    try:
        lib = PestUtilsLib()

        # Create temporary factor file
        fac_fname = "temp_tensor_aware_kriging.fac"
        if os.path.exists(fac_fname):
            os.remove(fac_fname)
        fac_ftype = "text"

        # Set up variogram type
        if variogram_model == 'exponential':
            vartype = 2
        elif variogram_model == 'gaussian':
            vartype = 3
        elif variogram_model == 'spherical':
            vartype = 1
        else:
            vartype = 2  # Default to exponential

        print(f"    Pypestutils parameters:")
        print(f"      Variogram: {variogram_model} (type {vartype})")
        print(f"      Transform: {transform_flag}")
        print(f"      Search radius: {max_search_radius}")
        print(f"      Max neighbors: {max_neighbors}")

        # Then use the modified zones in kriging
        npts = lib.calc_kriging_factors_2d(
            cp_coords[:, 0], cp_coords[:, 1], cp_zones_for_kriging,  # Modified zones
            grid_x, grid_y, zones_for_kriging,  # Modified zones
            vartype, 1, corrlen_major, anisotropy_ratios, bearing_deg,
            max_search_radius, max_neighbors, min_points,
            fac_fname, fac_ftype
        )

        # Calculate kriging factors using tensor-derived anisotropy
        # npts = lib.calc_kriging_factors_2d(
        #     cp_coords[:, 0], cp_coords[:, 1], cp_zones,
        #     grid_x, grid_y, zones_flat,
        #     vartype,  # Variogram type
        #     1,  # Ordinary kriging
        #     corrlen_major,  # Correlation lengths from tensors
        #     anisotropy_ratios,  # Anisotropy ratios from tensors
        #     bearing_deg,  # Bearings from tensors
        #     max_search_radius,
        #     max_neighbors,
        #     min_points,
        #     fac_fname, fac_ftype
        # )

        print(f"    Pypestutils kriging factors calculated for {npts} interpolation points")

        # Apply kriging
        result = lib.krige_using_file(
            fac_fname, fac_ftype,
            len(grid_x),
            1,  # Ordinary kriging
            transform_flag,  # Pass through the transform flag as specified
            cp_values_transformed,  # Use values exactly as provided
            background_value,  # Use background exactly as provided
            background_value  # Use no-interpolation value exactly as provided
        )

        # Clean up
        if os.path.exists(fac_fname):
            os.remove(fac_fname)
        lib.free_all_memory()

        # Get results - NO back-transformation here, pypestutils handles it
        interp_values_1d = result["targval"]

        print(f"    Pypestutils kriging complete")
        print(f"    Result range: [{interp_values_1d.min():.6f}, {interp_values_1d.max():.6f}]")

        return interp_values_1d.reshape(shape)

    except Exception as e:
        print(f"    Error: Pypestutils kriging failed: {e}")
        print(f"    Falling back to simple IDW interpolation...")

        # Fallback to simple IDW
        interp_values_1d = np.full(n_grid, background_value)

        for i in range(n_grid):
            distances = np.linalg.norm(cp_coords - grid_coords[i], axis=1)
            distances = np.maximum(distances, 1e-10)

            # Simple IDW weights
            weights = 1.0 / (distances ** 2)
            weights = weights / np.sum(weights)

            interp_values_1d[i] = np.sum(weights * cp_values_transformed)

        print(f"    IDW fallback complete")
        print(f"    Result range: [{interp_values_1d.min():.6f}, {interp_values_1d.max():.6f}]")

        return interp_values_1d.reshape(shape)


def _generate_stochastic_fields(grid_coords, mean_field, sd_field,
                                bearing, anisotropy, corrlen,
                                zones, n_realizations, vartransform,
                                config_dict, ny, nx, iids=None, area=None,
                                active=None):
    """
    Generate stochastic fields using pypestutils FIELDGEN2D_SVA.
    FINAL FIXED version with correct reshaping logic.
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

    # Flatten fields - ENSURE PROPER ORDERING
    mean_flat = mean_field.flatten().clip(1e-8, None)
    variance_native = sd_field.flatten() ** 2
    log_variance_flat = np.log(1 + variance_native / mean_flat ** 2)
    trans_flag = 1  # Log domain

    # Geostatistical parameters from tensors
    aa_values = corrlen.flatten()
    aniso_values = anisotropy.flatten()
    bearing_values = bearing.flatten()

    # FIXED: domain based on zones
    if active is None:
        active = np.ones(n_points, dtype=int)
    if area is None:
        area = np.ones(n_points)
    else:
        area = area.flatten()

    # Generate or use provided IIDs
    if iids is None:
        print("  Generating new IIDs...")
        iids = np.random.normal(0, 1, size=(n_points, n_realizations))
    else:
        print("  Using provided IIDs...")
        if iids.shape != (n_points, n_realizations):
            raise ValueError(f"IIDs shape {iids.shape} doesn't match expected {(n_points, n_realizations)}")

    try:
        # Set up pypestutils parameters
        lib = PestUtilsLib()
        lib.initialize_randgen(11235813)

        # Averaging function type
        avg_func = config_dict.get('averaging_function', 'gaussian')
        if isinstance(avg_func, dict):
            if 'pow' in avg_func:
                avg_flag = 4
                power_value = avg_func['pow']
            else:
                avg_flag, power_value = next(iter(avg_func.items()))
        else:
            power_value = 0
            if avg_func == 'exponential':
                avg_flag = 2
            elif avg_func == 'gaussian':
                avg_flag = 3
            elif avg_func == 'spherical':
                avg_flag = 1
            else:
                avg_flag = 2  # Default to exponential

        # Call FIELDGEN2D_SVA through pypestutils interface
        print(f"    Calling FIELDGEN2D_SVA for {n_realizations} realizations...")

        fields = lib.fieldgen2d_sva(
            x_coords, y_coords,
            area=area,
            active=active,
            mean=mean_flat,
            var=log_variance_flat,
            aa=aa_values,
            anis=aniso_values,
            bearing=bearing_values,
            transtype=trans_flag,
            avetype=avg_flag,
            power=power_value,
            nreal=n_realizations,
        )

        # Handle the output format from FIELDGEN2D_SVA
        if fields.shape == (n_points, n_realizations):
            # Output is (n_points, n_realizations) - transpose to (n_realizations, n_points)
            fields = fields.T  # Now shape is (n_realizations, n_points)
            # Reshape each realization to (ny, nx)
            fields = fields.reshape((n_realizations, ny, nx))

        elif fields.shape == (n_realizations * n_points,):
            # Output is flattened - reshape directly
            fields = fields.reshape((n_realizations, ny, nx))

        elif fields.shape == (n_realizations, n_points):
            # Output is already (n_realizations, n_points) - just reshape spatial dimensions
            fields = fields.reshape((n_realizations, ny, nx))

        else:
            raise ValueError(f"Unexpected fieldgen2d_sva output shape: {fields.shape}")

        print(f"    Successfully generated {n_realizations} realizations with shape {fields.shape}")

        # Clean up
        lib.free_all_memory()

        return fields

    except Exception as e:
        print(f"    Warning: FIELDGEN2D_SVA failed: {e}")
        print(f"    Using fallback method...")

        # Fallback to simple field generation
        fields = np.zeros((n_realizations, ny, nx))
        for real in range(n_realizations):
            current_iids = iids[:, real]
            field_values = _generate_simple_field(
                grid_coords, mean_flat, log_variance_flat,
                aa_values, aniso_values, bearing_values,
                vartransform, current_iids, ny, nx
            )
            fields[real] = field_values.reshape(ny, nx)

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
    IMPROVED simple fallback field generation if FIELDGEN2D_SVA fails.
    Fixed to eliminate checkering and provide proper spatial correlation.
    """
    print("    Using improved simple correlated field generation fallback...")

    n_points = len(grid_coords)

    # Convert bearing from degrees to radians for calculations
    bearing_rad = np.radians(bearing_values)

    # Create spatially correlated field using moving average approach
    correlated_field = np.zeros(n_points)

    # For computational efficiency, use a subset of points for correlation
    # This prevents the O(n²) complexity from being too slow
    max_correlation_points = min(2000, n_points)

    if n_points > max_correlation_points:
        # Sample points for correlation calculation
        correlation_indices = np.random.choice(n_points, max_correlation_points, replace=False)
        correlation_indices = np.sort(correlation_indices)
    else:
        correlation_indices = np.arange(n_points)

    # Apply spatial correlation
    for i in correlation_indices:
        # Get local correlation parameters
        local_aa = aa_values[i]
        local_aniso = aniso_values[i]
        local_bearing = bearing_rad[i]

        # Calculate distances to all points
        dx = grid_coords[:, 0] - grid_coords[i, 0]
        dy = grid_coords[:, 1] - grid_coords[i, 1]

        # Apply anisotropic transformation
        # Rotate coordinates to align with anisotropy ellipse
        cos_bear = np.cos(local_bearing)
        sin_bear = np.sin(local_bearing)

        # Rotate to principal axes (geological bearing: 0° = North, clockwise positive)
        dx_rot = dx * sin_bear + dy * cos_bear  # East component (major axis)
        dy_rot = -dx * cos_bear + dy * sin_bear  # North component (minor axis)

        # Scale by anisotropy (major axis has length aa, minor axis has length aa/aniso)
        minor_aa = local_aa / local_aniso
        dx_scaled = dx_rot / local_aa
        dy_scaled = dy_rot / minor_aa

        # Calculate anisotropic distance
        aniso_distances = np.sqrt(dx_scaled ** 2 + dy_scaled ** 2)

        # Exponential correlation function
        correlation_weights = np.exp(-aniso_distances)

        # Normalize weights
        correlation_weights = correlation_weights / np.sum(correlation_weights)

        # Apply correlation to the random field
        correlated_field[i] = np.sum(correlation_weights * iids)

    # For points not in correlation_indices, use interpolation
    if n_points > max_correlation_points:
        # Simple interpolation for remaining points
        remaining_indices = np.setdiff1d(np.arange(n_points), correlation_indices)

        for i in remaining_indices:
            # Find nearest correlation points
            distances_to_corr = np.sum((grid_coords[correlation_indices] - grid_coords[i]) ** 2, axis=1)
            nearest_corr_idx = correlation_indices[np.argmin(distances_to_corr)]
            correlated_field[i] = correlated_field[nearest_corr_idx]

    # Scale by local standard deviation
    stochastic_component = correlated_field * np.sqrt(variance_flat)

    # Combine with mean field
    if vartransform == 'log':
        # Log domain: mean * exp(stochastic_component)
        field_values = mean_flat * np.exp(stochastic_component)
    else:
        # Normal domain: mean + stochastic_component
        field_values = mean_flat + stochastic_component

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
