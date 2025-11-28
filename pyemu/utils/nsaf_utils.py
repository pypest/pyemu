import numpy as np
import pandas as pd
from scipy.linalg import logm, expm
import os
import matplotlib as mpl

mpl.use('Tkagg')
import matplotlib.pyplot as plt


def generate_single_layer(conceptual_points, modelgrid, iids=None, zones=None,
                          tensor_interp='idw', transform='log', layer=1,
                          boundary_smooth=False, boundary_enhance=False,
                          mean_col='mean', sd_col='sd', sd_is_log_space=False,
                          n_realizations=1, random_seed=None, save_dir='.'):
    """
    Generate single layer using NSAF approach with boundary adjustments.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points DataFrame
    modelgrid : flopy.discretization.grid
        Flopy modelgrid object
    iids : np.ndarray, optional
        Pre-generated noise, shape (ny*nx, )
    zones : np.ndarray, optional
        Zone IDs, shape (ny, nx)
    tensor_interp : str, default 'idw'
        Tensor interpolation method
    transform : str, default 'log'
        Transformation: 'log' or 'none'
    boundary_smooth : bool, default True
        Apply smoothing to mean at zone boundaries
    boundary_enhance : bool, default True
        Apply variance enhancement at zone boundaries
    mean_col : str, default 'mean'
        Name of mean column in conceptual points
    sd_col : str, default 'sd'
        Name of sd column in conceptual points
    n_realizations : int, default 1
        Number of realizations. Only used when iids=None (FIELDGEN2D_SVA mode).
        When iids is provided (PEST mode), only 1 realization is generated.
    random_seed : int, optional
        Random seed for reproducibility. Only used when iids=None.
        When iids is provided, the iids themselves control the realization.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'field': Single field (ny, nx) if n_realizations=1 or iids provided
        - 'fields': Multiple fields (n_realizations, ny, nx) if n_realizations>1 and iids=None
        - 'mean', 'sd', 'tensors', etc.
    """

    results = apply_nsaf_hyperpars(
        conceptual_points=conceptual_points,
        modelgrid=modelgrid,
        zones=zones,
        out_filename=None,
        n_realizations=n_realizations,
        random_seed=random_seed,
        layer=layer,
        transform=transform,
        boundary_smooth=boundary_smooth,
        boundary_enhance=boundary_enhance,
        tensor_method=tensor_interp,
        config_dict=None,
        iids=iids,
        mean_col=mean_col,
        sd_col=sd_col,
        sd_is_log_space=sd_is_log_space
    )

    return results


def apply_nsaf_hyperpars(conceptual_points, modelgrid, zones=None, out_filename=None,
                         n_realizations=1, random_seed=None, layer=0, transform="none",
                         boundary_smooth=False, boundary_enhance=False,
                         tensor_method='krig', config_dict=None, iids=None,
                         mean_col='mean', sd_col='sd',
                         sd_is_log_space=False, **kwargs):
    """
    Parameter interpolation and field generation combining NSAF tensor mathematics
    with pypestutils FIELDGEN2D_SVA for spatially correlated noise.

    REFACTORED to use modelgrid consistently throughout.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points DataFrame
    modelgrid : flopy.discretization.grid
        Flopy modelgrid object containing grid information
    zones : np.ndarray, optional
        Zone array, shape (ny, nx)
    out_filename : str, optional
        Base filename for output (if None, returns arrays only)
    n_realizations : int, default 1
        Number of stochastic realizations. Only used when iids=None.
    random_seed : int, optional
        Random seed for reproducibility. Only used when iids=None.
    layer : int, default 0
        Layer number to process
    transform : str, default "none"
        Transformation: "none" or "log"
    boundary_smooth : bool, or dict default False
        Apply smoothing to mean at zone boundaries
    boundary_enhance : bool, default True
        Apply variance enhancement at zone boundaries
    tensor_method : str, default 'krig'
        Tensor interpolation method: 'krig', 'idw', or 'nearest'
    config_dict : dict, optional
        Configuration parameters for pypestutils
    iids : np.ndarray, optional
        Pre-generated IIDs with shape (ny*nx, n_realizations)
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

    # Extract grid information from modelgrid
    grid_info = _extract_grid_info(modelgrid)
    ny = grid_info['ny']
    nx = grid_info['nx']
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']

    # Load conceptual points for this layer
    if 'layer' in conceptual_points.columns:
        layer_cp = conceptual_points[conceptual_points['layer'] == layer].copy()
    else:
        layer_cp = conceptual_points.copy()

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

    print(f"Processing layer {layer} with {len(layer_cp)} conceptual points")
    print(f"Grid dimensions: {ny} x {nx} = {ny * nx} cells")
    print(f"Using columns: mean='{mean_col}', sd='{sd_col}'")

    # === ZONE ASSIGNMENT - DO THIS ONCE, EARLY ===
    n_grid = ny * nx
    cp_coords = np.column_stack([layer_cp['x'].values, layer_cp['y'].values])

    if zones is not None:
        # Zones array provided - assign CPs to zones by nearest grid point
        print("  Assigning conceptual points to zones from zone array...")
        zones_flat = zones.flatten().astype(int)
        cp_zones = _assign_cp_to_zones(cp_coords, xcentergrid, ycentergrid, zones_flat, nx)
        layer_cp['zone'] = cp_zones  # Add to dataframe
        print(f"    Assigned CPs to zones: {np.unique(cp_zones)}")
    elif 'zone' in layer_cp.columns:
        # Use existing zone column from conceptual points
        print("  Using 'zone' column from conceptual points...")
        zones_flat = None  # No grid-based zones
        cp_zones = layer_cp['zone'].values.astype(int)
        print(f"    CP zones: {np.unique(cp_zones)}")
    else:
        # No zones - use zone=1 for everything
        print("  No zones provided - using zone=1 for all points")
        zones_flat = np.ones(n_grid, dtype=int)
        cp_zones = np.ones(len(layer_cp), dtype=int)
        layer_cp['zone'] = cp_zones

    # Verify we have zone information
    if 'zone' not in layer_cp.columns:
        layer_cp['zone'] = cp_zones

    # Step 1: Interpolate tensors
    print("\nStep 1: Interpolating tensor field...")
    tensors = interpolate_tensors(
        layer_cp, modelgrid, zones=zones,
        method=tensor_method, layer=layer,
        config_dict=config_dict
    )

    # Step 2: Convert tensors to geostatistical parameters
    print("\nStep 2: Converting tensors to geostatistical parameters...")
    bearing_deg, anisotropy, corrlen_major = _tensors_to_geostat_params(tensors)

    # Step 3: Prepare conceptual point data
    cp_means = layer_cp[mean_col].values
    cp_sd = layer_cp[sd_col].values.clip(1e-8, None)

    print(f"Conceptual point data ranges:")
    print(f"  {mean_col}: [{cp_means.min():.3f}, {cp_means.max():.3f}]")
    print(f"  {sd_col}: [{cp_sd.min():.3f}, {cp_sd.max():.3f}]")
    print(f"  Zones: {sorted(np.unique(layer_cp['zone']))}")

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
    transform_str = 'log' if transform == 'log' else None
    min_mean = 1e-8 if transform == 'log' else None
    max_mean = 1e8 if transform == 'log' else None
    bounds = (min_mean, max_mean)

    # Interpolate mean
    print("  Interpolating mean...")
    interp_means_2d = tensor_aware_kriging(
        cp_coords, cp_means, modelgrid, tensors,
        variogram_model='exponential',
        background_value=np.mean(cp_means), max_search_radius=1e20,
        min_points=3, transform=transform_str, bounds=bounds,
        max_neighbors=4, zones=zones
    )

    # Apply boundary smoothing to mean
    if boundary_smooth and zones is not None:
        if isinstance(boundary_smooth, dict):
            smooth_params = boundary_smooth
        else:
            smooth_params = {'transition_cells': 2}

        print("  Smoothing mean at geological boundaries...")
        interp_means_2d = create_boundary_modified_scalar(
            interp_means_2d, zones, transform=transform_str,
            transition_cells=smooth_params.get('transition_cells', 2),
            mode='smooth'
        )

    min_sd = 1e-8 if transform == 'log' else None
    max_sd = 1e8 if transform == 'log' else None
    bounds = (min_sd, max_sd)

    # Interpolate standard deviation
    print("  Interpolating standard deviation...")
    sd_transform = None if sd_is_log_space else transform_str
    interp_sd_2d = tensor_aware_kriging(
        cp_coords, cp_sd, modelgrid, tensors,
        variogram_model='exponential',
        background_value=np.mean(cp_sd), max_search_radius=1e20,
        min_points=3, transform=sd_transform, bounds=bounds,
        max_neighbors=4, zones=zones,
    )

    # Apply boundary enhancement to sd
    if boundary_enhance and zones is not None:
        if isinstance(boundary_enhance, dict):
            smooth_params = boundary_enhance
        else:
            smooth_params = {'peak_increase': 2.0, 'transition_cells': 2}

        print("  Enhancing variance at geological boundaries...")
        interp_sd_2d = create_boundary_modified_scalar(
            interp_sd_2d, zones, transform=transform_str,
            peak_increase=smooth_params.get('peak_increase', 2.0),
            transition_cells=smooth_params.get('transition_cells', 2),
            mode='enhance'
        )

    print(f"Interpolated field ranges:")
    print(f"  Mean: [{interp_means_2d.min():.3f}, {interp_means_2d.max():.3f}]")
    print(f"  SD: [{interp_sd_2d.min():.3f}, {interp_sd_2d.max():.3f}]")

    # Step 5: Generate stochastic fields
    print(f"\nStep 5: Generating stochastic realization(s)...")

    if iids is not None:
        print("  Note: iids provided - generating single realization (n_realizations parameter ignored)")
        n_realizations = 1  # Force to 1 when iids provided

    if n_realizations > 0:
        # Get active cells if available
        active = kwargs.get('active', None)
        if active is None and hasattr(modelgrid, 'idomain'):
            active = modelgrid.idomain[
                layer - 1].flatten() if modelgrid.idomain.ndim == 3 else modelgrid.idomain.flatten()

        fields = _generate_stochastic_field(
            modelgrid=modelgrid,
            mean_field=interp_means_2d,
            sd_field=interp_sd_2d,
            iids=iids,
            bearing=bearing_deg,
            anisotropy=anisotropy,
            corrlen=corrlen_major,
            sd_is_log_space=sd_is_log_space,
            config_dict=config_dict,
            active=active,
            n_realizations=n_realizations,
            random_seed=random_seed
        )
    else:
        fields = None

    # Prepare output - handle both single and multiple realizations
    results = {
        'mean': interp_means_2d,
        'sd': interp_sd_2d,
        'tensors': tensors,
        'bearing': bearing_deg.reshape(ny, nx),
        'anisotropy': anisotropy.reshape(ny, nx),
        'corrlen': corrlen_major.reshape(ny, nx)
    }

    # Add field(s) to results
    if fields is not None:
        if n_realizations == 1:
            results['field'] = fields  # Single field (ny, nx)
        else:
            results['fields'] = fields  # Multiple fields (n_realizations, ny, nx)
            results['field'] = fields[0]  # Also provide first one as 'field' for backward compat

    # Save to files if requested
    if out_filename is not None:
        save_results(results, out_filename, transform)

    print("=== Interpolation complete ===")
    return results

def load_conceptual_points(cp_file):
    """Load conceptual points from CSV file or DataFrame - SIMPLIFIED.

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

    # Validate required columns (excluding anisotropy/transverse for now)
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


def create_2d_tensors(theta, major, minor):
    """Create 2x2 anisotropy tensors from geostatistical parameters - NO CHANGES to math.

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


def interpolate_tensors(cp_file, modelgrid, zones=None, method='idw',
                        layer=0, config_dict=None):
    """
    Tensor field interpolation using NSAF log-Euclidean mathematics.
    REFACTORED to use modelgrid consistently.

    Parameters
    ----------
    cp_file : str or pd.DataFrame
        Conceptual points file path or DataFrame
    modelgrid : flopy.discretization.grid
        Flopy modelgrid object
    zones : np.ndarray, optional
        2D zone array (ny, nx)
    method : str, default 'idw'
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
    # todo change to warning, use other method

    # Extract grid info
    grid_info = _extract_grid_info(modelgrid)
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']
    ny = grid_info['ny']
    nx = grid_info['nx']

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
    xcentergrid = xcentergrid.flatten()
    ycentergrid = ycentergrid.flatten()

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
        cp_zones = _assign_cp_to_zones(cp_coords, xcentergrid, ycentergrid, zones_flat, nx)
    else:
        zones_flat = np.ones(len(xcentergrid), dtype=int)
        cp_zones = np.ones(len(layer_cp), dtype=int)

    # Interpolate tensors using log-Euclidean approach
    if method == 'krig':
        interp_tensors = _interpolate_tensors_kriging(
            cp_coords, cp_tensors, cp_zones,
            xcentergrid, ycentergrid, zones_flat,
            config_dict, nx, ny
        )
    elif method == 'idw':
        interp_tensors = _interpolate_tensors_idw(
            cp_coords, cp_tensors, cp_zones,
            xcentergrid, ycentergrid, zones_flat,
            nx, ny
        )
    elif method == 'nearest':
        interp_tensors = _interpolate_tensors_nearest(
            cp_coords, cp_tensors, cp_zones,
            xcentergrid, ycentergrid, zones_flat,
            nx, ny
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return interp_tensors


def tensor_aware_kriging(cp_coords, cp_values, modelgrid, interp_tensors,
                         variogram_model='exponential',
                         background_value=0.0, max_search_radius=1e20, min_points=1,
                         max_neighbors=4, transform=None, bounds=(1e-8, 1e8),
                         zones=None, n_dims=2):
    """
    Tensor-aware kriging using pypestutils
    """
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    # Get grid shape
    grid_info = _extract_grid_info(modelgrid)
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']
    n_grid = np.prod(xcentergrid.shape)

    print(f"    Tensor-aware kriging: {len(cp_coords)} conceptual points -> {len(xcentergrid)} grid points")
    print(
        f"    Conceptual points range: X=[{cp_coords[:, 0].min():.1f}, {cp_coords[:, 0].max():.1f}], Y=[{cp_coords[:, 1].min():.1f}, {cp_coords[:, 1].max():.1f}]")
    print(f"    Grid points range: X=[{xcentergrid.min():.1f}, {xcentergrid.max():.1f}], Y=[{ycentergrid.min():.1f}, {ycentergrid.max():.1f}]")
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
        unique_x = np.unique(xcentergrid)
        unique_y = np.unique(ycentergrid)
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

    # Handle zones
    if zones is not None:
        zones_flat = zones.flatten().astype(int)
        cp_zones = _assign_cp_to_zones(cp_coords, xcentergrid, ycentergrid, zones_flat, nx)
    else:
        zones_flat = np.ones(n_grid, dtype=int)
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
            xcentergrid.flatten(), ycentergrid.flatten(), zones_for_kriging,  # Modified zones
            vartype, 1, corrlen_major, anisotropy_ratios, bearing_deg,
            max_search_radius, max_neighbors, min_points,
            fac_fname, fac_ftype
        )

        print(f"    Pypestutils kriging factors calculated for {npts} interpolation points")

        # Apply kriging
        result = lib.krige_using_file(
            fac_fname, fac_ftype,
            len(xcentergrid.flatten()),
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

        if bounds is not None:
            interp_values_1d = np.clip(interp_values_1d, min=bounds[0], max=bounds[1])

        return interp_values_1d.reshape(shape)

    except Exception as e:
        print(f"    Error: Pypestutils kriging failed: {e}")
        print(f"    Falling back to simple IDW interpolation...")

        # Fallback to simple IDW
        interp_values_1d = np.full(n_grid, background_value)

        for i in range(len(ycentergrid.flatten())):
            distances = np.linalg.norm(cp_coords - np.stack([xcentergrid.flatten(), ycentergrid.flatten()]).T[i], axis=1)
            distances = np.maximum(distances, 1e-10)

            # Simple IDW weights
            weights = 1.0 / (distances ** 2)
            weights = weights / np.sum(weights)

            interp_values_1d[i] = np.sum(weights * cp_values_transformed)

        print(f"    IDW fallback complete")
        print(f"    Result range: [{interp_values_1d.min():.6f}, {interp_values_1d.max():.6f}]")

        return interp_values_1d.reshape(shape)


def create_boundary_modified_scalar(base_field, zones,
                                    peak_increase=0.3, transition_cells=5, mode="enhance",
                                    transform=None):
    """
    Modify scalar field values near geological zone boundaries.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if mode not in ("enhance", "smooth"):
        raise ValueError("mode must be 'enhance' or 'smooth'")

    if zones.shape != base_field.shape:
        raise ValueError(f"Zones shape {zones.shape} must match field shape {base_field.shape}")

    boundary_mask, _ = _detect_zone_boundaries(zones)
    distance = distance_transform_edt(~boundary_mask)
    transition_mask = distance <= transition_cells

    modified = base_field.copy()
    smoothed_field = gaussian_filter(base_field, sigma=transition_cells)

    if mode == "enhance":
        factor = 1 - distance[transition_mask] / transition_cells
        modified[transition_mask] = smoothed_field[transition_mask] * (1 + peak_increase * factor)

    elif mode == "smooth":
        # Smooth field with Gaussian filter
        #smoothed_field = gaussian_filter(base_field, sigma=transition_cells)
        weight = 1 - distance[transition_mask] / transition_cells
        modified[transition_mask] = (
                weight * smoothed_field[transition_mask] +
                (1 - weight) * base_field[transition_mask]
        )

    print(f"    {'Enhanced' if mode == 'enhance' else 'Smoothed'} {np.count_nonzero(transition_mask)} points near boundaries")
    return modified


def save_results(results, out_filename):
    """Save results to files in PyEMU-compatible format."""

    # Save mean field
    mean_file = f"{out_filename}_mean.txt"
    np.savetxt(mean_file, results['mean'], fmt="%20.8E")
    print(f"  Saved mean field to {mean_file}")

    # Save standard deviation field
    sd_file = f"{out_filename}_sd.txt"
    np.savetxt(sd_file, results['sd'], fmt="%20.8E")
    print(f"  Saved SD field to {sd_file}")

    # Save geostatistical parameter field
    for param in ['bearing', 'anisotropy', 'corrlen']:
        param_file = f"{out_filename}_{param}.txt"
        np.savetxt(param_file, results[param], fmt="%20.8E")
        print(f"  Saved {param} field to {param_file}")

    # Save stochastic realizations
    if results['field'] is not None:
        field_file = f"{out_filename}.arr"
        np.savetxt(field_file, results['field'], fmt="%20.8E")
        print(f"  Saved realization to {field_file}")


def _detect_zone_boundaries(zones):
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


def _ensure_positive_definite(tensor, min_eigenval=1e-8):
    """Ensure tensor is positive definite - NO CHANGES."""
    eigenvals, eigenvecs = np.linalg.eigh(tensor)
    eigenvals_reg = np.maximum(eigenvals, min_eigenval)
    return eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T


def _extract_grid_info(modelgrid):
    """Extract grid information from flopy modelgrid object.

    Parameters
    ----------
    modelgrid : flopy.discretization.grid
        Flopy modelgrid object

    Returns
    -------
    dict
        Dictionary with grid information including:
        - xcentergrid: X coordinates of cell centers
        - ycentergrid: Y coordinates of cell centers
        - nx, ny: Grid dimensions
        - area: Cell areas
        - grid_type: Type of grid
    """
    grid_info = {}

    # Get grid type
    grid_info['grid_type'] = modelgrid.grid_type

    # Get cell centers
    if hasattr(modelgrid, 'xcellcenters'):
        grid_info['xcentergrid'] = modelgrid.xcellcenters
        grid_info['ycentergrid'] = modelgrid.ycellcenters
    else:
        raise AttributeError("modelgrid missing xcellcenters/ycellcenters")

    # Get dimensions
    if modelgrid.grid_type == 'structured':
        grid_info['ny'] = modelgrid.nrow
        grid_info['nx'] = modelgrid.ncol
        # Calculate areas from delr, delc
        if hasattr(modelgrid, 'delr') and hasattr(modelgrid, 'delc'):
            delr = modelgrid.delr
            delc = modelgrid.delc
            grid_info['area'] = np.outer(delc, delr)
        else:
            # Uniform area assumption
            grid_info['area'] = np.ones((grid_info['ny'], grid_info['nx']))
    elif modelgrid.grid_type == 'vertex': # untested
        # Vertex grid - reshape if needed
        xc = grid_info['xcentergrid']
        yc = grid_info['ycentergrid']
        if len(xc.shape) == 1:
            grid_info['xcentergrid'] = xc.reshape(-1, 1)
            grid_info['ycentergrid'] = yc.reshape(-1, 1)
            grid_info['nx'] = 1
            grid_info['ny'] = xc.shape[0]
        else:
            grid_info['ny'], grid_info['nx'] = xc.shape
        # Calculate areas for vertex grid
        grid_info['area'] = _calculate_vertex_areas(modelgrid)
    else:
        # Fallback for other grid types
        xc = grid_info['xcentergrid']
        grid_info['ny'], grid_info['nx'] = xc.shape
        grid_info['area'] = np.ones((grid_info['ny'], grid_info['nx']))

    # get active domain
    if hasattr(modelgrid, 'idomain'):
        grid_info['idomain'] = modelgrid.idomain
    else:
        raise AttributeError("modelgrid missing idomain")

    return grid_info


def _calculate_vertex_areas(modelgrid):
    """Calculate cell areas for vertex grid using shoelace formula."""
    ncpl = modelgrid.ncpl if hasattr(modelgrid, 'ncpl') else len(modelgrid.xcellcenters)
    areas = np.zeros(ncpl)

    for i in range(ncpl):
        try:
            verts = modelgrid.get_cell_vertices(i)
            # Shoelace formula
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            area = 0.5 * abs(sum(x[j] * y[j + 1] - x[j + 1] * y[j] for j in range(-1, len(x) - 1)))
            areas[i] = area
        except:
            areas[i] = 1.0  # Default area

    # Reshape if needed
    if hasattr(modelgrid, 'ncpl'):
        return areas.reshape(-1, 1)
    else:
        return areas.reshape(modelgrid.nrow, modelgrid.ncol)


def _interpolate_tensors_kriging(cp_coords, cp_tensors, cp_zones,
                                 xcentergrid, ycentergrid, zones_flat, config_dict, nx, ny):
    """Interpolate tensors using pypestutils kriging with log-Euclidean approach."""
    n_grid = np.prod(xcentergrid.shape)
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
        zone_xcentergrid = xcentergrid[zone_indices]
        zone_ycentergrid = ycentergrid[zone_indices]

        # Interpolate each tensor component separately
        zone_tensors = np.zeros((len(zone_indices), 2, 2))

        for i in range(2):
            for j in range(i, 2):  # Only upper triangle, then mirror
                component_values = zone_log_tensors[:, i, j].real

                # Use pypestutils for kriging
                try:
                    from pypestutils.pestutilslib import PestUtilsLib
                    interp_component = _krig_component_pypestutils(
                        zone_cp_coords[:, 0], zone_cp_coords[:, 1],
                        component_values, zone_xcentergrid, zone_ycentergrid,
                        config_dict
                    )
                except Exception as e:
                    raise Exception(f"Error importing pypestutils, using python based method: {e}")
                    # todo: change to warning use other method

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


def _krig_component_pypestutils(cp_x, cp_y, cp_values, xcentergrid, ycentergrid, config_dict):
    """Use pypestutils to krig a single tensor component."""
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")
        # todo: change to warning use other method

    lib = PestUtilsLib()

    # Create temporary factor file
    fac_fname = "temp_tensor_component.fac"
    if os.path.exists(fac_fname):
        os.remove(fac_fname)
    fac_ftype = "text"

    # All points in same zone for component interpolation
    cp_zones = np.ones(len(cp_x), dtype=int)
    grid_zones = np.ones(len(xcentergrid), dtype=int)

    # Use isotropic parameters for component interpolation
    x_extent = np.max(xcentergrid) - np.min(xcentergrid)
    y_extent = np.max(ycentergrid) - np.min(ycentergrid)
    domain_extent = max(x_extent, y_extent)
    corrlen = np.full(len(xcentergrid), np.max([10000.0, domain_extent/3]))
    aniso = np.ones(len(xcentergrid))  # Isotropic
    bearing = np.zeros(len(xcentergrid))  # No rotation

    try:
        npts = lib.calc_kriging_factors_2d(
            cp_x, cp_y, cp_zones,
            xcentergrid, ycentergrid, grid_zones,
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
            len(xcentergrid),
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
        return _idw_component(cp_x, cp_y, cp_values, xcentergrid, ycentergrid)


def _idw_component(cp_x, cp_y, cp_values, xcentergrid, ycentergrid, power=2):
    """Inverse distance weighting fallback for component interpolation."""
    result = np.zeros(len(xcentergrid))

    for i in range(len(xcentergrid)):
        distances = np.sqrt((cp_x - xcentergrid[i]) ** 2 + (cp_y - ycentergrid[i]) ** 2)
        distances = np.maximum(distances, 1e-10)  # Avoid division by zero

        weights = 1.0 / (distances ** power)
        result[i] = np.sum(weights * cp_values) / np.sum(weights)

    return result


def _interpolate_tensors_idw(cp_coords, cp_tensors, cp_zones,
                             xcentergrid, ycentergrid, zones_flat, nx, ny):
    """IDW interpolation using log-Euclidean approach."""
    n_grid = np.prod(xcentergrid.shape)
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
            gx, gy = xcentergrid[idx], ycentergrid[idx]
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
                                 xcentergrid, ycentergrid, zones_flat, nx, ny):
    """Nearest neighbor interpolation."""
    n_grid = np.prod(xcentergrid.shape)
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
            gx, gy = xcentergrid[idx], ycentergrid[idx]
            distances = np.sqrt((zone_cp_coords[:, 0] - gx) ** 2 +
                                (zone_cp_coords[:, 1] - gy) ** 2)
            nearest_idx = np.argmin(distances)
            result_tensors[idx] = zone_cp_tensors[nearest_idx]

    return result_tensors


def _assign_cp_to_zones(cp_coords, xcentergrid, ycentergrid, zones_flat, nx):
    """Assign conceptual points to zones based on nearest grid point."""
    grid_coords = np.column_stack([xcentergrid.flatten(), ycentergrid.flatten()])
    cp_zones = np.zeros(len(cp_coords), dtype=int)

    for i, cp_coord in enumerate(cp_coords):
        distances = np.sum((grid_coords - cp_coord) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        cp_zones[i] = zones_flat[closest_idx]

    return cp_zones


def _generate_stochastic_field(modelgrid, mean_field, sd_field,
                               bearing, anisotropy, corrlen,
                               sd_is_log_space, config_dict, iids=None, area=None,
                               active=None, n_realizations=1, random_seed=None):
    """
    Generate stochastic field realization(s).

    Parameters
    ----------

    Note: FIELDGEN2D_SVA1 is required for external iid files but not yet in pypestutils.
    When iids are provided, uses Python fallback implementation.
    When iids are None, uses FIELDGEN2D_SVA but only returns first realization.

    Parameters
    ----------
    modelgrid : flopy modelgrid
    mean_field : np.ndarray
        Mean field values, shape (ny, nx)
    sd_field : np.ndarray
        Standard deviation field values, shape (ny, nx)
    bearing : np.ndarray
        Bearing field values in degrees, shape (ny, nx)
    anisotropy : np.ndarray
        Anisotropy field values, shape (ny, nx)
    corrlen : np.ndarray
        Correlation length field values, shape (ny, nx)
    sd_is_log_space : str
        how stochasticity is expressed, 0 for native, 1 for log
    config_dict : dict
        Configuration dictionary with averaging function settings
    iids : np.ndarray, optional
        Independent identically distributed random values, shape (ny*nx,)
        If provided, uses fallback method. If None, uses FIELDGEN2D_SVA.
    area : np.ndarray, optional
        Cell areas, shape (ny*nx,)
    active : np.ndarray, optional
        Active domain mask, shape (ny*nx,)
    n_realizations : int, default 1
        Number of realizations to generate. Only used when iids=None (FIELDGEN2D_SVA mode).
    random_seed : int, optional
        Random seed for reproducibility. Only used when iids=None (FIELDGEN2D_SVA mode).

    Returns
    -------
    np.ndarray
        - If iids provided OR n_realizations=1: shape (ny, nx)
        - If iids=None AND n_realizations>1: shape (n_realizations, ny, nx)
    """
    try:
        from pypestutils.pestutilslib import PestUtilsLib
        plib = PestUtilsLib()
        # Initialize with seed if provided
        if random_seed is not None:
            plib.initialize_randgen(random_seed)
        else:
            plib.initialize_randgen(11235813)  # Default seed
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    # Prepare grid data
    grid_info = _extract_grid_info(modelgrid)
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']
    ny = grid_info['ny']
    nx = grid_info['nx']
    active = active.astype(int)
    if area is None:
        area = grid_info['area']

    n_points = ny * nx

    # Flatten fields
    mean_flat = mean_field.flatten().clip(1e-8, None)  # Always native space
    sd_flat = sd_field.flatten()

    # Compute variance and set transtype based on what space sd_field is in
    if sd_is_log_space:
        # SD is in log space, variance is sd^2
        variance_flat = sd_flat ** 2
        trans_flag = 1  # Tell FIELDGEN2D_SVA: variance is in log space
    else:
        # SD is in native space, variance is sd^2
        variance_flat = sd_flat ** 2
        trans_flag = 0  # Tell FIELDGEN2D_SVA: variance is in native space

    # Geostatistical parameters from tensors
    aa_values = corrlen.flatten()
    aniso_values = anisotropy.flatten()
    bearing_values = bearing.flatten()

    if area is not None:
        area = area.flatten()

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

    # Handle IID-based generation (PEST mode - always single realization)
    if iids is not None:
        print("  Using provided IIDs for single realization (PEST mode)...")
        if n_realizations > 1:
            print("    Note: n_realizations>1 ignored when iids are provided")

        if iids.shape[0] != n_points:
            raise ValueError(f"IIDs shape {iids.shape} doesn't match expected ({n_points},)")

        print(f"    FIELDGEN2D_SVA1 (with external iids) not available, using Python fallback...")

        # Use fallback method for single realization
        field_values = _moving_average_field(
            modelgrid, mean_flat, variance_flat,
            aa_values, aniso_values, bearing_values,
            sd_is_log_space, iids,
            avg_flag=avg_flag,
            power_value=power_value,
            active=active
        )
        field = field_values.reshape(ny, nx)

        return field  # Shape: (ny, nx)

    # Handle FIELDGEN2D_SVA generation (internal random generation)
    else:
        try:
            print(f"  Generating {n_realizations} realization(s) using FIELDGEN2D_SVA...")
            if random_seed is not None:
                print(f"    Using random seed: {random_seed}")

            fields = plib.fieldgen2d_sva(
                xcentergrid.flatten(), ycentergrid.flatten(),
                area=area,
                active=active,
                mean=mean_flat,
                var=variance_flat,
                aa=aa_values,
                anis=aniso_values,
                bearing=bearing_values,
                transtype=sd_is_log_space,
                avetype=avg_flag,
                power=power_value,
                nreal=n_realizations,  # Pass n_realizations to FIELDGEN2D_SVA
            )

            # Handle output shapes
            if n_realizations == 1:
                # Single realization - return (ny, nx)
                if fields.shape == (n_points, 1):
                    field = fields[:, 0].reshape(ny, nx)
                elif fields.shape == (1, n_points):
                    field = fields[0, :].reshape(ny, nx)
                elif fields.shape == (n_points,):
                    field = fields.reshape(ny, nx)
                else:
                    raise ValueError(f"Unexpected fieldgen2d_sva output shape: {fields.shape}")

                print(f"    Successfully generated field with shape {field.shape}")
                plib.free_all_memory()
                return field  # Shape: (ny, nx)

            else:
                # Multiple realizations - return (n_realizations, ny, nx)
                if fields.shape == (n_points, n_realizations):
                    # Transpose and reshape
                    fields_3d = fields.T.reshape(n_realizations, ny, nx)
                elif fields.shape == (n_realizations, n_points):
                    # Just reshape
                    fields_3d = fields.reshape(n_realizations, ny, nx)
                else:
                    raise ValueError(f"Unexpected fieldgen2d_sva output shape: {fields.shape}")

                print(f"    Successfully generated {n_realizations} fields with shape {fields_3d.shape}")
                plib.free_all_memory()
                return fields_3d  # Shape: (n_realizations, ny, nx)

        except Exception as e:
            print(f"    Error: FIELDGEN2D_SVA failed: {e}")

            if iids is None:
                print(f"    Falling back to Python with random iids...")
                # Generate random iids for fallback
                if random_seed is not None:
                    np.random.seed(random_seed)
                random_iids = np.random.normal(0, 1, size=n_points)

                field_values = _moving_average_field(
                    modelgrid, mean_flat, variance_flat,
                    aa_values, aniso_values, bearing_values,
                    sd_is_log_space, random_iids,
                    avg_flag=avg_flag,
                    power_value=power_value
                )
                field = field_values.reshape(ny, nx)
                return field  # Shape: (ny, nx)
            else:
                raise ValueError(f"Unexpected fieldgen2d_sva output shape: {fields.shape}")

            print(f"    Successfully generated field with shape {field.shape}")

            # Clean up
            plib.free_all_memory()

            return field

        except Exception as e:
            print(f"    Error: FIELDGEN2D_SVA failed: {e}")
            print(f"    Falling back to simple field generation with random iids...")

            # Generate random iids and use fallback
            random_iids = np.random.normal(0, 1, size=n_points)
            field_values = _moving_average_field(
                modelgrid, mean_flat, variance_flat,
                aa_values, aniso_values, bearing_values,
                sd_is_log_space, random_iids,
                avg_flag=avg_flag,
                power_value=power_value
            )
            field = field_values.reshape(ny, nx)

            return field


def _moving_average_field(modelgrid, mean_flat, variance_flat,
                           aa_values, aniso_values, bearing_values,
                           sd_is_log_space, iids, avg_flag=3, power_value=0,
                          active=None):
    """
    Generate field using moving average (convolution) method - matches FIELDGEN2D_SVA approach.

    Parameters
    ----------
    modelgrid : flopy modelgrid
    mean_flat : np.ndarray
        Flattened mean values in native space, shape (ny*nx,)
    variance_flat : np.ndarray
        Flattened variance values (log-space if sd_is_log_space=True), shape (ny*nx,)
    aa_values : np.ndarray
        Flattened correlation length values, shape (ny*nx,)
    aniso_values : np.ndarray
        Flattened anisotropy values, shape (ny*nx,)
    bearing_values : np.ndarray
        Flattened bearing values in degrees, shape (ny*nx,)
    sd_is_log_space : bool
        True if variance_flat is in log space
    iids : np.ndarray
        Independent identically distributed random values, shape (ny*nx,)
    avg_flag : int, default 3
        Averaging function type: 1=spherical, 2=exponential, 3=gaussian, 4=power
    power_value : float, default 0
        Power value if avg_flag=4

    Returns
    -------
    np.ndarray
        Generated field values in native space, shape (ny*nx,)
    """
    from scipy.spatial import cKDTree

    # Map avg_flag to function name for display
    avg_names = {1: 'spherical', 2: 'exponential', 3: 'gaussian', 4: 'power'}
    avg_name = avg_names.get(avg_flag, 'unknown')
    print(f"    Using moving average convolution with {avg_name} function...")

    grid_info = _extract_grid_info(modelgrid)
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']
    ny, nx = xcentergrid.shape

    xcentergrid_flat = xcentergrid.flatten()
    ycentergrid_flat = ycentergrid.flatten()
    n_points = len(xcentergrid_flat)

    # Build KD-tree for neighbor search
    coords = np.column_stack([xcentergrid_flat, ycentergrid_flat])
    tree = cKDTree(coords)

    # Determine cutoff threshold and search radius based on averaging function
    cutoff_threshold = 1e-5

    if avg_flag == 1 or avg_flag == 4:  # spherical or power
        # Compact support - cutoff at h = 1.0
        h_cutoff = 1.0
    elif avg_flag == 2:  # exponential
        # exp(-h) = 1e-5 => h = -ln(1e-5) ≈ 11.51
        h_cutoff = -np.log(cutoff_threshold)
    elif avg_flag == 3:  # gaussian
        # exp(-h²) = 1e-5 => h = sqrt(-ln(1e-5)) ≈ 3.39
        h_cutoff = np.sqrt(-np.log(cutoff_threshold))
    else:
        raise ValueError(f"Unknown avg_flag: {avg_flag}")

    # Initialize output
    Z_field = np.zeros(n_points)

    # Ensure we have active mask
    if active is None:
        active = np.ones(n_points, dtype=bool)
    else:
        active = active.astype(bool)

    # For each target point, apply moving average
    for i in range(n_points):
        # Skip inactive cells
        if not active[i]:
            Z_field[i] = 0.0
            continue
        if i % 2000 == 0:
            print(f"      Progress: {i}/{n_points} points")

        # Get local parameters for this target point
        x0 = xcentergrid_flat[i]
        y0 = ycentergrid_flat[i]
        local_aa = aa_values[i]
        local_aniso = aniso_values[i]
        local_bearing = np.radians(bearing_values[i])
        local_var = variance_flat[i]

        # Search radius in actual distance units
        search_radius = h_cutoff * local_aa
        # Find neighbors
        neighbor_indices = tree.query_ball_point([x0, y0], search_radius)

        if len(neighbor_indices) == 0:
            neighbor_indices = [i]

        # CRITICAL: Filter to only active neighbors
        neighbor_indices = [idx for idx in neighbor_indices if active[idx]]

        # MAX_NEIGHBORS = 2  # Reasonable limit
        #
        # if len(neighbor_indices) > MAX_NEIGHBORS:
        #     # Keep only the closest MAX_NEIGHBORS points
        #     neighbor_coords = coords[neighbor_indices]
        #     distances = np.sqrt(np.sum((neighbor_coords - [x0, y0]) ** 2, axis=1))
        #     closest_indices = np.argsort(distances)[:MAX_NEIGHBORS]
        #     neighbor_indices = [neighbor_indices[i] for i in closest_indices]

        if len(neighbor_indices) == 0:
            # No active neighbors - use unconditional simulation
            Z_field[i] = np.sqrt(local_var) * iids[i]
            continue

        # Compute anisotropic distances and weights
        weights = np.zeros(len(neighbor_indices))
        for j, idx in enumerate(neighbor_indices):
            dx = xcentergrid_flat[idx] - x0
            dy = ycentergrid_flat[idx] - y0

            # Rotate to principal axes
            cos_bear = np.cos(local_bearing)
            sin_bear = np.sin(local_bearing)
            dx_rot = dx * sin_bear + dy * cos_bear
            dy_rot = -dx * cos_bear + dy * sin_bear

            # Scale by anisotropy
            minor_aa = local_aa / local_aniso
            h_major = dx_rot / local_aa
            h_minor = dy_rot / minor_aa

            # Anisotropic distance in "range units"
            h = np.sqrt(h_major ** 2 + h_minor ** 2)

            # Apply averaging function based on avg_flag
            if avg_flag == 1:  # spherical
                if h <= 1.0:
                    weights[j] = 1.0 - 1.5 * h + 0.5 * h ** 3
                else:
                    weights[j] = 0.0
            elif avg_flag == 2:  # exponential
                weights[j] = np.exp(-h)
                if weights[j] < cutoff_threshold:
                    weights[j] = 0.0
            elif avg_flag == 3:  # gaussian
                weights[j] = np.exp(-h ** 2)
                if weights[j] < cutoff_threshold:
                    weights[j] = 0.0
            elif avg_flag == 4:  # power
                if h <= 1.0:
                    weights[j] = (1.0 - h) ** power_value
                else:
                    weights[j] = 0.0

        # # Apply weighted sum (NOT normalized)
        # weighted_sum = np.sum(weights * iids[neighbor_indices])
        #
        # # Compute variance scaling factor
        # variance_of_weighted_sum = np.sum(weights ** 2)
        #
        # if variance_of_weighted_sum > 0:
        #     scaling_factor = np.sqrt(local_var / variance_of_weighted_sum)
        #     # if i < 10 or scaling_factor > 10 or scaling_factor < 0.1:
        #         # print(f"Point {i}: n_neighbors={len(neighbor_indices)}, "
        #         #       f"sum(w²)={variance_of_weighted_sum:.6f}, "
        #         #       f"local_var={local_var:.6f}, "
        #         #       f"scaling={scaling_factor:.6f}")
        #     Z_field[i] = weighted_sum * scaling_factor
        # else:
        #     # Fallback: unconditional simulation
        #     Z_field[i] = np.sqrt(local_var) * iids[i]

        # After computing all weights, normalize them
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum  # Now they sum to 1

        # Apply weighted sum
        weighted_sum = np.sum(weights * iids[neighbor_indices])

        # Compute variance scaling factor
        # After normalization, sum(w²) will be much smaller
        variance_of_weighted_sum = np.sum(weights ** 2)

        scaling_factor = np.sqrt(local_var / variance_of_weighted_sum)
        Z_field[i] = weighted_sum * scaling_factor

    print(f"    Convolution complete. Z-field stats: mean={Z_field.mean():.3f}, std={Z_field.std():.3f}")

    # Apply transformation
    if sd_is_log_space:
        field_values = mean_flat * np.power(10.0, Z_field)
    else:
        field_values = mean_flat + Z_field

    return field_values


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