import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, solve
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib as mpl
from pyemu.plot import plot_utils as pu

mpl.use('Tkagg')
import matplotlib.pyplot as plt
import os


# ============================================================================
# UTILITY FUNCTIONS FOR CONSISTENT ARRAY HANDLING
# ============================================================================

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


def generate_fields_from_files(data_dir, conceptual_points_file, grid_file, zone_file=None, iids_file=None,
                               field_name=['field'], layer_mode=True, save_path='.', tensor_interp='idw'):
    """
    Generate spatially correlated fields using tensor-based geostatistics.

    Parameters
    ----------
    data_dir: str
        Path to files
    conceptual_points_file : str
        CSV file containing conceptual points
    grid_file : str
        whitespace-delimited file with grid coordinates (columns: x, y, z)
    zone_file : str or list of str, optional
        zone file(s):
        - Single string: One zone file used for all layers (2D zones)
        - List of strings: One zone file per layer (creates 3D zones)
    iids_file : str or list of str, optional
        noise file(s):
        - Single string: Assumes pattern like "iids_layer{N}.arr"
        - List of strings: Explicit paths to each layer's noise file
        - None: Auto-generates noise and saves as iids_layer{N}.arr files
    field_name : list of str, default ['field']
        Name(s) of field(s) to generate
    layer_mode : bool, default True
        If True, process each layer independently
    save_path : str, default '.'
        Directory path to save output files
    tensor_interp : str, default 'idw'
        Tensor interpolation method

    Returns
    -------
    None
        Saves results to output/ directory
    """
    print(f"=== Tensor-Based Non-Stationary Field Generation from {conceptual_points_file} ===")

    # Load data
    conceptual_points = pd.read_csv(os.path.join(data_dir, conceptual_points_file))
    grid = pd.read_csv(os.path.join(data_dir, grid_file), sep=r'\s+')
    grid_coords = grid[['x', 'y', 'z']].values

    print(f"Loaded {len(conceptual_points)} conceptual points")
    print(f"Loaded {len(grid_coords)} grid points")

    # Infer grid structure
    shape = infer_grid_shape(grid_coords)
    nz, ny, nx = shape
    print(f"Grid shape: {shape} (nz, ny, nx)")

    # Load or generate iids (always layer-specific)
    if iids_file is not None:
        if isinstance(iids_file, list):
            # Explicit list of files
            if len(iids_file) != nz:
                raise ValueError(f"Number of iids files ({len(iids_file)}) must match number of layers ({nz})")
            iids = np.zeros(shape)
            for i, file_path in enumerate(iids_file):
                layer_iids = np.loadtxt(os.path.join(data_dir, file_path))
                iids[i] = layer_iids.reshape(ny, nx)
                print(f"  Loaded iids layer {i + 1} from {os.path.join(data_dir, file_path)}")
        else:
            if '{' in iids_file:
                # Pattern like "iids_layer{}.arr"
                pattern = os.path.join(data_dir, iids_file)
            else:
                # Assume pattern from filename
                name, ext = os.path.splitext(iids_file)
                pattern = os.path.join(data_dir, f"{name}_layer{{}}{ext}")

            iids = np.zeros(shape)
            for i in range(nz):
                file_path = pattern.format(i + 1)
                try:
                    layer_iids = np.loadtxt(file_path)
                    iids[i] = layer_iids.reshape(ny, nx)
                    print(f"  Loaded iids layer {i + 1} from {file_path}")
                except FileNotFoundError:
                    print(f"  Warning: Could not find {file_path}, generating random noise")
                    iids[i] = np.random.randn(ny, nx)
    else:
        print("  Generating correlated noise...")
        np.random.seed(42)
        iids = np.random.randn(*shape)
        for i in range(nz):
            iids_file_path = os.path.join(data_dir, f'iids_layer{i + 1}.arr')
            np.savetxt(iids_file_path, iids[i])
            print(f"  Saved iids layer {i + 1} to {iids_file_path}")

    # Validate grid coordinates
    if not validate_grid_coordinates(grid_coords, shape):
        raise ValueError(f"Grid coordinates ({len(grid_coords)} points) don't match inferred shape {shape}")

    # Load zones with flexible handling
    zones = None
    zones_3d = None

    if zone_file:
        if isinstance(zone_file, list):
            # Multiple zone files - create 3D zones
            if len(zone_file) != nz:
                raise ValueError(f"Number of zone files ({len(zone_file)}) must match number of layers ({nz})")

            zones_list = []
            for i, file in enumerate(zone_file):
                layer_zones = np.loadtxt(os.path.join(data_dir, file)).astype(np.int64)
                zones_list.append(layer_zones)
                print(f"  Loaded zones layer {i + 1} from {os.path.join(data_dir, file)}, shape: {layer_zones.shape}")

            zones_3d = np.array(zones_list)
            print(f"  Created 3D zones array: {zones_3d.shape}")

        else:
            # Single zone file - use for all layers
            zones_raw = np.loadtxt(os.path.join(data_dir, zone_file)).astype(np.int64)
            zones, _, original_zones_shape = standardize_input_arrays(zones_raw, grid_coords, shape)
            print(f"  Loaded single zone file: original shape {original_zones_shape}, standardized to 2D {zones.shape}")

            # Convert to 3D by replication
            if zones is not None:
                zones_3d = np.tile(zones[np.newaxis, :, :], (nz, 1, 1))
                print(f"  Replicated 2D zones to 3D: {zones.shape} -> {zones_3d.shape}")

    # Process each field
    for fn in field_name:
        cols = ['name', 'x', 'y', 'major', 'transverse', 'bearing']
        cols = cols + [_ for _ in conceptual_points.columns if f'_{fn}' in _]

        if layer_mode and 'layer' in conceptual_points.columns:
            # Process by layer
            cols = cols + ['layer']
            field = generate_field_by_layer(conceptual_points[cols], grid_coords, iids,
                                            shape, zones=zones_3d, save_path=save_path,
                                            tensor_interp=tensor_interp)
        else:
            # Process full 3D - pass the 3D zones
            cols = cols + ['z', 'normal', 'dip']
            field = generate_field_3d(conceptual_points[cols], grid_coords, iids,
                                      shape, zones=zones_3d, save_path=save_path,
                                      tensor_interp=tensor_interp)

        # Save and plot results
        for i in conceptual_points.layer.unique():
            save_layer(field[i - 1], layer=i, field_name=fn, save_path=save_path)
            pu.plot_layer(np.where(zones == 0, np.nan, field[i - 1]), layer=i, field_name=fn, save_path=save_path)

        print("\n=== Complete ===")
        print(f"Field shape: {field.shape}")
        print(f"Field statistics: mean={np.mean(field):.3f}, std={np.std(field):.3f}")


def generate_field_by_layer(conceptual_points, grid_coords, iids,
                            shape, zones=None, save_path='.',
                            tensor_interp='idw'):
    """
    Generate field processing each layer independently.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with tensor and statistical parameters
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    iids : np.ndarray
        Independent identically distributed random noise, shape (nz, ny, nx)
    shape : tuple
        Grid shape as (nz, ny, nx)
    zones : np.ndarray, optional
        Zone IDs for zone-based processing, shape (ny, nx)
    save_path : str, default '.'
        Directory path to save output files
    tensor_interp : str, default 'idw'
        Tensor interpolation method: 'idw', 'krig', or 'nearest'

    Returns
    -------
    np.ndarray
        Field values, shape (nz, ny, nx)
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
            if len(zones.shape) == 3:  # 3D zones
                layer_zones = zones[layer_idx]
            else:  # 2D zones (standard case), use for all layers
                layer_zones = zones
            # zones are already guaranteed to be 2D from standardize_input_arrays()

        # Generate field for this layer
        layer_field_2d, geological_tensors = generate_single_layer_zone_based(layer_cp, layer_coords,
                                                          layer_iids, zones=layer_zones, shape=layer_shape,
                                                          save_path=save_path, tensor_interp=tensor_interp)
        pu.visualize_geological_tensors(geological_tensors, layer_coords, layer_shape, zones=layer_zones,
                                     conceptual_points=layer_cp, subsample=4,
                                     save_path=save_path, title_suf=f'layer {layer_idx+1}')

        # layer_field_2d is already 2D
        field_3d[layer_idx] = layer_field_2d

    return field_3d  # Return as 3D array, NOT flattened


def generate_field_3d(conceptual_points, grid_coords, iids, shape, zones=None,
                      save_path='.', tensor_interp='idw'):
    """
    Generate field using full 3D tensor processing with zone-based interpolation.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with 3D tensor and statistical parameters
        Must include columns: x, y, z, major, transverse, normal, bearing, dip, mean_{field}, sd_{field}
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    iids : np.ndarray
        Independent identically distributed random noise, shape (nz, ny, nx)
    shape : tuple
        Grid shape as (nz, ny, nx)
    zones : np.ndarray, optional
        Zone IDs for zone-based processing. Shape (ny, nx) for 2D zones applied to all layers,
        or (nz, ny, nx) for 3D zones
    save_path : str, default '.'
        Directory path to save output files
    tensor_interp : str, default 'idw'
        Tensor interpolation method: 'idw', 'krig', or 'nearest'

    Returns
    -------
    np.ndarray
        Field values, shape (nz, ny, nx)
    """
    print("Generating 3D field with full tensor processing...")
    nz, ny, nx = shape

    # Extract conceptual point data
    cp_coords = conceptual_points[['x', 'y', 'z']].values

    # Get field statistics
    field_cols = [col for col in conceptual_points.columns if col.startswith('mean_') or col.startswith('sd_')]
    if field_cols:
        mean_col = [col for col in field_cols if col.startswith('mean_')][0]
        sd_col = [col for col in field_cols if col.startswith('sd_')][0]
        cp_means = conceptual_points[mean_col].values
        cp_sds = conceptual_points[sd_col].values
    else:
        # Default values if not specified
        cp_means = np.ones(len(cp_coords))
        cp_sds = np.ones(len(cp_coords)) * 0.5

    # Handle zones for 3D
    zones_3d = None
    if zones is not None:
        if len(zones.shape) == 2:
            # 2D zones - replicate for each layer
            zones_3d = np.tile(zones[np.newaxis, :, :], (nz, 1, 1))
            print(f"  Using 2D zones replicated to 3D: {zones.shape} -> {zones_3d.shape}")
        else:
            # Already 3D zones
            zones_3d = zones
            print(f"  Using 3D zones: {zones_3d.shape}")

    # Step 1: Create geological tensors from conceptual points
    print("  Creating 3D geological correlation tensors...")
    bearing = np.radians(conceptual_points['bearing'].values)
    dip = np.radians(conceptual_points['dip'].values)
    major = conceptual_points['major'].values
    transverse = conceptual_points['transverse'].values
    normal = conceptual_points['normal'].values

    cp_tensors_3d = create_3d_tensors(bearing, dip, major, transverse, normal)

    # Debug: Test tensor creation with first few points
    print("  Tensor creation verification:")
    for i in range(min(3, len(cp_tensors_3d))):
        eigenvals = np.linalg.eigvals(cp_tensors_3d[i])
        print(f"    Point {i}: Eigenvalues={eigenvals}, All positive={np.all(eigenvals > 0)}")

    # Step 2: Interpolate tensors using zone-based approach
    geological_tensors_3d = interpolate_tensors(cp_coords, cp_tensors_3d, grid_coords,
                                                zones=zones_3d, method=tensor_interp)

    # Step 3: Interpolate means using zone-aware kriging
    print("  Interpolating mean field with 3D geological structure...")
    mean_field_1d = tensor_aware_kriging(
        cp_coords, cp_means, grid_coords, geological_tensors_3d,
        variogram_model='exponential', sill=1.0, nugget=0.01,
        background_value=np.mean(cp_means), max_search_radius=1e20,
        min_points=3, transform='log', min_value=1e-8, max_neighbors=8,
        zones=zones_3d
    )
    mean_field_3d = mean_field_1d.reshape(shape)

    # Apply boundary smoothing if zones are present
    if zones_3d is not None:
        print("  Smoothing values at geological boundaries...")
        mean_field_3d = create_boundary_modified_scalar(
            mean_field_3d, zones_3d,
            transition_cells=3, mode='smooth'
        )

    # Step 4: Interpolate standard deviation using geological tensors
    print("  Interpolating standard deviation field...")
    sd_field_1d = tensor_aware_kriging(
        cp_coords, cp_sds, grid_coords, geological_tensors_3d,
        variogram_model='exponential', sill=0.5, nugget=0.01,
        background_value=np.mean(cp_sds), max_search_radius=1e20,
        min_points=3, transform=None, min_value=1e-8, max_neighbors=8,
        zones=zones_3d
    )
    sd_field_3d = sd_field_1d.reshape(shape)

    # Apply boundary enhancement if zones are present
    if zones_3d is not None:
        print("  Enhancing variance at geological boundaries...")
        sd_field_3d = create_boundary_modified_scalar(
            sd_field_3d, zones_3d,
            peak_increase=1, transition_cells=3, mode='enhance'
        )

    # Step 5: Generate correlated noise using 3D tensors
    print("  Generating 3D correlated noise...")
    # Work with 1D for noise generation, then reshape
    iids_flat = iids.flatten()
    correlated_noise_1d = generate_correlated_noise_3d(
        grid_coords, geological_tensors_3d, iids_flat,
        n_neighbors=50, anisotropy_strength=1.0
    )
    correlated_noise_3d = correlated_noise_1d.reshape(shape)

    # Step 6: Combine into log-normal field
    print("  Combining fields using log-normal formulation...")
    print(f"  Mean field shape: {mean_field_3d.shape}")
    print(f"  SD field shape: {sd_field_3d.shape}")
    print(f"  Noise shape: {correlated_noise_3d.shape}")

    # Check if correlated noise has proper statistics (should be ~N(0,1))
    noise_mean = np.mean(correlated_noise_3d)
    noise_std = np.std(correlated_noise_3d)
    print(f"  Correlated noise stats: mean={noise_mean:.6f}, std={noise_std:.6f}")

    # Log-normal field generation: field = 10^(log10(mean) + noise * sd)
    log_mean_field = np.log10(np.maximum(mean_field_3d, 1e-8))  # Avoid log(0)
    field_3d = 10 ** (log_mean_field + correlated_noise_3d * sd_field_3d)

    print(f"  Log-mean field range: [{np.min(log_mean_field):.3f}, {np.max(log_mean_field):.3f}]")
    print(f"  Generated field range: [{np.min(field_3d):.3e}, {np.max(field_3d):.3e}]")

    print(f"  Generated 3D field with shape {field_3d.shape}")
    print(f"  Field stats: mean={np.mean(field_3d):.3f}, std={np.std(field_3d):.3f}")

    # Export to ParaView
    export_3d_results_to_paraview(
        field_3d=field_3d,
        tensors_3d=geological_tensors_3d,
        grid_coords=grid_coords,
        shape=shape,
        zones=zones,
        conceptual_points=conceptual_points,
        save_path=save_path,
        field_name='conductivity'
    )

    return field_3d



def generate_single_layer_zone_based(conceptual_points, grid_coords_2d, iids,
                                     zones=None, shape=None, save_path='.',
                                     tensor_interp='idw'):
    """
    Modified generate_single_layer with zone-based tensor interpolation.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with tensor and statistical parameters
    grid_coords_2d : np.ndarray
        Grid coordinates for single layer, shape (n_points, 2) or (n_points, 3)
    iids : np.ndarray
        Independent identically distributed random noise for this layer
    zones : np.ndarray, optional
        Zone IDs, shape (ny, nx)
    shape : tuple, optional
        Grid shape (ny, nx). If None, inferred from grid_coords_2d
    save_path : str, default '.'
        Directory path to save output files
    tensor_interp : str, default 'idw'
        Tensor interpolation method: 'idw', 'krig', or 'nearest'

    Returns
    -------
    tuple
        (field_2d, geological_tensors) where:
        - field_2d: Generated field, shape (ny, nx)
        - geological_tensors: Interpolated tensors, shape (n_points, 2, 2)
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
    minor = conceptual_points['transverse'].values

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

    # Step 2: Interpolate tensors, various methods
    geological_tensors = interpolate_tensors(cp_coords, cp_tensors, grid_coords_2d,
                                             zones=zones, method=tensor_interp)

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

    log_mean_field = np.log10(np.maximum(interp_means_2d, 1e-8))  # Avoid log(0)
    field_2d = 10 ** (log_mean_field + correlated_noise_2d * interp_sd_2d)

    return field_2d, geological_tensors


def interpolate_tensors(cp_coords, cp_tensors, grid_coords, zones=None, method='idw'):
    """
    Single tensor interpolation function - replaces all the variants.

    Parameters
    ----------
    zones : np.ndarray, optional
        If provided, interpolation is done within zones
    method : str, default 'idw'
        'nearest', 'idw', or 'krig'
    """
    if zones is not None:
        return _interpolate_by_zone(cp_coords, cp_tensors, grid_coords, zones, method)
    else:
        return _interpolate_global(cp_coords, cp_tensors, grid_coords, method)


def _interpolate_by_zone(cp_coords, cp_tensors, grid_coords, zones, method):
    """Handle zone-based interpolation."""
    n_grid = len(grid_coords)
    tensor_size = cp_tensors.shape[-1]
    result = np.zeros((n_grid, tensor_size, tensor_size))

    # Get zone assignment function based on zones shape
    if len(zones.shape) == 2:
        ny, nx = zones.shape
        get_zone = lambda i: zones[divmod(i, nx)]
    else:
        nz, ny, nx = zones.shape
        get_zone = lambda i: zones[i // (ny * nx), (i % (ny * nx)) // nx, i % nx]

    # Assign conceptual points to zones
    cp_zones = assign_conceptual_points_to_zones(cp_coords, grid_coords, zones)

    # Process each zone
    for zone_id in np.unique(zones):
        zone_cp_mask = cp_zones == zone_id
        zone_cp_coords = cp_coords[zone_cp_mask]
        zone_cp_tensors = cp_tensors[zone_cp_mask]

        zone_indices = [i for i in range(n_grid) if get_zone(i) == zone_id]

        if len(zone_cp_coords) > 0 and len(zone_indices) > 0:
            zone_grid_coords = grid_coords[zone_indices]
            zone_tensors = _interpolate_global(zone_cp_coords, zone_cp_tensors,
                                               zone_grid_coords, method)
            result[zone_indices] = zone_tensors
        elif len(zone_indices) > 0:
            # Default tensor for zones without conceptual points
            default_tensor = np.eye(tensor_size) * 1000 ** 2
            result[zone_indices] = default_tensor

    return result


def _interpolate_global(cp_coords, cp_tensors, grid_coords, method):
    """Handle global interpolation without zones."""
    if method == 'nearest':
        return _nearest_method(cp_coords, cp_tensors, grid_coords)
    elif method == 'idw':
        return _idw_method(cp_coords, cp_tensors, grid_coords)
    elif method == 'krig':
        return _kriging_method(cp_coords, cp_tensors, grid_coords)
    else:
        raise ValueError(f"Unknown method: {method}")


def _nearest_method(cp_coords, cp_tensors, grid_coords):
    """Nearest neighbor interpolation."""
    from scipy.spatial.distance import cdist
    distances = cdist(grid_coords, cp_coords)
    nearest_indices = np.argmin(distances, axis=1)
    return cp_tensors[nearest_indices]


def _idw_method(cp_coords, cp_tensors, grid_coords):
    """IDW using log-Euclidean approach."""
    n_grid = len(grid_coords)
    tensor_size = cp_tensors.shape[-1]

    # Convert to log space
    log_tensors = np.array([logm(_ensure_positive_definite(tensor))
                            for tensor in cp_tensors])

    # Interpolate each component
    result_log = np.zeros((n_grid, tensor_size, tensor_size))

    for i in range(tensor_size):
        for j in range(i, tensor_size):
            values = log_tensors[:, i, j].real
            interp_values = _idw_interpolation(cp_coords, values, grid_coords)
            result_log[:, i, j] = interp_values
            if i != j:
                result_log[:, j, i] = interp_values

    # Convert back from log space
    result = np.array([_ensure_positive_definite(expm(log_tensor))
                       for log_tensor in result_log])
    return result


def _kriging_method(cp_coords, cp_tensors, grid_coords):
    """Kriging using log-Euclidean approach."""
    n_grid = len(grid_coords)
    tensor_size = cp_tensors.shape[-1]

    # Convert to log space
    log_tensors = np.array([logm(_ensure_positive_definite(tensor))
                            for tensor in cp_tensors])

    # Interpolate each component
    result_log = np.zeros((n_grid, tensor_size, tensor_size))

    for i in range(tensor_size):
        for j in range(i, tensor_size):
            values = log_tensors[:, i, j].real
            interp_values = ordinary_kriging(cp_coords, values, grid_coords,
                                             variogram_model='exponential', range_param=10000,
                                             sill=1.0, nugget=0.1, background_value=0.0,
                                             max_search_radius=1e20, min_points=1,
                                             transform=None, min_value=1e-8, max_neighbors=5)
            result_log[:, i, j] = interp_values
            if i != j:
                result_log[:, j, i] = interp_values

    # Convert back from log space
    result = np.array([_ensure_positive_definite(expm(log_tensor))
                       for log_tensor in result_log])
    return result

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


def _idw_interpolation(cp_coords, cp_values, grid_coords, power=1.0, radius=1000):
    """Simple IDW interpolation for scalar values."""
    n_grid = len(grid_coords)
    interp_values = np.zeros(n_grid)

    for i in range(n_grid):
        distances = np.linalg.norm(cp_coords - grid_coords[i], axis=1)
        weights = np.exp(-distances / radius)  # Exponential decay weights

        if np.sum(weights) > 1e-10:
            interp_values[i] = np.sum(weights * cp_values) / np.sum(weights)
        else:
            interp_values[i] = np.mean(cp_values)  # Fallback

    return interp_values



def _ensure_positive_definite(tensor, min_eigenval=1e-8):
    """Ensure tensor is positive definite."""
    eigenvals, eigenvecs = np.linalg.eigh(tensor)
    eigenvals_reg = np.maximum(eigenvals, min_eigenval)
    return eigenvecs @ np.diag(eigenvals_reg) @ eigenvecs.T

def create_boundary_modified_scalar(base_field, zones,
                                    peak_increase=0.3, transition_cells=5, mode="enhance"):
    """
    Modify scalar field values near geological zone boundaries with either enhancement or smoothing.
    Supports both 2D and 3D fields/zones.

    Parameters
    ----------
    base_field : np.ndarray
        Base scalar values, shape (ny, nx) for 2D or (nz, ny, nx) for 3D
    zones : np.ndarray
        Zone IDs, shape (ny, nx) for 2D or (nz, ny, nx) for 3D
    peak_increase : float
        Max enhancement or smoothing strength
    transition_cells : int
        Width of transition region
    mode : str
        'enhance' or 'smooth'

    Returns
    -------
    np.ndarray
        Modified scalar field, same shape as input
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if mode not in ("enhance", "smooth"):
        raise ValueError("mode must be 'enhance' or 'smooth'")

    # Determine if we're working with 2D or 3D
    is_3d = len(base_field.shape) == 3

    if is_3d:
        # 3D case
        nz, ny, nx = base_field.shape
        if zones.shape != base_field.shape:
            raise ValueError(f"Zones shape {zones.shape} must match field shape {base_field.shape}")

        # Process each layer separately for better boundary detection
        modified = base_field.copy()
        total_modified_points = 0

        for layer in range(nz):
            layer_field = base_field[layer]
            layer_zones = zones[layer]

            # Detect boundaries in this layer
            boundary_mask, _ = detect_zone_boundaries(layer_zones)
            distance = distance_transform_edt(~boundary_mask)
            transition_mask = distance <= transition_cells

            if mode == "enhance":
                # Linear enhancement near boundaries
                factor = 1 - distance[transition_mask] / transition_cells
                enhancement = peak_increase * factor
                modified[layer][transition_mask] += enhancement

            elif mode == "smooth":
                # Smooth field with Gaussian filter as the target blending value
                smoothed_field = gaussian_filter(layer_field, sigma=transition_cells / 2)

                # Blend with original based on distance to boundary
                weight = 1 - distance[transition_mask] / transition_cells
                modified[layer][transition_mask] = (
                        weight * smoothed_field[transition_mask] +
                        (1 - weight) * layer_field[transition_mask]
                )

            total_modified_points += np.count_nonzero(transition_mask)

        print(
            f"    {'Enhanced' if mode == 'enhance' else 'Smoothed'} {total_modified_points} points near boundaries across {nz} layers")

    else:
        # 2D case - original logic
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
            # Smooth field with Gaussian filter as the target blending value
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


def create_boundary_enhanced_tensors(geological_tensors, zones,
                                     boundary_anisotropy=10.0, boundary_major_scale=2.0,
                                     transition_cells=3):
    """
    Enhance geological tensors at zone boundaries with increased anisotropy.
    Uses distance-based transition from geological to boundary behavior.

    Note: Alternative boundary enhancement method. Implemented but found to be
    suboptimal compared to create_boundary_modified_scalar(). Kept for reference.

    Parameters
    ----------
    geological_tensors : np.ndarray
        Base geological tensors, shape (n_points, 2, 2)
    zones : np.ndarray
        Zone IDs, shape (ny, nx)
    boundary_anisotropy : float, default 10.0
        Anisotropy ratio at boundaries (major/minor)
    boundary_major_scale : float, default 2.0
        Scale factor for major axis at boundaries
    transition_cells : int, default 3
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
        Zone IDs, shape (ny, nx)

    Returns
    -------
    tuple
        (boundary_mask, boundary_directions) where:
        - boundary_mask: Boolean array indicating boundary points, shape (ny, nx)
        - boundary_directions: Direction vectors at boundaries, shape (ny, nx, 2)
    """
    ny, nx = zones.shape
    boundary_mask = np.zeros((ny, nx)).astype(bool)
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
    cp_coords : np.ndarray
        Conceptual point coordinates, shape (n_cp, 2) for 2D or (n_cp, 3) for 3D
    cp_values : np.ndarray
        Values at conceptual points, shape (n_cp,)
    grid_coords : np.ndarray
        Grid coordinates, shape (n_grid, 2) for 2D or (n_grid, 3) for 3D
    interp_tensors : np.ndarray
        Interpolated tensors at grid points, shape (n_grid, 2, 2) for 2D or (n_grid, 3, 3) for 3D
    variogram_model : str, default 'exponential'
        Variogram model: 'exponential', 'gaussian', or 'spherical'
    sill : float, default 1.0
        Variogram sill parameter
    nugget : float, default 0.01
        Variogram nugget parameter
    background_value : float, default 0.0
        Default value for points with insufficient neighbors
    max_search_radius : float, default 1e20
        Maximum search radius for neighbors
    min_points : int, default 1
        Minimum number of points required for interpolation
    max_neighbors : int, default 4
        Maximum number of neighbors to use
    transform : str, optional
        Data transformation: 'log' for log-transform, None for no transform
    min_value : float, default 1e-8
        Minimum value for log transform
    zones : np.ndarray, optional
        Zone IDs for zone-aware kriging. Shape (ny, nx) for 2D or (nz, ny, nx) for 3D

    Returns
    -------
    np.ndarray
        Interpolated values. Shape (ny, nx) for 2D grids or (nz, ny, nx) for 3D grids
    """
    n_grid = len(grid_coords)
    n_dims = grid_coords.shape[1]

    # Get shape from zones if provided, or infer from grid
    if zones is not None:
        shape = zones.shape
        if n_dims == 2:
            ny, nx = shape
        else:  # 3D zones would be (nz, ny, nx)
            nz, ny, nx = shape
    else:
        # Use existing function to infer shape
        shape = infer_grid_shape(grid_coords)
        if n_dims == 2:
            ny, nx = shape[1], shape[2]  # shape is (1, ny, nx)
        else:
            nz, ny, nx = shape

    # Define index conversion functions based on dimensions
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

    interp_values_1d = np.full(n_grid, background_value)

    # Calculate zone for each conceptual point (once, outside the main loop)
    if zones is not None:
        cp_zones = []
        for coord in cp_coords:
            # Find closest grid point to get zone
            distances = np.sum((grid_coords - coord) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            if n_dims == 3:
                layer, row, col = get_indices(closest_idx)
                cp_zones.append(zones[layer, row, col])
            else:
                row, col = get_indices(closest_idx)
                cp_zones.append(zones[row, col])
        cp_zones = np.array(cp_zones)

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

    return interp_values_1d.reshape(shape)



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


def generate_correlated_noise_3d(grid_coords, tensors, iids,
                                 n_neighbors=50, anisotropy_strength=1.0):
    """Generate spatially correlated noise respecting 3D tensor anisotropy."""
    n_points = len(grid_coords)
    correlated_noise = np.zeros(n_points)

    for i in range(n_points):
        evals, evecs = np.linalg.eigh(tensors[i])

        # Sort eigenvalues/vectors by magnitude (largest to smallest)
        sort_idx = np.argsort(evals)[::-1]
        evals = evals[sort_idx]
        evecs = evecs[:, sort_idx]

        # Extract correlation lengths (sqrt of eigenvalues)
        major_corr_length = np.sqrt(evals[0])  # Largest eigenvalue
        intermediate_corr_length = np.sqrt(evals[1])  # Middle eigenvalue
        minor_corr_length = np.sqrt(evals[2])  # Smallest eigenvalue

        # Apply anisotropy strength scaling
        scale_major = major_corr_length * anisotropy_strength
        scale_intermediate = intermediate_corr_length  # Keep intermediate neutral
        scale_minor = minor_corr_length / anisotropy_strength

        # Create 3D transformation matrix
        transform = evecs @ np.diag([scale_major, scale_intermediate, scale_minor])

        # Transform all grid points relative to current point
        all_dx = grid_coords - grid_coords[i]
        try:
            inv_transform = np.linalg.inv(transform)
            dx_transformed = all_dx @ inv_transform.T
            aniso_distances = np.linalg.norm(dx_transformed, axis=1)
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if transform is singular
            aniso_distances = np.linalg.norm(all_dx, axis=1)

        neighbor_indices = np.argsort(aniso_distances)[:min(n_neighbors, n_points)]

        weighted_sum = 0.0
        sum_weights = 0.0

        for j in neighbor_indices:
            # Use anisotropic distance for weight calculation
            weight = np.exp(-(aniso_distances[j]) ** 2)
            weighted_sum += weight * iids.flatten()[j]
            sum_weights += weight

        if sum_weights > 1e-10:
            correlated_noise[i] = weighted_sum / sum_weights
        else:
            correlated_noise[i] = iids[i]

    return correlated_noise



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


def create_3d_tensors(bearing, dip, major, transverse, normal):
    """
    Create 3x3 anisotropy tensors from geostatistical parameters using geological convention.

    [Same docstring as original]
    """
    bearing = np.atleast_1d(bearing)
    dip = np.atleast_1d(dip)
    major = np.atleast_1d(major)
    transverse = np.atleast_1d(transverse)
    normal = np.atleast_1d(normal)

    n = len(bearing)
    tensors = np.zeros((n, 3, 3))

    for i in range(n):
        # Ensure minimum values to prevent numerical issues
        min_axis = 1e-6
        major_val = max(major[i], min_axis)
        transverse_val = max(transverse[i], min_axis)
        normal_val = max(normal[i], min_axis)

        # Step 1: Create diagonal tensor in principal axes coordinate system
        # Order: [transverse, major, normal]
        S = np.diag([transverse_val ** 2, major_val ** 2, normal_val ** 2])

        # Step 2: Rotation around vertical (z) axis by bearing
        cos_bearing = np.cos(bearing[i])
        sin_bearing = np.sin(bearing[i])

        R_bearing = np.array([
            [cos_bearing, sin_bearing, 0],
            [-sin_bearing, cos_bearing, 0],
            [0, 0, 1]
        ])

        # Step 3: Rotation around transverse axis (x-axis) by dip
        cos_dip = np.cos(dip[i])
        sin_dip = np.sin(dip[i])

        R_dip = np.array([
            [1, 0, 0],
            [0, cos_dip, sin_dip],
            [0, -sin_dip, cos_dip]
        ])

        # Step 4: Combine rotations - apply dip first, then bearing
        R_total = R_bearing @ R_dip

        # Step 5: Transform tensor
        tensors[i] = R_total @ S @ R_total.T

        # Ensure symmetry (handle numerical errors)
        tensors[i] = 0.5 * (tensors[i] + tensors[i].T)

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
    Assign conceptual points to zones based on nearest grid point.
    Supports both 2D and 3D zones.

    Parameters
    ----------
    cp_coords : np.ndarray
        Conceptual point coordinates, shape (n_cp, 2) or (n_cp, 3)
    grid_coords : np.ndarray
        Grid coordinates, shape (n_grid, 2) or (n_grid, 3)
    zones : np.ndarray
        Zone array, shape (ny, nx) for 2D or (nz, ny, nx) for 3D

    Returns
    -------
    np.ndarray
        Zone IDs for each conceptual point, shape (n_cp,)
    """
    n_cp = len(cp_coords)
    cp_zones = np.zeros(n_cp).astype(np.int64)

    # Determine zone dimensions
    if len(zones.shape) == 2:
        # 2D zones
        ny, nx = zones.shape

        def get_zone_from_index(index):
            row, col = divmod(index, nx)
            return zones[row, col]

    else:
        # 3D zones
        nz, ny, nx = zones.shape

        def get_zone_from_index(index):
            layer = index // (ny * nx)
            remainder = index % (ny * nx)
            row = remainder // nx
            col = remainder % nx
            return zones[layer, row, col]

    # For each conceptual point, find nearest grid point and get its zone
    for i, cp_coord in enumerate(cp_coords):
        # Find closest grid point
        distances = np.sum((grid_coords - cp_coord) ** 2, axis=1)
        closest_idx = np.argmin(distances)

        # Get zone ID from closest grid point
        cp_zones[i] = get_zone_from_index(closest_idx)

    return cp_zones


def infer_grid_shape(grid_coords):
    """Infer grid dimensions from coordinate array."""
    if grid_coords.shape[1] == 2:
        x_unique = np.unique(grid_coords[:, 0])
        y_unique = np.unique(grid_coords[:, 1])
        ny, nx = len(y_unique), len(x_unique)
        return (1, ny, nx)
    elif grid_coords.shape[1] == 3:
        x_unique = np.unique(grid_coords[:, 0])
        y_unique = np.unique(grid_coords[:, 1])
        z_unique = np.unique(grid_coords[:, 2])
        nx = len(x_unique)
        ny = len(y_unique)
        nz = len(grid_coords) // (nx * ny)
        return(nz, ny, nx)


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

    filename = os.path.join(save_path, f'{field_name}_layer_{layer + 1:02d}.arr')
    np.savetxt(filename, field, fmt='%.6f')
    print(f"Saved layer {layer} to '{filename}'")


def add_indices_to_conceptual_points(conceptual_points, grid_coords, shape):
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
        Conceptual points with added 'i' and 'j' columns
    """
    ny, nx = shape
    cp_coords = conceptual_points[['x', 'y']].values

    i_list = []
    j_list = []

    for coord in cp_coords:
        # Find closest grid point
        distances = np.sum((grid_coords[:, :2] - coord) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        row, col = divmod(closest_idx, nx)

        i_list.append(row)
        j_list.append(col)

    # Add to dataframe
    conceptual_points_copy = conceptual_points.copy()
    conceptual_points_copy['i'] = i_list
    conceptual_points_copy['j'] = j_list

    return conceptual_points_copy


def debug_zone_assignments(conceptual_points, grid_coords, zones, shape, save_path='.'):
    """
    Complete workflow to debug zone assignments.
    """
    # Add grid indices
    cp_with_indices = add_indices_to_conceptual_points(
        conceptual_points, grid_coords, shape
    )

    # Print summary
    print("Conceptual Point Zone Assignments:")
    print("=" * 50)
    for idx, row in cp_with_indices.iterrows():
        ny, nx = shape
        i, j = int(row['i']), int(row['j'])

        # zones are guaranteed to be 2D from entry point
        zone_id = zones[i, j]

        print(f"CP {idx}: ({row['x']:.1f}, {row['y']:.1f}) -> "
              f"grid[{i},{j}] -> zone {zone_id}")

    # Create plot
    pu.plot_zones_with_conceptual_points(
        zones, cp_with_indices, grid_coords,
        save_path=os.path.join(save_path, f'zone_debug.png'))

    return cp_with_indices


def analyze_cp_tensor_alignment(conceptual_points, geological_tensors, grid_coords, zones):
    """
    Compare conceptual point bearings with actual tensor orientations at those locations.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with x, y, bearing columns
    geological_tensors : np.ndarray
        Interpolated tensor field, shape (n_points, 2, 2)
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 2) or (n_points, 3)
    zones : np.ndarray
        Zone IDs, shape (ny, nx)

    Returns
    -------
    None
        Prints comparison table to console
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
    geological_tensors : np.ndarray
        Tensor field, shape (n_points, 2, 2)
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 2) or (n_points, 3)
    zones : np.ndarray
        Zone IDs, shape (ny, nx)

    Returns
    -------
    None
        Prints statistical analysis to console
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


def export_to_paraview(grid_coords, shape, field_data=None, tensor_data=None,
                       zones=None, conceptual_points=None, save_path='.',
                       filename='field_3d'):
    """
    Export 3D field and tensor data to ParaView-compatible VTK files.

    Parameters
    ----------
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    shape : tuple
        Grid shape (nz, ny, nx)
    field_data : dict, optional
        Dictionary of field arrays to export, e.g. {'permeability': field_array}
        Each array should have shape (nz, ny, nx)
    tensor_data : np.ndarray, optional
        Tensor field, shape (n_points, 3, 3)
    zones : np.ndarray, optional
        Zone array, shape (nz, ny, nx) or (ny, nx)
    conceptual_points : pd.DataFrame, optional
        Conceptual points with x, y, z, bearing, dip columns
    save_path : str, default '.'
        Directory to save VTK files
    filename : str, default 'field_3d'
        Base filename for output files

    Returns
    -------
    None
        Saves .vtr (structured grid) and .vtp (point cloud) files
    """
    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        print("VTK not installed. Install with: pip install vtk")
        return

    nz, ny, nx = shape

    # Create structured grid for field data
    if field_data is not None or tensor_data is not None or zones is not None:
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)

        # Set points
        points = vtk.vtkPoints()
        grid_coords_reshaped = grid_coords.reshape(shape + (3,))

        # VTK expects points in a specific order (k varies fastest, then j, then i)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x, y, z = grid_coords_reshaped[k, j, i]
                    points.InsertNextPoint(x, y, z)

        grid.SetPoints(points)

        # Add field data
        if field_data is not None:
            for field_name, field_array in field_data.items():
                # Reshape and reorder for VTK
                field_flat = field_array.flatten(order='F')  # Fortran order for VTK
                field_vtk = numpy_support.numpy_to_vtk(field_flat)
                field_vtk.SetName(field_name)
                grid.GetPointData().AddArray(field_vtk)
                if grid.GetPointData().GetScalars() is None:
                    grid.GetPointData().SetScalars(field_vtk)

        # Add zones
        if zones is not None:
            if len(zones.shape) == 2:
                # 2D zones - replicate for each layer
                zones_3d = np.tile(zones[np.newaxis, :, :], (nz, 1, 1))
            else:
                zones_3d = zones

            zones_flat = zones_3d.flatten(order='F')
            zones_vtk = numpy_support.numpy_to_vtk(zones_flat)
            zones_vtk.SetName('zones')
            grid.GetPointData().AddArray(zones_vtk)

        # Add tensor data
        if tensor_data is not None:
            # Convert 3x3 tensors to VTK format (symmetric tensors)
            n_points = len(tensor_data)
            tensor_array = np.zeros((n_points, 6))  # VTK uses 6-component symmetric tensors

            for i in range(n_points):
                tensor = tensor_data[i]
                # VTK tensor format: [xx, yy, zz, xy, yz, xz]
                tensor_array[i] = [tensor[0, 0], tensor[1, 1], tensor[2, 2],
                                   tensor[0, 1], tensor[1, 2], tensor[0, 2]]

            # Reorder for VTK grid
            tensor_reordered = np.zeros_like(tensor_array)
            idx = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        idx = k * ny * nx + j * nx + i
                        tensor_reordered[idx] = tensor_array[idx]
                        idx += 1

            tensor_vtk = numpy_support.numpy_to_vtk(tensor_reordered)
            tensor_vtk.SetName('tensors')
            tensor_vtk.SetNumberOfComponents(6)
            grid.GetPointData().SetTensors(tensor_vtk)

        # Write structured grid
        writer = vtk.vtkXMLStructuredGridWriter()
        grid_filename = os.path.join(save_path, f'{filename}_grid.vts')
        writer.SetFileName(grid_filename)
        writer.SetInputData(grid)
        writer.Write()
        print(f"Saved structured grid to {grid_filename}")

    # Create point cloud for conceptual points
    if conceptual_points is not None:
        points = vtk.vtkPoints()
        polydata = vtk.vtkPolyData()

        # Add points
        for idx, row in conceptual_points.iterrows():
            x, y = row['x'], row['y']
            z = row.get('z', 0)  # Default z=0 if not provided
            points.InsertNextPoint(x, y, z)

        polydata.SetPoints(points)

        # Create vertices
        vertices = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        polydata.SetVerts(vertices)

        # Add conceptual point data
        for col in conceptual_points.columns:
            if col in ['x', 'y', 'z']:
                continue

            values = conceptual_points[col].values
            if np.issubdtype(values.dtype, np.number):
                vtk_array = numpy_support.numpy_to_vtk(values)
                vtk_array.SetName(col)
                polydata.GetPointData().AddArray(vtk_array)

        # Add tensor glyphs if we have bearing/dip data
        if 'bearing' in conceptual_points.columns and 'dip' in conceptual_points.columns:
            # Create orientation vectors for visualization
            n_cp = len(conceptual_points)
            orientations = np.zeros((n_cp, 3))

            for i, (_, row) in enumerate(conceptual_points.iterrows()):
                bearing_rad = np.radians(row['bearing'])
                dip_rad = np.radians(row.get('dip', 0))

                # Convert geological convention to cartesian
                # Bearing: CW from north, Dip: down from horizontal
                x_comp = np.sin(bearing_rad) * np.cos(dip_rad)
                y_comp = np.cos(bearing_rad) * np.cos(dip_rad)
                z_comp = -np.sin(dip_rad)  # Negative because dip is down

                orientations[i] = [x_comp, y_comp, z_comp]

            orientation_vtk = numpy_support.numpy_to_vtk(orientations)
            orientation_vtk.SetName('orientations')
            orientation_vtk.SetNumberOfComponents(3)
            polydata.GetPointData().SetVectors(orientation_vtk)

        # Write point cloud
        writer = vtk.vtkXMLPolyDataWriter()
        points_filename = os.path.join(save_path, f'{filename}_conceptual_points.vtp')
        writer.SetFileName(points_filename)
        writer.SetInputData(polydata)
        writer.Write()
        print(f"Saved conceptual points to {points_filename}")

    print("\nParaView visualization tips:")
    print("1. Load both .vts and .vtp files in ParaView")
    print("2. For field data: Use 'Volume' or 'Slice' representation")
    print("3. For tensors: Add 'Tensor Glyph' filter to show ellipsoids")
    print("4. For conceptual points: Use 'Glyph' filter with 'Arrow' glyphs")
    print("5. Use 'Clip' or 'Slice' filters to see inside the 3D volume")


def export_3d_results_to_paraview(field_3d, tensors_3d, grid_coords, shape,
                                  zones=None, conceptual_points=None,
                                  save_path='.', field_name='permeability'):
    """
    Convenience function to export 3D results to ParaView.

    Parameters
    ----------
    field_3d : np.ndarray
        Field values, shape (nz, ny, nx)
    tensors_3d : np.ndarray
        3D tensors, shape (n_points, 3, 3)
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)
    shape : tuple
        Grid shape (nz, ny, nx)
    zones : np.ndarray, optional
        Zone data
    conceptual_points : pd.DataFrame, optional
        Conceptual points
    save_path : str
        Output directory
    field_name : str
        Name for the field data
    """
    field_data = {field_name: field_3d}

    export_to_paraview(
        grid_coords=grid_coords,
        shape=shape,
        field_data=field_data,
        tensor_data=tensors_3d,
        zones=zones,
        conceptual_points=conceptual_points,
        save_path=save_path,
        filename=f'{field_name}_3d'
    )

# ============================================================================
# EXAMPLE USAGE
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
    data_dir = r'..\..\examples\Wairau'
    zone_files = [f'waq_arr.geoclass_layer{z+1}.arr' for z in range(13)]
    save_path = os.path.join(data_dir, 'output_idw_3d')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    generate_fields_from_files(data_dir,
        conceptual_points_file='conceptual_points.csv',
        grid_file='grid.dat',
        zone_file=zone_files,
        field_name=['kh'],
        layer_mode=False,
        save_path=save_path,
        tensor_interp='idw'
    )