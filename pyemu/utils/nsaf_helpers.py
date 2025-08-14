import numpy as np
import pandas as pd
import os
from scipy.ndimage import distance_transform_edt, gaussian_filter

# Import from our Chunk 1 and 2
from nsaf_utils import (
    interpolate_tensors,
    load_conceptual_points,
    create_2d_tensors,
    get_tensor_bearing,
    apply_ppu_hyperpars,
    tensor_aware_kriging,
    create_boundary_modified_scalar,
    detect_zone_boundaries
)


def generate_fields_from_files(tmp_model_ws, model_name, conceptual_points_file,
                               zone_file=None, iids_file=None, field_name=['field'],
                               layer_mode=True, save_path=None, tensor_interp='krig',
                               vartransform='log', n_realizations=1, ml=None, sr=None):
    """
    Generate spatially correlated fields using tensor-based geostatistics.
    CLEAN version that works with flexible property columns.

    Parameters
    ----------
    tmp_model_ws : str
        Path to model workspace containing flopy model files
    model_name : str
        Name of the flopy model (e.g., 'freyberg6')
    conceptual_points_file : str
        CSV file containing conceptual points
    zone_file : str or list of str, optional
        zone file(s)
    iids_file : str or list of str, optional
        noise file(s)
    field_name : list of str, default ['field']
        Name(s) of field(s) to generate
    layer_mode : bool, default True
        If True, process each layer independently
    save_path : str, optional
        Directory path to save output files (default: tmp_model_ws)
    tensor_interp : str, default 'krig'
        Tensor interpolation method
    vartransform : str, default 'log'
        Transformation: 'log' or 'none'
    n_realizations : int, default 1
        Number of stochastic realizations
    ml : flopy.mbase, optional
        A flopy mbase derived type. If None, sr must not be None.
    sr : flopy.utils.reference.SpatialReference, optional
        A spatial reference to locate the model grid. If None, ml must not be None.

    Returns
    -------
    dict
        Results from workflow
    """
    try:
        import flopy
        import pyemu
    except ImportError as e:
        raise ImportError(f"flopy and pyemu are required: {e}")

    print(f"=== Tensor-Based Field Generation ===")
    print(f"Model workspace: {tmp_model_ws}")
    print(f"Model name: {model_name}")

    # Set default save path
    if save_path is None:
        save_path = tmp_model_ws

    # Setup spatial reference following PyEMU pattern
    if ml is None and sr is None:
        print("Loading flopy model...")
        try:
            # Try MF6 first - only load DIS package for efficiency
            sim = flopy.mf6.MFSimulation.load(sim_ws=tmp_model_ws, load_only=['dis'])
            ml = sim.get_model(model_name)
            print(f"Loaded MF6 model: {model_name} (DIS package only)")
        except:
            try:
                # Fall back to MODFLOW-2005 - only load DIS package
                ml = flopy.modflow.Modflow.load(f"{model_name}.nam", model_ws=tmp_model_ws,
                                                load_only=['dis'])
                print(f"Loaded MODFLOW-2005 model: {model_name} (DIS package only)")
            except Exception as e:
                raise ValueError(f"Could not load model {model_name} from {tmp_model_ws}: {e}")

    # Get grid coordinates
    try:
        xcentergrid = ml.modelgrid.xcellcenters
        ycentergrid = ml.modelgrid.ycellcenters
    except AttributeError as e:
        raise Exception(f"Could not get grid centers from spatial reference: {e}")

    ny, nx = xcentergrid.shape
    print(f"Grid dimensions: {ny} x {nx}")
    print(
        f"Grid extent: X=[{xcentergrid.min():.1f}, {xcentergrid.max():.1f}], Y=[{ycentergrid.min():.1f}, {ycentergrid.max():.1f}]")

    # Get cell areas for area parameter
    cell_area = np.outer(ml.dis.delc.array, ml.dis.delr.array)

    # Load conceptual points
    conceptual_points = pd.read_csv(os.path.join(tmp_model_ws, conceptual_points_file))
    print(f"Loaded {len(conceptual_points)} conceptual points")
    print(
        f"Conceptual points extent: X=[{conceptual_points['x'].min():.1f}, {conceptual_points['x'].max():.1f}], Y=[{conceptual_points['y'].min():.1f}, {conceptual_points['y'].max():.1f}]")

    # Add transverse if missing
    if 'transverse' not in conceptual_points.columns and 'anisotropy' in conceptual_points.columns:
        conceptual_points['transverse'] = conceptual_points['major'] / conceptual_points['anisotropy']

    # Load zones
    zones = None
    if zone_file:
        if isinstance(zone_file, str):
            zones = np.loadtxt(os.path.join(tmp_model_ws, zone_file)).astype(np.int64)
            if zones.shape != (ny, nx):
                if zones.size == ny * nx:
                    zones = zones.reshape(ny, nx)
                else:
                    raise ValueError(f"Zone file shape {zones.shape} doesn't match grid {(ny, nx)}")
            print(f"Loaded zones from {zone_file}, shape: {zones.shape}")
        else:
            zones = np.loadtxt(os.path.join(tmp_model_ws, zone_file[0])).astype(np.int64)
            print(f"Loaded zones from {zone_file[0]} (multi-layer support TODO)")

    # Determine layers
    if hasattr(ml, 'dis'):
        if hasattr(ml.dis, 'nlay'):
            total_layers = ml.dis.nlay.array if hasattr(ml.dis.nlay, 'array') else ml.dis.nlay
        elif hasattr(ml.dis, 'botm'):
            total_layers = len(ml.dis.botm.array)
        else:
            total_layers = 1
    else:
        total_layers = 1

    print(f"Model has {total_layers} layers")

    # Handle layers in conceptual points
    if 'layer' in conceptual_points.columns:
        available_cp_layers = sorted(conceptual_points['layer'].unique())
        print(f"Conceptual points available for layers: {available_cp_layers}")
    else:
        available_cp_layers = [1]
        print("No 'layer' column in conceptual points - using layer 1")

    # Generate IIDs for all model layers
    iids_dict = {}
    for layer_num in range(total_layers):
        layer_1based = layer_num + 1
        iids_dict[layer_1based] = _load_or_generate_iids(
            tmp_model_ws, iids_file, layer_num, ny, nx, n_realizations
        )

    # Process each field
    all_results = {}

    for fn in field_name:
        print(f"\n=== Processing field: {fn} ===")

        # SIMPLIFIED: Look for columns with field name pattern
        mean_col = None
        sd_col = None

        # First, try field-specific columns
        if f'mean_{fn}' in conceptual_points.columns:
            mean_col = f'mean_{fn}'
            sd_col = f'sd_{fn}'
        elif f'{fn}_mean' in conceptual_points.columns:
            mean_col = f'{fn}_mean'
            sd_col = f'{fn}_sd'
        else:
            # Look for generic columns
            mean_candidates = [col for col in conceptual_points.columns if 'mean' in col.lower()]
            sd_candidates = [col for col in conceptual_points.columns if 'sd' in col.lower()]

            if mean_candidates and sd_candidates:
                mean_col = mean_candidates[0]  # Use first match
                sd_col = sd_candidates[0]
                print(f"Using generic columns: mean='{mean_col}', sd='{sd_col}'")
            else:
                raise ValueError(f"No mean/sd columns found for field '{fn}'")

        # Verify columns exist
        if mean_col not in conceptual_points.columns:
            raise ValueError(f"Mean column '{mean_col}' not found")
        if sd_col not in conceptual_points.columns:
            raise ValueError(f"SD column '{sd_col}' not found")

        # Handle layer processing
        if layer_mode and 'layer' in conceptual_points.columns:
            # Process each available layer
            for target_layer in available_cp_layers:
                print(f"\n--- Processing layer {target_layer} ---")

                # Filter conceptual points for this layer
                layer_cp = conceptual_points[conceptual_points['layer'] == target_layer].copy()

                if len(layer_cp) == 0:
                    print(f"No conceptual points for layer {target_layer}, skipping")
                    continue

                # Check data ranges
                print(f"Conceptual point data for layer {target_layer}:")
                print(f"  {mean_col}: [{layer_cp[mean_col].min():.2f}, {layer_cp[mean_col].max():.2f}]")
                print(f"  {sd_col}: [{layer_cp[sd_col].min():.2f}, {layer_cp[sd_col].max():.2f}]")

                # Get IIDs for this layer
                layer_iids = iids_dict.get(target_layer, iids_dict[1])

                # apply_ppu_hyperpars
                results = apply_ppu_hyperpars(
                    layer_cp, xcentergrid, ycentergrid,
                    area=cell_area,
                    zones=zones,
                    n_realizations=n_realizations,
                    layer=target_layer - 1,  # Convert to 0-based for internal use
                    vartransform=vartransform,
                    tensor_method=tensor_interp,
                    out_filename=os.path.join(save_path, f"{fn}_layer_{target_layer}"),
                    iids=layer_iids,
                    mean_col=mean_col,  # Pass column names
                    sd_col=sd_col,
                    active=ml.dis.idomain.array[target_layer].flatten()
                )

                # Store results with layer key
                result_key = f"{fn}_layer_{target_layer}"
                all_results[result_key] = results

                # Save results
                if results['fields'] is not None:
                    for i, field in enumerate(results['fields']):
                        save_layer(field, layer=target_layer - 1,
                                   field_name=f"{fn}_real_{i + 1}", save_path=save_path)

                save_layer(results['mean'], layer=target_layer - 1,
                           field_name=f"{fn}_mean", save_path=save_path)
                save_layer(results['sd'], layer=target_layer - 1,
                           field_name=f"{fn}_sd", save_path=save_path)

                print(f"Results for layer {target_layer}:")
                print(f"  Mean field: [{results['mean'].min():.3f}, {results['mean'].max():.3f}]")
                print(f"  SD field: [{results['sd'].min():.3f}, {results['sd'].max():.3f}]")

        else:
            # Single layer mode
            target_layer = 1

            # Get IIDs for this layer
            layer_iids = iids_dict.get(1, None)

            # Use workflow
            results = apply_ppu_hyperpars(
                conceptual_points, xcentergrid, ycentergrid,
                area=cell_area,
                zones=zones,
                n_realizations=n_realizations,
                layer=0,
                vartransform=vartransform,
                tensor_method=tensor_interp,
                out_filename=os.path.join(save_path, f"{fn}"),
                iids=layer_iids,
                mean_col=mean_col,
                sd_col=sd_col
            )

            all_results[fn] = results

    print("\n=== Complete ===")
    return all_results


def _load_or_generate_iids(tmp_model_ws, iids_file, layer_num, ny, nx, n_realizations):
    """
    Load IIDs from file or generate and save new ones for PEST parameterization.
    FIXED version with proper array handling.

    Parameters
    ----------
    tmp_model_ws : str
        Model workspace path
    iids_file : str or list or None
        IID file specification
    layer_num : int
        Layer number (0-based)
    ny, nx : int
        Grid dimensions
    n_realizations : int
        Number of realizations needed

    Returns
    -------
    np.ndarray
        IIDs array with shape (ny*nx, n_realizations)
    """

    # Convert to 1-based for file naming
    layer_1based = layer_num + 1

    # Determine the IID file path
    if iids_file is None:
        # Default pattern
        iid_path = os.path.join(tmp_model_ws, f"iids_layer_{layer_1based}.arr")
    elif isinstance(iids_file, str):
        # Single string - assume pattern or explicit path
        if "{" in iids_file or "layer" in iids_file.lower():
            # Pattern with layer number
            iid_path = os.path.join(tmp_model_ws, iids_file.format(layer_1based))
        else:
            # Explicit path - append layer number
            base_name, ext = os.path.splitext(iids_file)
            iid_path = os.path.join(tmp_model_ws, f"{base_name}_layer_{layer_1based}{ext}")
    elif isinstance(iids_file, list):
        # List of explicit paths
        if layer_num < len(iids_file):
            iid_path = os.path.join(tmp_model_ws, iids_file[layer_num])
        else:
            # Generate default name if not enough files specified
            iid_path = os.path.join(tmp_model_ws, f"iids_layer_{layer_1based}.arr")
    else:
        raise ValueError(f"Invalid iids_file specification: {iids_file}")

    # Try to load existing IIDs
    if os.path.exists(iid_path):
        print(f"  Loading existing IIDs from {iid_path}")
        try:
            # Try to load as text array
            iids_data = np.loadtxt(iid_path)

            # Handle different shapes
            if iids_data.ndim == 1:
                # Single realization
                if len(iids_data) == ny * nx:
                    iids = iids_data.reshape(-1, 1)
                else:
                    raise ValueError(f"IID file has wrong size: {len(iids_data)} vs {ny * nx}")
            elif iids_data.ndim == 2:
                # Multiple realizations
                if iids_data.shape[0] == ny * nx:
                    iids = iids_data
                elif iids_data.shape[1] == ny * nx:
                    iids = iids_data.T
                else:
                    raise ValueError(f"IID file has wrong shape: {iids_data.shape}")
            else:
                raise ValueError(f"IID file has too many dimensions: {iids_data.ndim}")

            # Expand or truncate to match n_realizations
            if iids.shape[1] < n_realizations:
                print(f"  Expanding IIDs from {iids.shape[1]} to {n_realizations} realizations")
                additional_iids = np.random.normal(0, 1, size=(ny * nx, n_realizations - iids.shape[1]))
                iids = np.column_stack([iids, additional_iids])
            elif iids.shape[1] > n_realizations:
                print(f"  Truncating IIDs from {iids.shape[1]} to {n_realizations} realizations")
                iids = iids[:, :n_realizations]

        except Exception as e:
            print(f"  Warning: Could not load IID file {iid_path}: {e}")
            print(f"  Generating new IIDs...")
            iids = None
    else:
        print(f"  IID file {iid_path} not found, generating new IIDs...")
        iids = None

    # Generate new IIDs if loading failed
    if iids is None:
        iids = np.random.normal(0, 1, size=(ny * nx, n_realizations))

        # Save for PEST parameterization
        print(f"  Saving IIDs to {iid_path} for PEST parameterization")
        os.makedirs(os.path.dirname(iid_path), exist_ok=True)
        np.savetxt(iid_path, iids, fmt="%.8f",
                   header=f"IIDs for layer {layer_1based}, shape: {ny * nx} x {n_realizations}")

    return iids


def generate_single_layer(cp_file, xcentergrid, ycentergrid, iids=None,
                          zones=None, save_path='.', tensor_interp='idw',
                          vartransform='log', boundary_smooth=True,
                          boundary_enhance=True):
    """
    Generate single layer using approach with boundary adjustments.
    Replaces the NSAF generate_single_layer_zone_based function.

    Parameters
    ----------
    cp_file : str or pd.DataFrame
        Conceptual points file or DataFrame
    xcentergrid : np.ndarray
        X-coordinates from pyemu SpatialReference, shape (ny, nx)
    ycentergrid : np.ndarray
        Y-coordinates from pyemu SpatialReference, shape (ny, nx)
    iids : np.ndarray, optional
        Pre-generated noise, shape (ny, nx)
    zones : np.ndarray, optional
        Zone IDs, shape (ny, nx)
    save_path : str, default '.'
        Directory path to save output files
    tensor_interp : str, default 'krig'
        Tensor interpolation method
    vartransform : str, default 'log'
        Transformation: 'log' or 'none'
    boundary_smooth : bool, default True
        Apply smoothing to mean at zone boundaries
    boundary_enhance : bool, default True
        Apply variance enhancement at zone boundaries

    Returns
    -------
    tuple
        (field_2d, tensors) where:
        - field_2d: Generated field, shape (ny, nx)
        - tensors: Interpolated tensors, shape (ny*nx, 2, 2)
    """

    # Load conceptual points
    cp_df = load_conceptual_points(cp_file)

    print("=== Single Layer Generation ===")
    print(f"Processing {len(cp_df)} conceptual points")

    ny, nx = xcentergrid.shape

    # Step 1: Create geological tensors from conceptual points
    print("  Creating geological correlation tensors...")
    cp_coords = cp_df[['x', 'y']].values

    # Get mean and sd columns
    mean_cols = [col for col in cp_df.columns if 'mean' in col.lower()]
    sd_cols = [col for col in cp_df.columns if 'sd' in col.lower()]

    if not mean_cols or not sd_cols:
        raise ValueError("Conceptual points must have mean and sd columns")

    cp_means = cp_df[mean_cols[0]].values
    cp_sd = cp_df[sd_cols[0]].values

    # Create tensors
    major = cp_df['major'].values
    minor = cp_df['transverse'].values

    def normalize_bearing(bearing):
        return bearing % 360

    bearings_normalized = [normalize_bearing(b) for b in cp_df['bearing'].values]
    bearing_rad = np.radians(bearings_normalized)
    cp_tensors = create_2d_tensors(bearing_rad, major, minor)

    # Debug tensor creation
    print("  Tensor creation verification:")
    for i in range(min(3, len(cp_tensors))):
        original_bearing = cp_df['bearing'].values[i]
        extracted_bearing = get_tensor_bearing(cp_tensors[i], original_bearing)
        print(f"    Point {i}: Input bearing={original_bearing:.1f}°, Extracted={extracted_bearing:.1f}°")

    # Step 2: Interpolate tensors
    print(f"  Interpolating tensors using {tensor_interp} method...")
    tensors = interpolate_tensors(
        cp_df, xcentergrid, ycentergrid,
        zones=zones, method=tensor_interp, layer=0
    )

    # Step 3: Interpolate means and sd with boundary adjustments
    print("  Interpolating means with geological structure...")
    grid_coords_2d = np.column_stack([xcentergrid.flatten(), ycentergrid.flatten()])

    # Configure kriging
    transform = 'log' if vartransform == 'log' else None
    min_value = 1e-8 if vartransform == 'log' else None

    # Interpolate mean
    interp_means_2d = tensor_aware_kriging(
        cp_coords, cp_means, grid_coords_2d, tensors,
        variogram_model='exponential', sill=1.0, nugget=0.1,
        background_value=np.mean(cp_means), max_search_radius=1e20,
        min_points=3, transform=transform, min_value=min_value,
        max_neighbors=4, zones=zones
    )

    # Apply boundary smoothing
    if boundary_smooth and zones is not None:
        print("  Smoothing values at geological boundaries...")
        interp_means_2d = create_boundary_modified_scalar(
            interp_means_2d, zones, transition_cells=3, mode='smooth'
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

    # Apply boundary enhancement
    if boundary_enhance and zones is not None:
        print("  Enhancing variance at geological boundaries...")
        interp_sd_2d = create_boundary_modified_scalar(
            interp_sd_2d, zones, peak_increase=1.0, transition_cells=3, mode='enhance'
        )

    # Step 4: Generate stochastic field using FIELDGEN2D_SVA approach
    print("  Generating stochastic field using FIELDGEN2D_SVA methodology...")

    # Convert tensors to geostatistical parameters for FIELDGEN2D_SVA
    from nsaf_utils import _tensors_to_geostat_params, _generate_stochastic_fields

    bearing_deg, anisotropy, corrlen_major = _tensors_to_geostat_params(tensors)

    # Prepare grid coordinates
    grid_coords_2d = np.column_stack([xcentergrid.flatten(), ycentergrid.flatten()])

    # Configuration for field generation
    config_dict = {
        'averaging_function': 'exponential',
        'vartype': 2,
        'krigtype': 1
    }

    # Generate one realization
    n_realizations = 1
    fields = _generate_stochastic_fields(
        grid_coords_2d, interp_means_2d, interp_sd_2d,
        bearing_deg, anisotropy, corrlen_major,
        zones, n_realizations, vartransform,
        config_dict, ny, nx
    )

    field_2d = fields[0]  # Extract the single realization

    print("=== Single layer generation complete ===")
    return field_2d, tensors


def save_layer(field_2d, layer, field_name, save_path='.'):
    """
    Save layer to file in PyEMU-compatible format.

    Parameters
    ----------
    field_2d : np.ndarray
        Field values, shape (ny, nx)
    layer : int
        Layer number
    field_name : str
        Field name for filename
    save_path : str
        Directory to save files
    """
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"{field_name}_layer_{layer + 1}.txt")
    np.savetxt(filename, field_2d, fmt="%20.8E")
    print(f"  Saved {field_name} layer {layer + 1} to {filename}")


def validate_grid_coordinates(xcentergrid, ycentergrid):
    """
    Validate PyEMU grid coordinates.

    Parameters
    ----------
    xcentergrid : np.ndarray
        X-coordinates, shape (ny, nx)
    ycentergrid : np.ndarray
        Y-coordinates, shape (ny, nx)

    Returns
    -------
    bool
        True if valid
    """
    if xcentergrid.shape != ycentergrid.shape:
        print(f"ERROR: Grid coordinate shapes don't match: {xcentergrid.shape} vs {ycentergrid.shape}")
        return False

    ny, nx = xcentergrid.shape
    print(f"Grid validation: {ny} x {nx} = {ny * nx} points")

    # Check for regular grid structure
    x_unique = len(np.unique(xcentergrid))
    y_unique = len(np.unique(ycentergrid))

    if x_unique == nx and y_unique == ny:
        print("  ✓ Regular rectangular grid detected")
        return True
    else:
        print(f"  ⚠ Irregular grid: {x_unique} unique x, {y_unique} unique y")
        return True  # Still valid, just not regular


def infer_grid_shape(xcentergrid, ycentergrid):
    """
    Infer grid shape from PyEMU coordinate arrays.

    Parameters
    ----------
    xcentergrid : np.ndarray
        X-coordinates, shape (ny, nx)
    ycentergrid : np.ndarray
        Y-coordinates, shape (ny, nx)

    Returns
    -------
    tuple
        Grid shape (nz, ny, nx) - nz=1 for 2D
    """
    ny, nx = xcentergrid.shape
    return (1, ny, nx)


# Example usage function
def example_field_generation():
    """Example using the restructured helpers."""
    import matplotlib.pyplot as plt

    # Create example PyEMU grid
    nx, ny = 50, 40
    x = np.linspace(0, 5000, nx)
    y = np.linspace(0, 4000, ny)
    xcentergrid, ycentergrid = np.meshgrid(x, y, indexing='xy')
    xcentergrid = xcentergrid.T
    ycentergrid = ycentergrid.T

    # Create example conceptual points
    cp_data = {
        'name': ['cp1', 'cp2', 'cp3', 'cp4'],
        'x': [1000, 3000, 1000, 3000],
        'y': [1000, 1000, 3000, 3000],
        'mean_kh': [1.5, 2.0, 1.8, 1.2],
        'sd_kh': [0.3, 0.4, 0.2, 0.5],
        'major': [2000, 1500, 1800, 2200],
        'anisotropy': [3.0, 2.5, 4.0, 2.0],
        'bearing': [45, 90, 135, 0],
        'layer': [1, 1, 1, 1]
    }
    cp_df = pd.DataFrame(cp_data)

    # Add transverse
    cp_df['transverse'] = cp_df['major'] / cp_df['anisotropy']

    # Create zones
    zones = np.ones((ny, nx), dtype=int)
    zones[:ny // 2, :] = 1
    zones[ny // 2:, :] = 2

    # Generate field using single layer function
    print("=== Testing single layer generation ===")
    field_2d, tensors = generate_single_layer(
        cp_df, xcentergrid, ycentergrid,
        zones=zones, tensor_interp='krig',
        vartransform='log'
    )

    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Field
    im1 = axes[0].imshow(field_2d, origin='lower')
    axes[0].set_title('Generated Field')
    plt.colorbar(im1, ax=axes[0])

    # Zones
    im2 = axes[1].imshow(zones, origin='lower', cmap='tab10')
    axes[1].set_title('Zones')
    plt.colorbar(im2, ax=axes[1])

    # Field statistics
    axes[2].hist(field_2d.flatten(), bins=50, alpha=0.7)
    axes[2].set_title('Field Distribution')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('field_example.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Field statistics: mean={np.mean(field_2d):.3f}, std={np.std(field_2d):.3f}")
    print("Example complete!")

    return field_2d, tensors