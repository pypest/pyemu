"""
NSAF Helpers - High-level API functions for field generation from files
Refactored to use modelgrid consistently and reduce redundancy
"""
import numpy as np
import pandas as pd
import os

# Import from refactored nsaf_utils
from nsaf_utils import (
    generate_single_layer,
    _detect_zone_boundaries,
    _extract_grid_info
)


def generate_fields_from_files(tmp_model_ws, model_name, conceptual_points_file,
                               zone_files=None, iid_files=None, field_name=['field'],
                               layer_mode=True, save_dir=None, tensor_interp='krig',
                               transform='log', n_realizations=1, ml=None, sr=None,
                               boundary_enhance=False, boundary_smooth=False,
                               cp_0based=False, use_depth=False):
    """
    Generate spatially correlated fields using tensor-based geostatistics
    from modelgrid and files, call generate_single_layer for each layer.

    Parameters
    ----------
    tmp_model_ws : str
        Path to model workspace containing flopy model files
    model_name : str
        Name of the flopy model (e.g., 'freyberg6')
    conceptual_points_file : str
        CSV file containing conceptual point locations and parameters
    zone_files : None, str, or dict, optional
        Zone file(s). Can be:
        - None: uses modelgrid.idomain
        - str: pattern with {} placeholder, e.g., 'zones_layer{}.arr' (replaced with 0-based layer number)
        - dict: explicit mapping of 0-based layer numbers to filenames, e.g., {0: 'zone_layer0.arr', 1: 'zone_layer1.arr'}
        If dict is provided, it must contain entries for all layers (will error if layers are missing).
    iid_files : None, str, or dict, optional
        IID file(s) containing noise arrays. Can be:
        - None: generates default files 'iids_layer{n}.arr' where n is 0-based layer number
        - str: pattern with {} placeholder, e.g., 'iids_layer{}.arr' (replaced with 0-based layer number)
        - dict: explicit mapping of 0-based layer numbers to filenames, e.g., {0: 'iid_layer0.arr', 1: 'iid_layer1.arr'}
        If dict is provided, it must contain entries for all layers (will error if layers are missing).
        If files don't exist, they will be generated and saved.
    field_name : list of str, optional, default ['field']
        Name(s) of field(s) to generate. Looks for field-specific mean/sd columns
        (e.g., 'mean_{field}' or '{field}_mean'). If empty/None, uses generic
        'mean' and 'sd' columns instead.
    layer_mode : bool, default True
        If True, process each layer independently
    save_dir : str, optional
        Directory path to save output files (default: tmp_model_ws)
    tensor_interp : str, default 'krig'
        Tensor interpolation method
    transform : str, default 'log'
        Transformation: 'log' or 'none'
    n_realizations : int, default 1
        Number of stochastic realizations
    ml : flopy.mbase, optional
        A flopy mbase derived type. If None, sr must not be None.
    sr : flopy.utils.reference.SpatialReference, optional
        A spatial reference to locate the model grid. If None, ml must not be None.
    boundary_enhance : bool, default False
        Whether to enhance boundaries in field generation
    boundary_smooth : bool, default False
        Whether to smooth boundaries in field generation
    cp_0based : bool, default False
        If False (default), conceptual points 'layer' column uses 1-based indexing (MODFLOW convention)
        and will be converted to 0-based internally. If True, 'layer' column is already 0-based.
    use_depth : bool, default False
        If True, use 'z' or 'depth' coordinate from conceptual points to determine layer by
        intersecting with model grid. Requires gridit package. Not yet implemented.

    Returns
    -------
    dict
        Dictionary of results for each field and layer, keyed by '{field}_layer{n}' where n is 0-based
    """
    try:
        import flopy
        import pyemu
    except ImportError as e:
        raise ImportError(f"flopy and pyemu are required: {e}")

    if use_depth:
        raise NotImplementedError("use_depth=True requires gridit package integration - not yet implemented")

    print(f"=== Tensor-Based Field Generation ===")
    print(f"Model workspace: {tmp_model_ws}")
    print(f"Model name: {model_name}")

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

    # Get modelgrid from the model
    modelgrid = ml.modelgrid

    # Extract grid info for display
    grid_info = _extract_grid_info(modelgrid)
    ny = grid_info['ny']
    nx = grid_info['nx']
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']

    print(f"Grid dimensions: {ny} x {nx}")
    print(f"Grid extent: X=[{xcentergrid.min():.1f}, {xcentergrid.max():.1f}], "
          f"Y=[{ycentergrid.min():.1f}, {ycentergrid.max():.1f}]")

    # Load conceptual points
    conceptual_points = pd.read_csv(os.path.join(tmp_model_ws, conceptual_points_file))
    print(f"Loaded {len(conceptual_points)} conceptual points")
    print(f"Conceptual points extent: X=[{conceptual_points['x'].min():.1f}, "
          f"{conceptual_points['x'].max():.1f}], Y=[{conceptual_points['y'].min():.1f}, "
          f"{conceptual_points['y'].max():.1f}]")

    # Add transverse if missing
    if 'transverse' not in conceptual_points.columns and 'anisotropy' in conceptual_points.columns:
        conceptual_points['transverse'] = conceptual_points['major'] / conceptual_points['anisotropy']

    # Determine total layers
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

    # Handle layer assignment in conceptual points
    if use_depth:
        # Future implementation: use z/depth coordinate to intersect with grid
        # This would call a function using gridit package
        conceptual_points = _assign_layers_from_depth(conceptual_points, modelgrid)
    elif 'layer' in conceptual_points.columns:
        # Convert layer numbering to 0-based if needed
        if not cp_0based:
            print("Converting conceptual points 'layer' column from 1-based to 0-based indexing")
            conceptual_points['layer'] = conceptual_points['layer'] - 1
            print(f"  Layer range after conversion: [{conceptual_points['layer'].min()}, "
                  f"{conceptual_points['layer'].max()}]")

        available_cp_layers = sorted(conceptual_points['layer'].unique())
        print(f"Conceptual points available for layers (0-based): {available_cp_layers}")

        # Validate layer numbers
        if conceptual_points['layer'].min() < 0:
            raise ValueError(f"Conceptual points contain negative layer numbers after conversion. "
                             f"Check cp_0based parameter.")
        if conceptual_points['layer'].max() >= total_layers:
            raise ValueError(f"Conceptual points contain layer {conceptual_points['layer'].max()} "
                             f"but model only has {total_layers} layers (0-{total_layers - 1})")
    else:
        available_cp_layers = [0]
        conceptual_points['layer'] = 0
        print("No 'layer' column in conceptual points - assigning all points to layer 0")

    # Load zones
    zone_paths = {}
    if zone_files is None:
        # Use modelgrid idomain for all layers
        print("Using modelgrid.idomain for zones")
        for layer_num in range(total_layers):
            zone_paths[layer_num] = None  # Signal to use idomain
    elif isinstance(zone_files, str):
        # String pattern - must contain {} placeholder
        if "{}" not in zone_files:
            raise ValueError(f"zone_files pattern must contain '{{}}' placeholder, got: {zone_files}")
        for layer_num in range(total_layers):
            zone_paths[layer_num] = os.path.join(tmp_model_ws, zone_files.format(layer_num))
    elif isinstance(zone_files, dict):
        # Explicit dict mapping - must contain all layers
        for layer_num in range(total_layers):
            if layer_num not in zone_files:
                raise ValueError(f"zone_files dict missing entry for layer {layer_num}. "
                                 f"Dict must contain all layers 0-{total_layers - 1}")
            zone_paths[layer_num] = os.path.join(tmp_model_ws, zone_files[layer_num])
    else:
        raise ValueError(f"zone_files must be None, str, or dict. Got: {type(zone_files)}")

    # Load zone arrays
    zones_dict = {}
    for layer_num, zone_path in zone_paths.items():
        if zone_path is None:
            # Use idomain
            if hasattr(modelgrid, 'idomain') and modelgrid.idomain is not None:
                if len(modelgrid.idomain.shape) == 3:
                    zones_dict[layer_num] = modelgrid.idomain[layer_num]
                else:
                    zones_dict[layer_num] = modelgrid.idomain
            else:
                zones_dict[layer_num] = None
        else:
            # Load from file
            z = np.loadtxt(zone_path).astype(np.int64)
            if z.shape != (ny, nx):
                if z.size == ny * nx:
                    z = z.reshape(ny, nx)
                else:
                    raise ValueError(f"Zone file {zone_path} shape {z.shape} doesn't match grid {(ny, nx)}")
            zones_dict[layer_num] = z
            print(f"  Loaded zones for layer {layer_num} from {os.path.basename(zone_path)}")

        # Resolve file paths for iids
        iids_dict = {}
        if iid_files is None:
            # No IID files specified - pass None to trigger FIELDGEN2D_SVA internal generation
            print("No IID files specified - will use FIELDGEN2D_SVA internal random generation")
            for layer_num in range(total_layers):
                iids_dict[layer_num] = None
        else:
            # IID files specified - resolve paths and load/generate
            iid_paths = {}
            for layer_num in range(total_layers):
                if isinstance(iid_files, str):
                    # String pattern - must contain {} placeholder
                    if "{}" not in iid_files:
                        raise ValueError(f"iid_files pattern must contain '{{}}' placeholder, got: {iid_files}")
                    iid_path = os.path.join(tmp_model_ws, iid_files.format(layer_num))
                elif isinstance(iid_files, dict):
                    # Explicit dict mapping - must contain all layers
                    if layer_num not in iid_files:
                        raise ValueError(f"iid_files dict missing entry for layer {layer_num}. "
                                         f"Dict must contain all layers 0-{total_layers - 1}")
                    iid_path = os.path.join(tmp_model_ws, iid_files[layer_num])
                else:
                    raise ValueError(f"iid_files must be None, str, or dict. Got: {type(iid_files)}")

                iid_paths[layer_num] = iid_path

            print(f"IID file paths resolved for {len(iid_paths)} layers")

            # Now load or generate IIDs using the resolved paths
            for layer_num, iid_path in iid_paths.items():
                iids_dict[layer_num] = _load_or_generate_iids(
                    iid_path, ny, nx, 1  # Single realization per call
                )

    # Process each field
    all_results = {}
    # If no field names provided, use generic columns once
    if not field_name or len(field_name) == 0:
        mean_candidates = [col for col in conceptual_points.columns if 'mean' in col.lower()]
        sd_candidates = [col for col in conceptual_points.columns if 'sd' in col.lower()]

        if mean_candidates and sd_candidates:
            mean_col = mean_candidates[0]
            sd_col = sd_candidates[0]
            print(f"Using generic columns: mean='{mean_col}', sd='{sd_col}'")
            field_name = ['']  # Empty string for generic processing
        else:
            raise ValueError("No mean/sd columns found")

    # Process each specific field
    for fn, trans in zip(field_name, transform):
        print(f"\n=== Processing field: {fn if fn else 'generic'} ===")

        # Try field-specific columns only if fn is not empty
        if fn:  # If we have a specific field name
            sd_is_log_space = False  # Default assumption

            if trans == 'log':
                # For log-transformed fields, prefer log-space sd columns
                if f'mean_{fn}' in conceptual_points.columns:
                    mean_col = f'mean_{fn}'
                    # Look for log-space sd (multiple naming conventions)
                    if f'logsd{fn}' in conceptual_points.columns:
                        sd_col = f'logsd{fn}'
                        sd_is_log_space = True
                    elif f'log_sd_{fn}' in conceptual_points.columns:
                        sd_col = f'log_sd_{fn}'
                        sd_is_log_space = True
                    elif f'sd_{fn}' in conceptual_points.columns:
                        sd_col = f'sd_{fn}'
                        sd_is_log_space = False
                        print(f"  WARNING: Using native-space 'sd_{fn}' for log-transformed field '{fn}'. "
                              f"Consider providing 'logsd{fn}' or 'log_sd_{fn}' instead for better interpretation.")
                    else:
                        raise ValueError(
                            f"No sd column found for field '{fn}'. Expected 'sd_{fn}', 'logsd{fn}', or 'log_sd_{fn}'")
                elif f'{fn}_mean' in conceptual_points.columns:
                    mean_col = f'{fn}_mean'
                    # Look for log-space sd (multiple naming conventions)
                    if f'{fn}_logsd' in conceptual_points.columns:
                        sd_col = f'{fn}_logsd'
                        sd_is_log_space = True
                    elif f'{fn}_log_sd' in conceptual_points.columns:
                        sd_col = f'{fn}_log_sd'
                        sd_is_log_space = True
                    elif f'{fn}_sd' in conceptual_points.columns:
                        sd_col = f'{fn}_sd'
                        sd_is_log_space = False
                        print(f"  WARNING: Using native-space '{fn}_sd' for log-transformed field '{fn}'. "
                              f"Consider providing '{fn}_logsd' or '{fn}_log_sd' instead for better interpretation.")
                    else:
                        raise ValueError(
                            f"No sd column found for field '{fn}'. Expected '{fn}_sd', '{fn}_logsd', or '{fn}_log_sd'")
                else:
                    raise ValueError(f"No mean column found for field '{fn}'. Expected 'mean_{fn}' or '{fn}_mean'")
            else:
                # Normal space - use regular sd columns
                if f'mean_{fn}' in conceptual_points.columns:
                    mean_col = f'mean_{fn}'
                    sd_col = f'sd_{fn}'
                elif f'{fn}_mean' in conceptual_points.columns:
                    mean_col = f'{fn}_mean'
                    sd_col = f'{fn}_sd'
                else:
                    raise ValueError(f"No mean/sd columns found for field '{fn}'. "
                                     f"Expected 'mean_{fn}' or '{fn}_mean'")
        # else: mean_col and sd_col already set from generic case

        # Verify columns exist
        if mean_col not in conceptual_points.columns:
            raise ValueError(f"Mean column '{mean_col}' not found")
        if sd_col not in conceptual_points.columns:
            raise ValueError(f"SD column '{sd_col}' not found")

        # Handle layer processing
        if layer_mode and 'layer' in conceptual_points.columns:
            # Process each available layer
            for target_layer in available_cp_layers:
                print(f"\n--- Processing layer {target_layer} (0-based) ---")

                # Filter conceptual points for this layer
                layer_cp = conceptual_points[conceptual_points['layer'] == target_layer].copy()

                if len(layer_cp) == 0:
                    print(f"No conceptual points for layer {target_layer}, skipping")
                    continue

                # Check data ranges
                print(f"Conceptual point data for layer {target_layer}:")
                print(f"  {mean_col}: [{layer_cp[mean_col].min():.2f}, {layer_cp[mean_col].max():.2f}]")
                print(f"  {sd_col}: [{layer_cp[sd_col].min():.2f}, {layer_cp[sd_col].max():.2f}]")

                # Get IIDs for this layer (now directly indexed with 0-based)
                layer_iids = iids_dict[target_layer]

                # Get zones for this layer (now directly indexed with 0-based)
                layer_zones = zones_dict.get(target_layer, None)

                # Use generate_single_layer which internally calls apply_nsaf_hyperpars
                results = generate_single_layer(
                    conceptual_points=layer_cp,
                    modelgrid=modelgrid,
                    iids=layer_iids,
                    zones=layer_zones,
                    layer=target_layer,
                    tensor_interp=tensor_interp,
                    transform=trans,
                    boundary_smooth=boundary_smooth,
                    boundary_enhance=boundary_enhance,
                    mean_col=mean_col,
                    sd_col=sd_col,
                    n_realizations=n_realizations,
                    sd_is_log_space=sd_is_log_space
                )

                results[fn] = results['field']

                # Store results with 0-based layer numbering
                if fn:
                    result_key = f"{fn}_layer{target_layer}"
                else:
                    result_key = f"field_layer{target_layer}"
                all_results[result_key] = results

                print(f"Field result for layer {target_layer}:")
                print(f"  Mean field: [{results['mean'].min():.3f}, {results['mean'].max():.3f}]")
                print(f"  SD field: [{results['sd'].min():.3f}, {results['sd'].max():.3f}]")

                # Save outputs if requested
                if save_dir is not None:
                    fig_path = os.path.join(save_dir, 'figure')
                    if not os.path.exists(fig_path):
                        os.makedirs(fig_path)

                    from pyemu import plot_utils as pu

                    # Prepare filename
                    fname = f"layer{target_layer}.arr"
                    np.savetxt(os.path.join(save_dir, fname), results[fn], fmt="%20.8E")

                    # Visualization
                    pu.visualize_nsaf(results, layer_cp, xcentergrid, ycentergrid,
                                      transform=transform,
                                      domain=ml.dis.idomain[target_layer],
                                      title_suf=mean_col.split('_')[-1] if '_' in mean_col else mean_col,
                                      save_path=os.path.join(fig_path, fname.replace('.arr', '.png')))
        else:
            # Non-layer mode - process all conceptual points together
            # TODO: Implement 3D field generation
            raise NotImplementedError("Non-layer mode (3D field generation) not yet implemented")

    print("\n=== Complete ===")
    return all_results


def _assign_layers_from_depth(conceptual_points, modelgrid):
    """
    Assign layer numbers to conceptual points based on z/depth coordinates
    by intersecting with model grid elevations.

    This function will use the gridit package to perform the intersection.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with 'z' or 'depth' column
    modelgrid : flopy.discretization.Grid
        Model grid with layer elevation information

    Returns
    -------
    pd.DataFrame
        Conceptual points with 'layer' column added (0-based)
    """
    # TODO: Implement using gridit package
    # Expected workflow:
    # 1. Extract z coordinate from conceptual_points ('z' or 'depth' column)
    # 2. Get layer elevations from modelgrid (top and botm arrays)
    # 3. Use gridit to find which layer each point falls into
    # 4. Add 'layer' column to conceptual_points (0-based)
    pass


def create_boundary_modified_scalar(base_field, zones,
                                    peak_increase=0.3, transition_cells=5, mode="enhance"):
    """
    Modify scalar field values near geological zone boundaries.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if mode not in ("enhance", "smooth"):
        raise ValueError("mode must be 'enhance' or 'smooth'")

    # 2D case only for now
    if zones.shape != base_field.shape:
        raise ValueError(f"Zones shape {zones.shape} must match field shape {base_field.shape}")

    boundary_mask, _ = _detect_zone_boundaries(zones)
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

    print(f"    {'Enhanced' if mode == 'enhance' else 'Smoothed'} "
          f"{np.count_nonzero(transition_mask)} points near boundaries")

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


def save_results(result, out_filename):
    """Save results to files in PyEMU-compatible format."""

    # todo: better file name handling

    # Save mean field
    mean_file = f"{out_filename}_mean.txt"
    np.savetxt(mean_file, result['mean'], fmt="%20.8E")
    print(f"  Saved mean field to {mean_file}")

    # Save standard deviation field
    sd_file = f"{out_filename}_sd.txt"
    np.savetxt(sd_file, result['sd'], fmt="%20.8E")
    print(f"  Saved SD field to {sd_file}")

    # Save geostatistical parameter field
    for param in ['bearing', 'anisotropy', 'corrlen']:
        param_file = f"{out_filename}_{param}.txt"
        np.savetxt(param_file, result[param], fmt="%20.8E")
        print(f"  Saved {param} field to {param_file}")

    # Save stochastic realizations
    if result['field'] is not None:
        for i, field in enumerate(result['field']):
            field_file = f"{out_filename}_real_{i + 1:03d}.txt"
            np.savetxt(field_file, field, fmt="%20.8E")

        print(f"  Saved {len(result['field'])} realizations")


def _load_or_generate_iids(iid_path, ny, nx, n_realizations):
    """
    Load IIDs from file or generate and save new ones.

    Parameters
    ----------
    iid_path : str
        Explicit path to IID file (including directory)
    ny, nx : int
        Grid dimensions
    n_realizations : int
        Number of realizations needed

    Returns
    -------
    np.ndarray
        IIDs array with shape (ny*nx, n_realizations)
    """

    # Try to load existing IIDs
    if os.path.exists(iid_path):
        print(f"  Loading existing IIDs from {os.path.basename(iid_path)}")
        try:
            # Load as text array
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
                print(f"    Expanding IIDs from {iids.shape[1]} to {n_realizations} realizations")
                additional_iids = np.random.normal(0, 1, size=(ny * nx, n_realizations - iids.shape[1]))
                iids = np.column_stack([iids, additional_iids])
            elif iids.shape[1] > n_realizations:
                print(f"    Truncating IIDs from {iids.shape[1]} to {n_realizations} realizations")
                iids = iids[:, :n_realizations]

            return iids

        except Exception as e:
            print(f"  Warning: Could not load IID file: {e}")
            print(f"  Generating new IIDs...")
    else:
        print(f"  IID file not found: {os.path.basename(iid_path)}")
        print(f"  Generating new IIDs...")

    # Generate new IIDs
    iids = np.random.normal(0, 1, size=(ny * nx, n_realizations))

    # Save for future use
    print(f"  Saving IIDs to {os.path.basename(iid_path)}")
    os.makedirs(os.path.dirname(iid_path), exist_ok=True)
    np.savetxt(iid_path, iids, fmt="%.8f",
               header=f"IIDs shape: {ny * nx} x {n_realizations}")

    return iids


def example_field_generation():
    """Example using concentric circles of tangentially-oriented anisotropic tensors."""
    import flopy
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Create example PyEMU grid - square domain
    nx, ny = 100, 100
    xmax = 5000
    ymax = 5000

    # Correct: delr is row spacing (y-direction), delc is column spacing (x-direction)
    delr = np.full(ny, ymax / ny)  # Cell height (y-direction, row spacing)
    delc = np.full(nx, xmax / nx)  # Cell width (x-direction, column spacing)

    modelgrid = flopy.discretization.StructuredGrid(
        delc=delc, delr=delr,
        top=np.ones((ny, nx)),
        botm=np.zeros((1, ny, nx)),
        xoff=0,
        yoff=0,
        angrot=0.0,
        idomain=np.ones((ny, nx)),
    )

    grid_info = _extract_grid_info(modelgrid)
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']

    center_x = 2500
    center_y = 2500
    zones = np.ones((ny, nx))
    zones[(ycentergrid < center_y) & (xcentergrid > center_x)] = 2  # SE
    zones[(ycentergrid < center_y) & (xcentergrid < center_x)] = 3  # SW
    zones[(ycentergrid > center_y) & (xcentergrid < center_x)] = 4  # NE
    # Define circle center and radius

    max_radius = 2500*np.sqrt(2)

    # Create conceptual points on concentric circles
    cp_list = []
    n_circles = 4

    # Define zone-specific mean values
    zone_mean_multipliers = {
        1: 1.0,  # NW quadrant - baseline
        2: 2.0,  # SE quadrant - 2x higher
        3: 3.0,  # SW quadrant - 2x lower
        4: 4.0  # NE quadrant - 4x higher
    }

    for circle_idx in range(n_circles):
        radius_fraction = (circle_idx + 0.3) / n_circles
        cp_radius = max_radius * radius_fraction
        n_points = 6 + circle_idx * 4
        base_mean_kh = 2 ** (circle_idx - n_circles)  # Base value from circle
        major_length = 500 + 50 * radius_fraction

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points

            # Position on circle
            cp_x = center_x + cp_radius * np.cos(angle)
            cp_y = center_y + cp_radius * np.sin(angle)

            # Determine which zone this point is in
            if cp_y > center_y and cp_x > center_x:
                point_zone = 1  # NW
            elif cp_y < center_y and cp_x > center_x:
                point_zone = 2  # SE
            elif cp_y < center_y and cp_x < center_x:
                point_zone = 3  # SW
            else:  # cp_y > center_y and cp_x < center_x
                point_zone = 4  # NE

            # Apply zone-specific multiplier
            zone_multiplier = zone_mean_multipliers[point_zone]
            mean_kh = base_mean_kh * zone_multiplier

            # Tangential bearing calculation
            geo_radial = np.degrees(np.arctan2(cp_x - center_x, cp_y - center_y))
            bearing = (geo_radial + 90) % 360

            cp_list.append({
                'name': f'cp_circle{circle_idx}_point{i}_zone{point_zone}',
                'x': cp_x,
                'y': cp_y,
                'mean': mean_kh,
                'sd': 1.0,
                'major': major_length,
                'anisotropy': 4.0,
                'bearing': bearing,
                'zone': point_zone  # Optional: track which zone
            })

    cp_df = pd.DataFrame(cp_list)
    cp_df['transverse'] = cp_df['major'] / cp_df['anisotropy']

    # Generate field
    print("=== Testing tangential field generation ===")
    for scen in ['fieldgen', 'python']:
        if scen=='fieldgen':
            iids = None # let fieldgen create them
        else:
            iids = np.random.normal(0, 1, size=(ny * nx))
        results = generate_single_layer(
            cp_df, modelgrid,
            iids=iids,
            zones=zones,
            tensor_interp='krig',
            transform='log',
            boundary_smooth={'transition_cells': 5},
            boundary_enhance={'transition_cells': 5,
                              'peak_increase': 2},
            sd_is_log_space=True
        )

        print(f"Field statistics: mean={np.mean(results['field']):.3f}, std={np.std(results['field']):.3f}")
        print(f"Number of conceptual points: {len(cp_df)}")
        print("Tangential field generation complete!")

        save_dir = os.path.join('..','..','examples',f'tangential_nsaf_{scen}')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if save_dir is not None:
            from pyemu import plot_utils as pu
            grid_info = _extract_grid_info(modelgrid)
            xcentergrid = grid_info['xcentergrid']
            ycentergrid = grid_info['ycentergrid']
            fname = f"tangential_example_{scen}.arr"
            np.savetxt(os.path.join(save_dir, fname), results['field'], fmt="%20.8E")
            fig_path = os.path.join(save_dir, 'figure')
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            pu.visualize_tensors(results['tensors'], xcentergrid, ycentergrid, zones=zones,
                                 conceptual_points=cp_df, subsample=4, max_ellipse_size=0.1,
                                 figsize=(14, 12), title_suf='tangential',
                                 save_path=os.path.join(fig_path, fname.replace('.arr', '_tensors.png')))

            pu.visualize_nsaf(results, cp_df, xcentergrid, ycentergrid,
                              transform='log', title_suf='tangential',
                              save_path=os.path.join(fig_path, fname.replace('.arr', '.png')))

    return results, cp_df

if __name__ == "__main__":
    example_field_generation()