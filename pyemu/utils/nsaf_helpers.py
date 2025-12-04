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


def generate_fields_from_files(model_name, conceptual_points_file,
                               model_ws='.',
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
    model_ws : str
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
        - None: defaults to fieldgen2d_sva internal iid generator
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
        Directory path to save output files (default: model_ws)
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
    print(f"Model workspace: {model_ws}")
    print(f"Model name: {model_name}")

    # Setup spatial reference following PyEMU pattern
    if ml is None and sr is None:
        print("Loading flopy model...")
        try:
            # Try MF6 first - only load DIS package for efficiency
            sim = flopy.mf6.MFSimulation.load(sim_ws=model_ws, load_only=['dis'])
            ml = sim.get_model(model_name)
            print(f"Loaded MF6 model: {model_name} (DIS package only)")
        except Exception as e_mf6:
            try:
                # Fall back to MODFLOW-2005 - only load DIS package
                ml = flopy.modflow.Modflow.load(f"{model_name}.nam", model_ws=model_ws,
                                                load_only=['dis'], check=False)
                print(f"Loaded MODFLOW-2005 model: {model_name} (DIS package only)")
            except Exception as e_mf2005:
                raise ValueError(f"Could not load model {model_name} from {model_ws}.\n"
                                 f"MF6 error: {e_mf6}\n"
                                 f"MF2005 error: {e_mf2005}")

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
    conceptual_points = pd.read_csv(os.path.join(model_ws, conceptual_points_file))
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
            zone_paths[layer_num] = os.path.join(model_ws, zone_files.format(layer_num))
    elif isinstance(zone_files, dict):
        # Explicit dict mapping - must contain all layers
        for layer_num in range(total_layers):
            if layer_num not in zone_files:
                raise ValueError(f"zone_files dict missing entry for layer {layer_num}. "
                                 f"Dict must contain all layers 0-{total_layers - 1}")
            zone_paths[layer_num] = os.path.join(model_ws, zone_files[layer_num])
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
                    iid_path = os.path.join(model_ws, iid_files.format(layer_num))
                elif isinstance(iid_files, dict):
                    # Explicit dict mapping - must contain all layers
                    if layer_num not in iid_files:
                        raise ValueError(f"iid_files dict missing entry for layer {layer_num}. "
                                         f"Dict must contain all layers 0-{total_layers - 1}")
                    iid_path = os.path.join(model_ws, iid_files[layer_num])
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
                                      transform=trans,
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
        IIDs array with shape (ny*nx,) for single realization or (ny*nx, n_realizations) for multiple
    """

    # Try to load existing IIDs
    if os.path.exists(iid_path):
        print(f"  Loading existing IIDs from {os.path.basename(iid_path)}")
        try:
            # Load as text array
            iids_data = np.loadtxt(iid_path)

            # Handle different shapes
            if iids_data.ndim == 1:
                # Single realization - keep as 1D if n_realizations==1
                if len(iids_data) == ny * nx:
                    if n_realizations == 1:
                        return iids_data  # Return 1D array (ny*nx,)
                    else:
                        iids = iids_data.reshape(-1, 1)
                else:
                    raise ValueError(f"IID file has wrong size: {len(iids_data)} vs {ny * nx}")
            elif iids_data.ndim == 2:
                # Could be (ny, nx) or (ny*nx, n_real)
                if iids_data.shape == (ny, nx):
                    # Saved as 2D grid - flatten it
                    iids_data = iids_data.flatten()
                    if n_realizations == 1:
                        return iids_data  # Return 1D array (ny*nx,)
                    else:
                        iids = iids_data.reshape(-1, 1)
                elif iids_data.shape[0] == ny * nx:
                    # Already in (ny*nx, n_real) format
                    iids = iids_data
                elif iids_data.shape[1] == ny * nx:
                    # Transposed - fix it
                    iids = iids_data.T
                else:
                    raise ValueError(f"IID file has wrong shape: {iids_data.shape}")
            else:
                raise ValueError(f"IID file has too many dimensions: {iids_data.ndim}")

            # If we get here and n_realizations==1, extract the single column
            if n_realizations == 1 and iids.ndim == 2:
                return iids[:, 0]  # Return 1D array (ny*nx,)

            # Expand or truncate to match n_realizations (only for multi-realization case)
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
    if n_realizations == 1:
        # Generate single realization as 1D array
        iids = np.random.normal(0, 1, size=(ny * nx))
    else:
        # Generate multiple realizations as 2D array
        iids = np.random.normal(0, 1, size=(ny * nx, n_realizations))

    # Save for future use
    print(f"  Saving IIDs to {os.path.basename(iid_path)}")
    os.makedirs(os.path.dirname(iid_path), exist_ok=True)

    if n_realizations == 1:
        # Save as 2D grid for easier viewing
        np.savetxt(iid_path, iids.reshape(ny, nx), fmt="%.8f")
    else:
        np.savetxt(iid_path, iids, fmt="%.8f",
                   header=f"IIDs shape: {ny * nx} x {n_realizations}")

    return iids


def example_field_generation_save_files():
    """Modified tangential example that saves files for testing generate_fields_from_files()."""
    import flopy
    import numpy as np
    import pandas as pd
    import os

    # Create test directory
    test_ws = os.path.join('..', '..', 'examples', 'tangential_test')
    if not os.path.exists(test_ws):
        os.makedirs(test_ws)

    # Create example grid parameters
    nx, ny = 100, 100
    xmax = 5000
    ymax = 5000

    delr = np.full(ny, ymax / ny)
    delc = np.full(nx, xmax / nx)

    # Create a minimal MODFLOW 6 model
    sim = flopy.mf6.MFSimulation(
        sim_name='tangential_test',
        sim_ws=test_ws,
        exe_name='mf6'
    )

    # Add TDIS
    tdis = flopy.mf6.ModflowTdis(
        sim,
        nper=1,
        perioddata=[(1.0, 1, 1.0)]
    )

    # Create groundwater flow model
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname='tangential_test',
        save_flows=True
    )

    # Add IMS (solver) - required for valid simulation
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option='SUMMARY',
        complexity='SIMPLE'
    )

    # Add DIS package
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=ny,
        ncol=nx,
        delr=delc,
        delc=delr,
        top=1.0,
        botm=0.0,
        idomain=1
    )

    # Add minimal IC (initial conditions) - required
    ic = flopy.mf6.ModflowGwfic(gwf, strt=1.0)

    # Add minimal NPF (node property flow) - required
    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1.0)

    # Write the simulation
    sim.write_simulation()
    print(f"Saved MODFLOW 6 model to: {test_ws}")

    # Get modelgrid for generating conceptual points
    modelgrid = gwf.modelgrid
    grid_info = _extract_grid_info(modelgrid)
    xcentergrid = grid_info['xcentergrid']
    ycentergrid = grid_info['ycentergrid']

    center_x = 2500
    center_y = 2500

    # Create zones
    zones = np.ones((ny, nx), dtype=int)
    zones[(ycentergrid < center_y) & (xcentergrid > center_x)] = 2  # SE
    zones[(ycentergrid < center_y) & (xcentergrid < center_x)] = 3  # SW
    zones[(ycentergrid > center_y) & (xcentergrid < center_x)] = 4  # NE

    # Save zones
    zone_file = 'zones_layer0.arr'
    np.savetxt(os.path.join(test_ws, zone_file), zones, fmt='%d')
    print(f"Saved zones to: {zone_file}")

    max_radius = 2500 * np.sqrt(2)

    # Create conceptual points
    cp_list = []
    n_circles = 4

    zone_mean_multipliers = {
        1: 1.0,
        2: 10.0,
        3: 100.0,
        4: 0.1
    }

    for circle_idx in range(n_circles):
        radius_fraction = (circle_idx + 0.3) / n_circles
        cp_radius = max_radius * radius_fraction
        n_points = 6 + circle_idx * 4
        base_mean_kh = 4 ** (circle_idx - n_circles)
        major_length = 500 + 50 * radius_fraction

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points

            cp_x = center_x + cp_radius * np.cos(angle)
            cp_y = center_y + cp_radius * np.sin(angle)

            if cp_y > center_y and cp_x > center_x:
                point_zone = 1
            elif cp_y < center_y and cp_x > center_x:
                point_zone = 2
            elif cp_y < center_y and cp_x < center_x:
                point_zone = 3
            else:
                point_zone = 4

            zone_multiplier = zone_mean_multipliers[point_zone]
            mean_kh = base_mean_kh * zone_multiplier

            geo_radial = np.degrees(np.arctan2(cp_x - center_x, cp_y - center_y))
            bearing = (geo_radial + 90) % 360

            cp_list.append({
                'name': f'cp_circle{circle_idx}_point{i}_zone{point_zone}',
                'x': cp_x,
                'y': cp_y,
                'layer': 0,  # 0-based layer column
                'mean_kh': mean_kh,
                'logsdkh': 0.5,  # Log-space SD
                'major': major_length,
                'anisotropy': 4.0,
                'bearing': bearing,
                'zone': point_zone
            })

    cp_df = pd.DataFrame(cp_list)
    cp_df['transverse'] = cp_df['major'] / cp_df['anisotropy']

    # Save conceptual points
    cp_file = 'conceptual_points.csv'
    cp_df.to_csv(os.path.join(test_ws, cp_file), index=False)
    print(f"Saved conceptual points to: {cp_file}")
    print(f"Columns: {list(cp_df.columns)}")

    # Generate and save IIDs
    np.random.seed(42)
    iids = np.random.normal(0, 1, size=(ny * nx))
    iid_file = 'iids_layer0.arr'
    np.savetxt(os.path.join(test_ws, iid_file), iids.reshape(ny, nx), fmt='%.6f')
    print(f"Saved IIDs to: {iid_file}")

    print(f"\n=== Files ready for testing ===")
    print(f"Test workspace: {test_ws}")

    return test_ws


# Then test with generate_fields_from_files():
def test_tangential_via_files():
    """Test tangential example through generate_fields_from_files()."""

    # Generate the files
    test_ws = example_field_generation_save_files()

    print("\n=== Testing with generate_fields_from_files() ===")

    # Test with fieldgen (iids=None)
    print("\n--- Test 1: FIELDGEN2D_SVA (iids=None) ---")
    results_fieldgen = generate_fields_from_files(
        model_ws=test_ws,
        model_name='tangential_test',
        conceptual_points_file='conceptual_points.csv',
        zone_files='zones_layer{}.arr',
        iid_files=None,  # Use FIELDGEN2D_SVA internal generation
        field_name=['kh'],
        layer_mode=True,
        save_dir=os.path.join(test_ws, 'output_fieldgen'),
        tensor_interp='krig',
        transform=['log'],
        boundary_smooth={'transition_cells': 5},
        boundary_enhance={'transition_cells': 5, 'peak_increase': 2},
        cp_0based=True  # Our layer column is already 0-based
    )

    # Test with Python fallback (iids provided)
    print("\n--- Test 2: Python fallback (iids provided) ---")
    results_python = generate_fields_from_files(
        model_ws=test_ws,
        model_name='tangential_test',
        conceptual_points_file='conceptual_points.csv',
        zone_files='zones_layer{}.arr',
        iid_files='iids_layer{}.arr',  # Use provided IIDs
        field_name=['kh'],
        layer_mode=True,
        save_dir=os.path.join(test_ws, 'output_python'),
        tensor_interp='krig',
        transform=['log'],
        boundary_smooth={'transition_cells': 5},
        boundary_enhance={'transition_cells': 5, 'peak_increase': 2},
        cp_0based=True
    )

    print("\n=== Comparison ===")
    field_fieldgen = results_fieldgen['kh_layer0']['field']
    field_python = results_python['kh_layer0']['field']

    print(f"FIELDGEN: mean={field_fieldgen.mean():.3f}, std={field_fieldgen.std():.3f}")
    print(f"Python:   mean={field_python.mean():.3f}, std={field_python.std():.3f}")

    return results_fieldgen, results_python


# Run the test
if __name__ == "__main__":
    results_fieldgen, results_python = test_tangential_via_files()