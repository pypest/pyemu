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
    Follows PyEMU pattern for model workspace and spatial reference handling.

    Parameters
    ----------
    tmp_model_ws : str
        Path to model workspace containing flopy model files
    model_name : str
        Name of the flopy model (e.g., 'freyberg6')
    conceptual_points_file : str
        CSV file containing conceptual points
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
        # Load model from workspace
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

    if sr is None:
        # Create spatial reference from model
        print("Creating spatial reference...")
        if hasattr(ml, 'dis'):
            # MF6 or MODFLOW with DIS
            try:
                sr = pyemu.helpers.SpatialReference.from_namfile(
                    os.path.join(tmp_model_ws, f"{model_name}.nam"),
                    delr=ml.dis.delr.array, delc=ml.dis.delc.array
                )
            except:
                # Alternative approach for MF6
                sr = pyemu.helpers.SpatialReference(
                    delr=ml.dis.delr.array, delc=ml.dis.delc.array,
                    xul=0, yul=0
                )
        else:
            raise ValueError("Could not determine grid spacing from model")

        print(f"Spatial reference: {sr}")

    # Get grid coordinates
    try:
        xcentergrid = sr.xcentergrid
        ycentergrid = sr.ycentergrid
    except AttributeError:
        try:
            xcentergrid = sr.xcellcenters
            ycentergrid = sr.ycellcenters
        except AttributeError as e:
            raise Exception(f"Could not get grid centers from spatial reference: {e}")

    ny, nx = xcentergrid.shape
    print(f"Grid dimensions: {ny} x {nx}")
    cell_area = np.outer(sr.delr, sr.delc)
    # Load conceptual points
    conceptual_points = pd.read_csv(os.path.join(tmp_model_ws, conceptual_points_file))

    # Add transverse if missing
    if 'transverse' not in conceptual_points.columns and 'anisotropy' in conceptual_points.columns:
        conceptual_points.loc[:, 'transverse'] = conceptual_points.loc[:, 'major'] / conceptual_points.loc[:,
                                                                                     'anisotropy']

    print(f"Loaded {len(conceptual_points)} conceptual points")

    # Load zones
    zones = None
    if zone_file:
        if isinstance(zone_file, str):
            # Single zone file
            zones = np.loadtxt(os.path.join(tmp_model_ws, zone_file)).astype(np.int64)
            if zones.shape != (ny, nx):
                # Try to reshape if possible
                if zones.size == ny * nx:
                    zones = zones.reshape(ny, nx)
                else:
                    raise ValueError(f"Zone file shape {zones.shape} doesn't match grid {(ny, nx)}")
            print(f"Loaded zones from {zone_file}, shape: {zones.shape}")
        else:
            # Multiple zone files - for now just use the first one
            zones = np.loadtxt(os.path.join(tmp_model_ws, zone_file[0])).astype(np.int64)
            print(f"Loaded zones from {zone_file[0]} (multi-layer support TODO)")

    # Determine the total number of layers in the model
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

    # Handle IIDs for PEST parameterization - generate for ALL model layers
    iids_dict = {}

    # Determine which layers have conceptual points
    if 'layer' in conceptual_points.columns:
        available_cp_layers = sorted(conceptual_points['layer'].unique())
        print(f"Conceptual points available for layers: {available_cp_layers}")
    else:
        available_cp_layers = []
        print("No 'layer' column in conceptual points - will propagate to all layers")

    # Generate IIDs for all model layers
    for layer_num in range(total_layers):
        layer_1based = layer_num + 1  # Convert to 1-based indexing
        iids_dict[layer_1based] = _load_or_generate_iids(
            tmp_model_ws, iids_file, layer_num, ny, nx, n_realizations
        )

    # Propagate conceptual points to missing layers if needed
    if layer_mode and 'layer' in conceptual_points.columns:
        # Check if all layers are covered
        missing_layers = []
        for layer_num in range(1, total_layers + 1):  # 1-based
            if layer_num not in available_cp_layers:
                missing_layers.append(layer_num)

        if missing_layers:
            print(f"Missing conceptual points for layers: {missing_layers}")
            print("Propagating conceptual points from available layers...")

            # Find the best layer to propagate from (prefer middle layers)
            if available_cp_layers:
                # Use the layer closest to the middle of available layers
                middle_idx = len(available_cp_layers) // 2
                source_layer = available_cp_layers[middle_idx]
                print(f"Using layer {source_layer} as source for propagation")

                # Create propagated conceptual points
                source_cp = conceptual_points[conceptual_points['layer'] == source_layer].copy()

                for missing_layer in missing_layers:
                    propagated_cp = source_cp.copy()
                    propagated_cp['layer'] = missing_layer

                    # Optionally modify some properties for different layers
                    # (could add depth-dependent scaling here if needed)

                    # Append to conceptual points
                    conceptual_points = pd.concat([conceptual_points, propagated_cp], ignore_index=True)
                    print(f"  Propagated {len(source_cp)} points to layer {missing_layer}")

            # Update available layers
            available_cp_layers = sorted(conceptual_points['layer'].unique())
            print(f"Updated conceptual points now cover layers: {available_cp_layers}")

    # Process each field for each layer
    all_results = {}

    # If layer_mode is True, process each layer separately
    if layer_mode and 'layer' in conceptual_points.columns:
        layers_to_process = available_cp_layers
    else:
        # Single layer mode - use layer 1
        layers_to_process = [1]

    # Process each field
    all_results = {}

    for fn in field_name:
        print(f"\n=== Processing field: {fn} ===")

        # Filter conceptual points for this field
        cols = ['name', 'x', 'y', 'major', 'transverse', 'bearing']
        field_cols = [col for col in conceptual_points.columns if f'_{fn}' in col]

        if not field_cols:
            # Look for generic mean/sd columns
            mean_cols = [col for col in conceptual_points.columns if 'mean' in col.lower()]
            sd_cols = [col for col in conceptual_points.columns if 'sd' in col.lower()]

            if mean_cols and sd_cols:
                # Rename to expected format
                field_cp = conceptual_points.copy()
                field_cp[f'mean_{fn}'] = field_cp[mean_cols[0]]
                field_cp[f'sd_{fn}'] = field_cp[sd_cols[0]]
                cols.extend([f'mean_{fn}', f'sd_{fn}'])
            else:
                raise ValueError(f"No columns found for field '{fn}' (expected mean_{fn}, sd_{fn})")
        else:
            cols.extend(field_cols)

        # Auto-detect layers and select target layer
        target_layer = None
        if layer_mode and 'layer' in conceptual_points.columns:
            cols.append('layer')
            # Auto-detect available layers
            available_layers = sorted(conceptual_points['layer'].unique())
            print(f"Available layers: {available_layers}")

            # Use first available layer instead of hardcoded 0
            target_layer = available_layers[0]
            field_cp = conceptual_points[conceptual_points['layer'] == target_layer][cols].copy()

            if len(field_cp) == 0:
                print(f"No conceptual points for layer {target_layer}, using all points")
                field_cp = conceptual_points[cols].copy()
                target_layer = None
        else:
            field_cp = conceptual_points[cols].copy()

        # Rename columns to match apply_ppu_hyperpars function expectations
        if f'mean_{fn}' in field_cp.columns:
            field_cp = field_cp.rename(columns={f'mean_{fn}': 'mean_kh', f'sd_{fn}': 'sd_kh'})
        elif 'mean' in field_cp.columns:
            field_cp = field_cp.rename(columns={'mean': 'mean_kh', 'sd': 'sd_kh'})

        # Get IIDs for this layer
        layer_key = target_layer if target_layer is not None else 0
        layer_iids = iids_dict.get(layer_key, iids_dict)

        # Use workflow
        results = apply_ppu_hyperpars(
            field_cp, xcentergrid, ycentergrid,
            area=cell_area,
            zones=zones,
            n_realizations=n_realizations,
            layer=target_layer if target_layer is not None else 0,
            vartransform=vartransform,
            tensor_method=tensor_interp,
            out_filename=os.path.join(save_path, f"{fn}"),
            iids=layer_iids  # Pass the IIDs
        )

        all_results[fn] = results

        # Save and visualize results
        if results['fields'] is not None:
            for i, field in enumerate(results['fields']):
                save_layer(field, layer=target_layer if target_layer is not None else 0,
                           field_name=f"{fn}_real_{i + 1}", save_path=save_path)

        # Save mean and SD
        save_layer(results['mean'], layer=target_layer if target_layer is not None else 0,
                   field_name=f"{fn}_mean", save_path=save_path)
        save_layer(results['sd'], layer=target_layer if target_layer is not None else 0,
                   field_name=f"{fn}_sd", save_path=save_path)

        # Visualize tensors - check if visualization module is available
        try:
            from tensor_visualization import visualize_tensors
            visualize_tensors(
                results['tensors'], xcentergrid, ycentergrid,
                zones=zones, conceptual_points=field_cp,
                subsample=8, title_suf=f'{fn}_layer_{target_layer if target_layer is not None else 0}',
                save_path=save_path
            )
        except ImportError:
            print("  Warning: tensor_visualization module not available, skipping visualization")

    print("\n=== Complete ===")
    return all_results


def _load_or_generate_iids(tmp_model_ws, iids_file, layer_num, ny, nx, n_realizations):
    """
    Load IIDs from file or generate and save new ones for PEST parameterization.

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
    """
    Generate spatially correlated fields using tensor-based geostatistics.
    Follows PyEMU pattern for model workspace and spatial reference handling.

    Parameters
    ----------
    tmp_model_ws : str
        Path to model workspace containing flopy model files
    model_name : str
        Name of the flopy model (e.g., 'freyberg6')
    conceptual_points_file : str
        CSV file containing conceptual points
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
        # Load model from workspace
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

    if sr is None:
        # Create spatial reference from model
        print("Creating spatial reference...")
        if hasattr(ml, 'dis'):
            # MF6 or MODFLOW with DIS
            try:
                sr = pyemu.helpers.SpatialReference.from_namfile(
                    os.path.join(tmp_model_ws, f"{model_name}.nam"),
                    delr=ml.dis.delr.array, delc=ml.dis.delc.array
                )
            except:
                # Alternative approach for MF6
                sr = pyemu.helpers.SpatialReference(
                    delr=ml.dis.delr.array, delc=ml.dis.delc.array,
                    xul=0, yul=0
                )
        else:
            raise ValueError("Could not determine grid spacing from model")

        print(f"Spatial reference: {sr}")

    # Get grid coordinates
    try:
        xcentergrid = sr.xcentergrid
        ycentergrid = sr.ycentergrid
    except AttributeError:
        try:
            xcentergrid = sr.xcellcenters
            ycentergrid = sr.ycellcenters
        except AttributeError as e:
            raise Exception(f"Could not get grid centers from spatial reference: {e}")

    ny, nx = xcentergrid.shape
    print(f"Grid dimensions: {ny} x {nx}")

    # Load conceptual points
    conceptual_points = pd.read_csv(os.path.join(tmp_model_ws, conceptual_points_file))

    # Add transverse if missing
    if 'transverse' not in conceptual_points.columns and 'anisotropy' in conceptual_points.columns:
        conceptual_points.loc[:, 'transverse'] = conceptual_points.loc[:, 'major'] / conceptual_points.loc[:,
                                                                                     'anisotropy']

    print(f"Loaded {len(conceptual_points)} conceptual points")

    # Load zones
    zones = None
    if zone_file:
        if isinstance(zone_file, str):
            # Single zone file
            zones = np.loadtxt(os.path.join(tmp_model_ws, zone_file)).astype(np.int64)
            if zones.shape != (ny, nx):
                # Try to reshape if possible
                if zones.size == ny * nx:
                    zones = zones.reshape(ny, nx)
                else:
                    raise ValueError(f"Zone file shape {zones.shape} doesn't match grid {(ny, nx)}")
            print(f"Loaded zones from {zone_file}, shape: {zones.shape}")
        else:
            # Multiple zone files - for now just use the first one
            zones = np.loadtxt(os.path.join(tmp_model_ws, zone_file[0])).astype(np.int64)
            print(f"Loaded zones from {zone_file[0]} (multi-layer support TODO)")

    # Process each field
    all_results = {}

    for fn in field_name:
        print(f"\n=== Processing field: {fn} ===")

        # Filter conceptual points for this field
        cols = ['name', 'x', 'y', 'major', 'transverse', 'bearing']
        field_cols = [col for col in conceptual_points.columns if f'_{fn}' in col]

        if not field_cols:
            # Look for generic mean/sd columns
            mean_cols = [col for col in conceptual_points.columns if 'mean' in col.lower()]
            sd_cols = [col for col in conceptual_points.columns if 'sd' in col.lower()]

            if mean_cols and sd_cols:
                # Rename to expected format
                field_cp = conceptual_points.copy()
                field_cp[f'mean_{fn}'] = field_cp[mean_cols[0]]
                field_cp[f'sd_{fn}'] = field_cp[sd_cols[0]]
                cols.extend([f'mean_{fn}', f'sd_{fn}'])
            else:
                raise ValueError(f"No columns found for field '{fn}' (expected mean_{fn}, sd_{fn})")
        else:
            cols.extend(field_cols)

        # Auto-detect layers and select target layer
        target_layer = None
        if layer_mode and 'layer' in conceptual_points.columns:
            cols.append('layer')
            # Auto-detect available layers
            available_layers = sorted(conceptual_points['layer'].unique())
            print(f"Available layers: {available_layers}")

            # Use first available layer instead of hardcoded 0
            target_layer = available_layers[0]
            field_cp = conceptual_points[conceptual_points['layer'] == target_layer][cols].copy()

            if len(field_cp) == 0:
                print(f"No conceptual points for layer {target_layer}, using all points")
                field_cp = conceptual_points[cols].copy()
                target_layer = None
        else:
            field_cp = conceptual_points[cols].copy()

        # Rename columns to match apply_ppu_hyperpars function expectations
        if f'mean_{fn}' in field_cp.columns:
            field_cp = field_cp.rename(columns={f'mean_{fn}': 'mean_kh', f'sd_{fn}': 'sd_kh'})
        elif 'mean' in field_cp.columns:
            field_cp = field_cp.rename(columns={'mean': 'mean_kh', 'sd': 'sd_kh'})

        # Use workflow
        results = apply_ppu_hyperpars(
            field_cp, xcentergrid, ycentergrid,
            zones=zones,
            n_realizations=n_realizations,
            layer=target_layer if target_layer is not None else 0,
            vartransform=vartransform,
            tensor_method=tensor_interp,
            out_filename=os.path.join(save_path, f"{fn}")
        )

        all_results[fn] = results

        # Save and visualize results
        if results['fields'] is not None:
            for i, field in enumerate(results['fields']):
                save_layer(field, layer=target_layer if target_layer is not None else 0,
                           field_name=f"{fn}_real_{i + 1}", save_path=save_path)

        # Save mean and SD
        save_layer(results['mean'], layer=target_layer if target_layer is not None else 0,
                   field_name=f"{fn}_mean", save_path=save_path)
        save_layer(results['sd'], layer=target_layer if target_layer is not None else 0,
                   field_name=f"{fn}_sd", save_path=save_path)

        # Visualize tensors - check if visualization module is available
        try:
            from tensor_visualization import visualize_tensors
            visualize_tensors(
                results['tensors'], xcentergrid, ycentergrid,
                zones=zones, conceptual_points=field_cp,
                subsample=8, title_suf=f'{fn}_layer_{target_layer if target_layer is not None else 0}',
                save_path=save_path
            )
        except ImportError:
            print("  Warning: tensor_visualization module not available, skipping visualization")

    print("\n=== Complete ===")
    return all_results


def generate_single_layer(cp_file, xcentergrid, ycentergrid, iids=None,
                          zones=None, save_path='.', tensor_interp='krig',
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