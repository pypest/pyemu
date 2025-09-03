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
    detect_zone_boundaries,
    _extract_grid_info,
    arr_to_rast
)


def generate_fields_from_files(tmp_model_ws, model_name, conceptual_points_file,
                               zone_file=None, iids_file=None, field_name=['field'],
                               layer_mode=True, save_dir=None, tensor_interp='krig',
                               transform='log', n_realizations=1, ml=None, sr=None,
                               boundary_enhance=False, boundary_smooth=False):
    """
    Generate spatially correlated fields using tensor-based geostatistics.
    REFACTORED to use modelgrid consistently and call generate_single_layer.

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

    Returns
    -------
    dict
        fieldresult from workflow
    """
    try:
        import flopy
        import pyemu
    except ImportError as e:
        raise ImportError(f"flopy and pyemu are required: {e}")

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

    # Load zones
    zones = modelgrid.idomain
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
            # Handle list of zone files for multiple layers
            zones = []
            for zf in zone_file:
                z = np.loadtxt(os.path.join(tmp_model_ws, zf)).astype(np.int64)
                if z.shape != (ny, nx):
                    if z.size == ny * nx:
                        z = z.reshape(ny, nx)
                zones.append(z)
            print(f"Loaded {len(zones)} zone arrays from multiple files")

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

    # Generate or load IIDs for all model layers
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

                # Get zones for this layer if available
                layer_zones = None
                if zones is not None:
                    if isinstance(zones, list) or (isinstance(zones, np.ndarray) & (len(zones.shape)==3)):
                        if target_layer - 1 < len(zones):
                            layer_zones = zones[target_layer - 1]
                    else:
                        layer_zones = zones

                # Use generate_single_layer which internally calls apply_nsaf_hyperpars
                results = generate_single_layer(
                    conceptual_points=layer_cp,
                    modelgrid=modelgrid,
                    iids=layer_iids,
                    zones=layer_zones,
                    layer=target_layer,
                    tensor_interp=tensor_interp,
                    transform=transform,
                    boundary_smooth=boundary_smooth,
                    boundary_enhance=boundary_enhance,
                    mean_col=mean_col,
                    sd_col=sd_col,
                    n_realizations=n_realizations
                )

                field = results['fields']

                # Store results
                result_key = f"{fn}_layer_{target_layer}"
                all_results[result_key] = results

                print(f"fieldresult for layer {target_layer}:")
                print(f"  Mean field: [{results['mean'].min():.3f}, {results['mean'].max():.3f}]")
                print(f"  SD field: [{results['sd'].min():.3f}, {results['sd'].max():.3f}]")
                # todo: better file name handling
                if save_dir is not None:
                    fig_path = os.path.join(save_dir, 'figure')
                    if not os.path.exists(fig_path):
                        os.mkdir(fig_path)
                    from pyemu import plot_utils as pu
                    xul = modelgrid.xoffset
                    yul = modelgrid.yoffset # + np.sum(modelgrid.delr)
                    if modelgrid.epsg is None:
                        epsg = 2193
                    else:
                        epsg = modelgrid.epsg
                    grid_info = _extract_grid_info(modelgrid)
                    xcentergrid = grid_info['xcentergrid']
                    ycentergrid = grid_info['ycentergrid']
                    fname = f"layer{target_layer}.arr"
                    np.savetxt(os.path.join(save_dir, fname), results['fields'][0], fmt="%20.8E")
                    arr_to_rast(results['fields'][0],
                                os.path.join(fig_path, fname.replace('.arr','.tif')),
                                xul, yul,
                                np.mean(modelgrid.delc), np.mean(modelgrid.delr), epsg)

                    # pu.visualize_tensors(results['tensors'], xcentergrid, ycentergrid, zones=zones[target_layer-1],
                    #                      conceptual_points=layer_cp, subsample=20, max_ellipse_size=0.1,
                    #                      figsize=(14, 12), title_suf=mean_col,
                    #                      save_path=os.path.join(fig_path, fname.replace('.arr', '_tensors.png')))

                    pu.visualize_nsaf(results, layer_cp, xcentergrid, ycentergrid,
                                      transform='log', title_suf=mean_col,
                                      save_path=os.path.join(fig_path, fname.replace('.arr', '.png')))


        else:
            # todo 3d
            pass
    print("\n=== Complete ===")
    return all_results

"""
Final helper functions for NSAF Utils
These complete the implementation with field generation and boundary functions
"""

# Add these to nsaf_utils.py

def _generate_stochastic_field(grid_coords, mean_field, sd_field,
                                bearing, anisotropy, corrlen,
                                zones, n_realizations, transform,
                                config_dict, ny, nx, iids=None, area=None,
                                active=None):
    """
    Generate stochastic field using pypestutils FIELDGEN2D_SVA.
    """
    try:
        from pypestutils.pestutilslib import PestUtilsLib
    except Exception as e:
        raise Exception(f"Error importing pypestutils: {e}")

    print("  Generating stochastic field using FIELDGEN2D_SVA...")

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

    # Set defaults
    if active is None:
        active = np.ones(n_points, dtype=int)
    if area is None:
        area = np.ones(n_points)
    else:
        if area.ndim > 1:
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

        field = lib.fieldgen2d_sva(
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
        if field.shape == (n_points, n_realizations):
            # Output is (n_points, n_realizations) - transpose to (n_realizations, n_points)
            field = field.T  # Now shape is (n_realizations, n_points)
            # Reshape each realization to (ny, nx)
            field = field.reshape((n_realizations, ny, nx))

        elif field.shape == (n_realizations * n_points,):
            # Output is flattened - reshape directly
            field = field.reshape((n_realizations, ny, nx))

        elif field.shape == (n_realizations, n_points):
            # Output is already (n_realizations, n_points) - just reshape spatial dimensions
            field = field.reshape((n_realizations, ny, nx))

        else:
            raise ValueError(f"Unexpected fieldgen2d_sva output shape: {field.shape}")

        print(f"    Successfully generated {n_realizations} realizations with shape {field.shape}")

        # Clean up
        lib.free_all_memory()

        return field

    except Exception as e:
        print(f"    Warning: FIELDGEN2D_SVA failed: {e}")
        print(f"    Using fallback method...")

        # Fallback to simple field generation
        field = np.zeros((n_realizations, ny, nx))
        for real in range(n_realizations):
            current_iids = iids[:, real]
            field_values = _generate_simple_field(
                grid_coords, mean_flat, log_variance_flat,
                aa_values, aniso_values, bearing_values,
                transform, current_iids, ny, nx
            )
            field[real] = field_values.reshape(ny, nx)

        return field


def _generate_simple_field(grid_coords, mean_flat, variance_flat,
                           aa_values, aniso_values, bearing_values,
                           transform, iids, ny, nx):
    """
    Simple fallback field generation if FIELDGEN2D_SVA fails.
    """
    print("    Using improved simple correlated field generation fallback...")

    n_points = len(grid_coords)

    # Convert bearing from degrees to radians for calculations
    bearing_rad = np.radians(bearing_values)

    # Create spatially correlated field using moving average approach
    correlated_field = np.zeros(n_points)

    # For computational efficiency, use a subset of points for correlation
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
        cos_bear = np.cos(local_bearing)
        sin_bear = np.sin(local_bearing)

        # Rotate to principal axes (geological bearing: 0° = North, clockwise positive)
        dx_rot = dx * sin_bear + dy * cos_bear  # East component (major axis)
        dy_rot = -dx * cos_bear + dy * sin_bear  # North component (minor axis)

        # Scale by anisotropy
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
        remaining_indices = np.setdiff1d(np.arange(n_points), correlation_indices)

        for i in remaining_indices:
            # Find nearest correlation points
            distances_to_corr = np.sum((grid_coords[correlation_indices] - grid_coords[i]) ** 2, axis=1)
            nearest_corr_idx = correlation_indices[np.argmin(distances_to_corr)]
            correlated_field[i] = correlated_field[nearest_corr_idx]

    # Scale by local standard deviation
    stochastic_component = correlated_field * np.sqrt(variance_flat)

    # Combine with mean field
    if transform == 'log':
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
                'sd': mean_kh * 0.2,
                'major': major_length,
                'anisotropy': 4.0,
                'bearing': bearing,
                'zone': point_zone  # Optional: track which zone
            })

    cp_df = pd.DataFrame(cp_list)
    cp_df['transverse'] = cp_df['major'] / cp_df['anisotropy']

    # Generate field
    print("=== Testing tangential field generation ===")
    results = generate_single_layer(
        cp_df, modelgrid,
        zones=zones,
        tensor_interp='krig',
        #transform='log',
        boundary_smooth={'transition_cells': 5},
        boundary_enhance={'transition_cells': 5,
                          'peak_increase': 2}
    )

    print(f"Field statistics: mean={np.mean(results['fields'][0]):.3f}, std={np.std(results['fields'][0]):.3f}")
    print(f"Number of conceptual points: {len(cp_df)}")
    print("Tangential field generation complete!")

    save_dir = os.path.join('..','..','examples','tangential_nsaf')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if save_dir is not None:
        from pyemu import plot_utils as pu
        grid_info = _extract_grid_info(modelgrid)
        xcentergrid = grid_info['xcentergrid']
        ycentergrid = grid_info['ycentergrid']
        fname = f"tangential_example.arr"
        np.savetxt(os.path.join(save_dir, fname), results['fields'][0], fmt="%20.8E")
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