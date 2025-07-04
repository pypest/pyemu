import numpy as np
import pandas as pd
from scipy.linalg import logm, expm, solve
import matplotlib as mpl

mpl.use('Tkagg')
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(420)


def main(conceptual_points_file, grid_file, field_name=['field'], layer_mode=True):
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

    # Infer grid structure
    shape = infer_grid_shape(grid_coords)
    print(f"Grid shape: {shape} (nz, ny, nx)")

    for fn in field_name:
        cols = ['x', 'y', 'major', 'anisotropy', 'bearing']
        cols = cols + [_ for _ in conceptual_points.columns if f'_{fn}' in _]
        if layer_mode and 'layer' in conceptual_points.columns:
            # Process by layer
            cols = cols + ['layer']
            field = generate_field_by_layer(conceptual_points[cols], grid_coords, shape)
        else:
            # Process full 3D
            cols = cols + ['z']
            field = generate_field_3d(conceptual_points[cols], grid_coords, shape)

        # Save results
        save_results(field, shape, field_name)

        # Plot first layer
        plot_layer(field, shape, layer=0, field_name=field_name)

        print("\n=== Complete ===")
        print(f"Field statistics: mean={np.mean(field):.3f}, std={np.std(field):.3f}")


def generate_field_by_layer(conceptual_points, grid_coords, shape):
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
        unique_layers = sorted([int(x) for x in conceptual_points['layer'].unique()])
        print(f"Processing layers: {unique_layers}")

    for layer_idx in unique_layers:
        print(f"\nProcessing layer {layer_idx + 1}/{len(unique_layers)}")

        # Get grid points for this layer
        layer_grid = grid_3d[layer_idx].reshape(-1, 3)
        layer_coords_2d = layer_grid[:, :2]  # Just x, y

        # Get conceptual points for this layer
        if 'layer' in conceptual_points.columns:
            if layer_idx + 1 in conceptual_points['layer'].values:
                layer_cp = conceptual_points[conceptual_points['layer'] == layer_idx + 1]
            elif layer_idx in conceptual_points['layer'].values:
                layer_cp = conceptual_points[conceptual_points['layer'] == layer_idx]
            else:
                print(f"  No conceptual points for layer {layer_idx + 1}, using all")
                layer_cp = conceptual_points
        else:
            layer_cp = conceptual_points

        print(f"  Using {len(layer_cp)} conceptual points")

        # Generate field for this layer
        layer_field = generate_single_layer(layer_cp, layer_coords_2d)
        field_3d[layer_idx] = layer_field.reshape(ny, nx)

    return field_3d.flatten()


def generate_single_layer(conceptual_points, grid_coords_2d):
    """
    Generate field for a single 2D layer using tensor-based kriging.

    This function performs the core tensor interpolation workflow:
    1. Interpolate tensor field from conceptual points
    2. Interpolate mean and standard deviation fields  
    3. Generate spatially correlated noise
    4. Combine into final lognormal field

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Conceptual points with required columns: x, y, major, anisotropy, bearing
        Plus columns matching 'mean_*' and 'sd_*' patterns
    grid_coords_2d : np.ndarray
        2D grid coordinates, shape (n_points, 2)

    Returns
    -------
    np.ndarray
        Generated field values at grid points
    """
    cp_coords = conceptual_points[['x', 'y']].values

    # Extract the property values to interpolate
    cp_means = conceptual_points.filter(like='mean').values.flatten()

    # Handle variance/sd columns
    variance_cols = [col for col in conceptual_points.columns if 'variance_' in col.lower()]
    sd_cols = [col for col in conceptual_points.columns if 'sd_' in col.lower()]

    if variance_cols:
        cp_sd = np.sqrt(conceptual_points[variance_cols[0]].values)
    elif sd_cols:
        cp_sd = conceptual_points[sd_cols[0]].values
    else:
        raise ValueError("No variance or sd columns found")

    # Step 1: Compute tensor field ONCE
    print("  Interpolating correlation tensors...")
    cp_tensors = create_2d_tensors(conceptual_points)
    interp_tensors = interpolate_tensors_2d(cp_coords, cp_tensors, grid_coords_2d)

    # Step 2: Use tensor field for ALL properties
    print("  Interpolating properties with tensor field...")
    interp_means = tensor_aware_kriging(cp_coords, cp_means, grid_coords_2d, interp_tensors, transform='log')
    interp_sd = tensor_aware_kriging(cp_coords, cp_sd, grid_coords_2d, interp_tensors)

    # Step 3: Generate correlated noise
    print("  Generating correlated noise...")
    white_noise = np.random.randn(len(grid_coords_2d))
    correlated_noise = generate_correlated_noise_2d(grid_coords_2d, interp_tensors, white_noise)

    # Step 4: Combine into lognormal field
    field = 10 ** (np.log10(interp_means) + correlated_noise * interp_sd)

    return field


def tensor_aware_kriging(cp_coords, cp_values, grid_coords, interp_tensors,
                         variogram_model='exponential', range_param=3333,
                         sill=1.0, nugget=0.01, background_value=0.0,
                         max_search_radius=1e20, min_points=1, max_neighbors=4,
                         transform=None, min_value=1e-8):
    """
    Ordinary kriging using tensor-aware anisotropic distances.

    This function performs kriging where distances are calculated using
    Mahalanobis distance with locally varying anisotropy tensors. This
    avoids coordinate transformation issues while respecting local
    correlation structure.

    Parameters
    ----------
    cp_coords : np.ndarray
        Conceptual point coordinates, shape (n_points, 2)
    cp_values : np.ndarray  
        Values at conceptual points
    grid_coords : np.ndarray
        Grid coordinates for interpolation, shape (n_grid, 2)
    interp_tensors : np.ndarray
        Anisotropy tensors at grid points, shape (n_grid, 2, 2)
    variogram_model : str, default 'exponential'
        Variogram model ('exponential', 'gaussian', 'spherical')
    range_param : float, default 3333
        Variogram range parameter
    transform : str, optional
        'log' for log-transform of values (ensures positive results)
    max_neighbors : int, default 4
        Maximum number of nearest neighbors to use

    Returns
    -------
    np.ndarray
        Interpolated values at grid points
    """
    n_grid = len(grid_coords)
    interp_values = np.full(n_grid, background_value)

    # Variogram function
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
        # Get local tensor at this grid point
        local_tensor = interp_tensors[i]

        try:
            # Calculate anisotropic distances to all conceptual points
            tensor_inv = np.linalg.inv(local_tensor)
            aniso_distances = []

            for j in range(len(cp_coords)):
                dx = cp_coords[j] - grid_coords[i]
                # Mahalanobis distance: sqrt(dx.T @ T^(-1) @ dx)
                aniso_dist = np.sqrt(dx.T @ tensor_inv @ dx)
                aniso_distances.append(aniso_dist)

            aniso_distances = np.array(aniso_distances)

        except np.linalg.LinAlgError:
            # Fallback to Euclidean if tensor is singular
            aniso_distances = np.linalg.norm(cp_coords - grid_coords[i], axis=1)

        # Find closest neighbors using anisotropic distances
        sorted_indices = np.argsort(aniso_distances)
        n_candidates = min(max_neighbors or len(aniso_distances),
                           np.sum(aniso_distances <= max_search_radius))

        if n_candidates < min_points:
            continue

        # Get closest neighbors
        closest_indices = sorted_indices[:n_candidates]
        nearby_values = cp_values[closest_indices]
        nearby_coords = cp_coords[closest_indices]
        nearby_distances = aniso_distances[closest_indices]

        # Transform values if needed
        if transform == 'log':
            if min_value is None:
                positive_values = nearby_values[nearby_values > 0]
                min_value = np.min(positive_values) * 0.01
            nearby_values_transformed = np.log10(np.maximum(nearby_values, min_value))
        else:
            nearby_values_transformed = nearby_values.copy()

        # Build kriging system using anisotropic distances
        C = np.zeros((n_candidates + 1, n_candidates + 1))
        c = np.zeros(n_candidates + 1)

        # Fill covariance matrix
        for j in range(n_candidates):
            for k in range(n_candidates):
                # Distance between two conceptual points (recalculate with tensor)
                dx = nearby_coords[j] - nearby_coords[k]
                try:
                    h = np.sqrt(dx.T @ tensor_inv @ dx)  # Anisotropic distance
                except:
                    h = np.linalg.norm(dx)  # Fallback to Euclidean
                C[j, k] = variogram(h)

            # Unbiasedness constraint
            C[j, -1] = 1
            C[-1, j] = 1

            # Right-hand side
            c[j] = variogram(nearby_distances[j])

        c[-1] = 1

        # Solve kriging system
        try:
            weights = np.linalg.solve(C, c)[:-1]
            interp_values[i] = np.sum(weights * nearby_values_transformed)
        except:
            # Simple fallback
            weights = np.exp(-nearby_distances / range_param)
            interp_values[i] = np.sum(weights * nearby_values_transformed) / np.sum(weights)

    # Back-transform if needed
    if transform == 'log':
        interp_values = 10 ** interp_values

    return interp_values


def create_2d_tensors(conceptual_points):
    """
    Create 2x2 anisotropy tensors from geostatistical parameters.

    Converts bearing angle, major axis length, and anisotropy ratio into
    2x2 positive definite tensors suitable for spatial interpolation.

    Parameters
    ----------
    conceptual_points : pd.DataFrame
        Must contain columns: major, anisotropy, bearing

    Returns
    -------
    np.ndarray
        Array of 2x2 tensors, shape (n_points, 2, 2)
    """
    n = len(conceptual_points)
    tensors = np.zeros((n, 2, 2))

    for i in range(n):
        major = conceptual_points.iloc[i]['major']
        minor = conceptual_points.iloc[i]['major'] / conceptual_points.iloc[i]['anisotropy']
        bearing = conceptual_points.iloc[i]['bearing']

        theta = np.radians(bearing)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        # Diagonal matrix with squared correlation lengths
        S = np.diag([minor ** 2, major ** 2])
        tensors[i] = R @ S @ R.T

    return tensors


def interpolate_tensors_2d(cp_coords, cp_tensors, grid_coords):
    """
    Interpolate 2x2 tensors using log-Euclidean approach.

    This method ensures interpolated tensors remain positive definite
    by working in the logarithmic tensor space.

    Parameters
    ----------
    cp_coords : np.ndarray
        Conceptual point coordinates, shape (n_points, 2)
    cp_tensors : np.ndarray
        Tensors at conceptual points, shape (n_points, 2, 2)
    grid_coords : np.ndarray
        Grid coordinates, shape (n_grid, 2)

    Returns
    -------
    np.ndarray
        Interpolated tensors at grid points, shape (n_grid, 2, 2)
    """
    n_grid = len(grid_coords)
    interp_tensors = np.zeros((n_grid, 2, 2))

    # Convert to log space
    log_tensors = np.array([logm(t) for t in cp_tensors])

    # Interpolate each tensor component
    for i in range(2):
        for j in range(i, 2):
            values = log_tensors[:, i, j].real
            interp_values = ordinary_kriging(cp_coords, values, grid_coords)
            interp_tensors[:, i, j] = interp_values
            if i != j:
                interp_tensors[:, j, i] = interp_values

    # Convert back from log space
    for i in range(n_grid):
        interp_tensors[i] = expm(interp_tensors[i])

    return interp_tensors


def generate_correlated_noise_2d(grid_coords, tensors, white_noise,
                                 n_neighbors=50, anisotropy_strength=1.0):
    """
    Generate spatially correlated noise respecting tensor anisotropy.

    Creates correlated noise by weighting white noise values based on
    anisotropic distances defined by local tensors.

    Parameters
    ----------
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 2)
    tensors : np.ndarray
        Anisotropy tensors at each grid point, shape (n_points, 2, 2)
    white_noise : np.ndarray
        Input white noise, shape (n_points,)
    n_neighbors : int, default 50
        Number of neighbors to use for correlation
    anisotropy_strength : float, default 1.0
        Strength of anisotropic correlation

    Returns
    -------
    np.ndarray
        Spatially correlated noise
    """
    n_points = len(grid_coords)
    correlated_noise = np.zeros(n_points)

    for i in range(n_points):
        # Diagonalize tensor to get principal axes and correlation lengths
        evals, evecs = np.linalg.eigh(tensors[i])
        if evals[1] < evals[0]:
            evals, evecs = evals[::-1], evecs[:, ::-1]

        major_corr_length = np.sqrt(evals[1])
        minor_corr_length = np.sqrt(evals[0])

        # Create anisotropic transform
        scale_for_minor_axis = minor_corr_length / anisotropy_strength
        scale_for_major_axis = major_corr_length * anisotropy_strength

        transform = evecs @ np.diag([scale_for_minor_axis, scale_for_major_axis])

        # Transform all relative positions to anisotropic space
        all_dx = grid_coords - grid_coords[i]
        inv_transform = np.linalg.inv(transform)
        dx_transformed = all_dx @ inv_transform.T
        aniso_distances = np.linalg.norm(dx_transformed, axis=1)
        actual_distances = np.linalg.norm(all_dx, axis=1)

        # Select neighbors based on anisotropic distance
        neighbor_indices = np.argsort(aniso_distances)[:min(n_neighbors, n_points)]

        # Compute correlation weights using anisotropic distances
        weighted_sum = 0.0
        sum_weights = 0.0

        for j in neighbor_indices:
            weight = np.exp(-(actual_distances[j] / scale_for_major_axis) ** 2)
            weighted_sum += weight * white_noise[j]
            sum_weights += weight

        # Normalize
        if sum_weights > 1e-10:
            correlated_noise[i] = weighted_sum / sum_weights
        else:
            correlated_noise[i] = white_noise[i]

    return correlated_noise


def ordinary_kriging(cp_coords, cp_values, grid_coords,
                     variogram_model='exponential', range_param=10000,
                     sill=1.0, nugget=0.1, background_value=0.0,
                     max_search_radius=1e20, min_points=1,
                     transform=None, min_value=1e-8, max_neighbors=5):
    """
    Standard ordinary kriging with variogram models and search controls.

    Parameters
    ----------
    cp_coords : np.ndarray
        Conceptual point coordinates  
    cp_values : np.ndarray
        Values at conceptual points
    grid_coords : np.ndarray
        Grid coordinates for interpolation
    max_neighbors : int, default 5
        Maximum neighbors to use (controls computational cost)
    transform : str, optional
        'log' for log transformation

    Returns
    -------
    np.ndarray
        Interpolated values at grid points
    """
    n_grid = len(grid_coords)
    interp_values = np.full(n_grid, background_value)

    # Variogram function
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

        # Transform values if needed
        if transform == 'log':
            if min_value is None:
                positive_values = nearby_values[nearby_values > 0]
                min_value = np.min(positive_values) * 0.01
            nearby_values_transformed = np.log10(np.maximum(nearby_values, min_value))
        else:
            nearby_values_transformed = nearby_values.copy()

        # Build kriging system
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

        # Solve
        try:
            weights = solve(C, c)[:-1]
            interp_values[i] = np.sum(weights * nearby_values_transformed)
        except:
            weights = np.exp(-nearby_distances / range_param)
            interp_values[i] = np.sum(weights * nearby_values_transformed) / np.sum(weights)

    # Back-transform if needed
    if transform == 'log':
        interp_values = 10 ** interp_values

    return interp_values


def generate_field_3d(conceptual_points, grid_coords, shape):
    """Placeholder for full 3D field generation (not implemented)."""
    raise NotImplementedError("Use layer_mode=True for now")


def infer_grid_shape(grid_coords):
    """
    Infer grid dimensions from coordinate array.

    Parameters
    ----------
    grid_coords : np.ndarray
        Grid coordinates, shape (n_points, 3)

    Returns
    -------
    tuple
        Grid shape as (nz, ny, nx)
    """
    x_unique = np.unique(grid_coords[:, 0])
    y_unique = np.unique(grid_coords[:, 1])
    z_unique = np.unique(grid_coords[:, 2])

    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(grid_coords) // (nx * ny)

    return (nz, ny, nx)


def save_results(field, shape, field_name):
    """
    Save field as individual layer files.

    Parameters
    ----------
    field : np.ndarray
        Flattened field values
    shape : tuple
        Grid shape (nz, ny, nx)  
    field_name : str
        Base filename for output
    """
    nz, ny, nx = shape
    field_3d = field.reshape(shape)

    os.makedirs('output', exist_ok=True)

    for layer in range(nz):
        filename = f"output/{field_name}_layer_{layer + 1:02d}.txt"
        np.savetxt(filename, field_3d[layer], fmt='%.6f')

    print(f"Saved {nz} layer files to output/")


def plot_layer(field, shape, layer=0, field_name='field'):
    """
    Plot and save a single layer of the field.

    Parameters
    ----------
    field : np.ndarray
        Flattened field values
    shape : tuple
        Grid shape (nz, ny, nx)
    layer : int, default 0
        Layer index to plot
    field_name : str
        Base filename for plot
    """
    import matplotlib.colors as mcolors
    colors = ['red', 'orange', 'yellow', 'green', 'darkblue', 'lightblue', 'white']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=256)

    nz, ny, nx = shape
    field_3d = field.reshape(shape)

    plt.figure(figsize=(10, 8))
    plt.imshow(field_3d[layer], origin='lower', cmap=custom_cmap)
    plt.colorbar(label='Parameter Value')
    plt.title(f'Layer {layer + 1}')
    plt.xlabel('X')
    plt.ylabel('Y')

    os.makedirs('output', exist_ok=True)
    plt.savefig(f'output/{field_name}_layer_{layer + 1:02d}.png', dpi=150)
    plt.close()


# Example usage
if __name__ == "__main__":
    """
    Example usage of the tensor-based field generator.

    Required file formats:
    - conceptual_points.csv: CSV with columns x,y,major,anisotropy,bearing,mean_kh,sd_kh
    - grid.dat: Whitespace-delimited with columns x,y,z
    """
    main(
        conceptual_points_file='data/conceptual_points.csv',
        grid_file='data/grid.dat',
        field_name=['kh'],
        layer_mode=True
    )