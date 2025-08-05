from pyemu.utils import tensor_utils
import os

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
    import time
    start_time = time.perf_counter()

    data_dir = r'..\..\examples\Hawkes_Bay'
    zone_files = [f'idomain.arr' for z in range(1)]
    con_pts_file = 'Bridgpa_skytem_sva_2.csv'
    save_path = os.path.join(data_dir, f'{con_pts_file}_output')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    tensor_utils.generate_fields_from_files(data_dir,
        conceptual_points_file=con_pts_file,
        grid_file='grid.dat',
        zone_file=zone_files,
        field_name=['kh'],
        layer_mode=True,
        save_path=save_path,
        tensor_interp='idw'
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")