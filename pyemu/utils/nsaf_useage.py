from pyemu.utils import nsaf_helpers
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

    data_dir = r'..\..\examples\Wairau\waq_arr'
    zone_files = [f'waq_arr.geoclass_layer{i+1}.arr' for i in range(0,13)]
    con_pts_file = 'conceptual_points.csv'
    save_dir = os.path.join(data_dir, f'{con_pts_file}_output')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    nsaf_helpers.generate_fields_from_files(data_dir, 'waq_arr', con_pts_file,
                                            zone_file=zone_files,
                                            field_name=['kh'],
                                            layer_mode=True,
                                            save_dir=save_dir,
                                            tensor_interp='idw')


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")