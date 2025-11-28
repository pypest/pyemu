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

    model_name = 'waq_arr'
    data_dir = r'..\..\examples\Wairau\waq_arr'
    # need to ensure proper name structure for zone files here
    zone_files = {}
    for i in range(0, 13):
        zone_files[i] = f'{model_name}.geoclass_layer{i+1}.arr'
    con_pts_file = 'conceptual_points.csv'

    save_dir = os.path.join(data_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    nsaf_helpers.generate_fields_from_files(data_dir, model_name, con_pts_file,
                                            zone_files=zone_files,
                                            iid_files='iids_layer{}.arr',
                                            field_name=['kh'],
                                            layer_mode=True,
                                            save_dir=save_dir,
                                            tensor_interp='idw',
                                            transform=['log'],
                                            boundary_smooth={'transition_cells': 2},
                                            boundary_enhance={'transition_cells': 2,
                                                              'peak_increase': 2.0}
                                            )


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.4f} seconds")