# This function is designed for precompute mouse behavior and calcium data into a .h5 file.
# [input]   CaimAn output path
#           DeepLabCut output path
# [output]  A .h5 structure contains:
#             - dimension
#             - time
#             - spikes train
#             - F/delta F
#             - location
#             - movements
# [note]

def Precompute(caiman_output_path, dlc_output_path, fs_cal = 10, fs_beh = 30, movement_path=None, fs_move = 30, output_path = 'precomputed_data.h5'):
    import h5py 
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d

    status = 0
    
    with h5py.File(caiman_output_path, 'r') as calcium_data:
        calcium_traces = calcium_data['/estimates/C'][:]        # shape (T,N)
        spikes = calcium_data['/estimates/S'][:]                # shape (T,N)
        Dim = calcium_data['/dims'][:]                          # shape (W,H)

    # load calcium traces
    df = pd.read_hdf(dlc_output_path)

    x = df[('bodycenter', 'x')]
    y = df[('bodycenter', 'y')]
    likelihood = df[('bodycenter', 'likelihood')]

    # replace low likelihood data with nan
    x[likelihood < 0.9] = np.nan
    y[likelihood < 0.9] = np.nan

    # interpolate the trace
    x = x.interpolate()
    y = y.interpolate()

    position = np.vstack([x.values, y.values]).T  # shape (T,2)
    
    # snc time traces
    t_cal = spikes.shape[0]
    t_beh = position.shape[0]

    time = np.arange(t_cal)
    t_cal = np.arange(t_cal) / fs_cal
    t_beh = np.arange(t_beh) / fs_beh

    interp_x = interp1d(t_beh, position[:,0], bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(t_beh, position[:,1], bounds_error=False, fill_value="extrapolate")

    x_new = interp_x(t_cal)
    y_new = interp_y(t_cal)

    location = np.vstack([x_new, y_new]).T

    if movement_path != None:
        movement_data = h5py.File(movement_path,'r')
        movements = movement_data['movements'][:]
        t_move = movements.shape[0]
        t_move = np.arange(t_move) / fs_move
        movements = interp1d(t_move, movements, bounds_error=False, fill_value="extrapolate")
    else:
        movements = None
    
    
    with h5py.File(output_path, 'w') as f:
        f['dim'] = Dim
        f['time'] = time
        f['spikes'] = spikes
        f['caltrace'] = calcium_traces
        f['location'] = location
        f['movements'] = movements

    
    return status