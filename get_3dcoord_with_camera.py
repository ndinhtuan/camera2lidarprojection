import pandas as pd
import numpy as np

path_to_marker_coord = "F:/project/projects/mapathon/calibration/platform_calib/Koordinaten_Ref_Netz_3D_Labor_Stand_01_2019_mm.txt"

cam_to_room_matrix1 = np.array([
        [-0.723144,0.667226,0.178528,14.861920],
        [-0.646748,-0.744849,0.164065,14.739432],
        [0.242445,0.003180,0.970160,1.061096],
        [0.000000,0.000000,0.000000,1.000000],
        ])

cam_to_room_matrix2 = np.array([
    [-0.582441,0.801411,0.136024,14.588307],
    [-0.778817,-0.598100,0.189002,15.540361],
    [0.232824,0.004145,0.972510,1.065083  ],
    [0.000000,0.000000,0.000000,1.000000  ],
    ])

cam_to_room_matrix3 = np.array([
    [-0.570845,0.810634,0.130417,13.658869],
    [-0.787733,-0.585516,0.191437,16.199232], 
    [0.231546,0.006547,0.972802,1.06680],
    [0.000000,0.000000,0.000000,1.00000],
    ])

p1 = np.array([-316.982750,  -6994.430560,  5643.933055, 1])
p2 = np.array([-7341.715202,  -5201.722662,  5605.110050, 1])
p22 = p1/1000# np.array([-7.341715202,  -5.201722662,  5.605110050, 1])

if __name__=="__main__":
    
    marker_coord = pd.read_csv(path_to_marker_coord)
    marker_id = marker_coord.iloc[:, 0].to_numpy()
    tmp = marker_coord.iloc[:, [1, 2, 3]].to_numpy() / 1000
    h, w = tmp.shape
    marker_coord = np.ones([h, w+1])
    marker_coord[:, :w] = tmp

    cam_to_room_matrix = np.linalg.inv(cam_to_room_matrix2)
    # cam_to_room_matrix = cam_to_room_matrix2
    print(cam_to_room_matrix)
    marker_coord_in_cam = marker_coord@cam_to_room_matrix.T[:,:3]
    marker_id = np.expand_dims(marker_id, axis=1)
    marker_coord_in_cam = np.hstack([marker_id, marker_coord_in_cam])
    df = pd.DataFrame(marker_coord_in_cam)
    convert_dict = {0: np.int16}
    df = df.astype(convert_dict)
    print(df.dtypes)
    df.to_csv("ref2.txt", index=False)
