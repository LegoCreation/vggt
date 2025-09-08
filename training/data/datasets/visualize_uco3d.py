import numpy as np
import cv2

def project_points_to_image(frame_num, K, extr, points, colors):
    # Convert points to homogeneous coordinates
    # depth_raw = np.load(f"/storage/group/dataset_mirrors/uco3d/uco3d_preprocessed_new/bags_and_luggage/backpack/7-59262-87968/depths/frame{frame_num+1:06d}.npy")
    points_3d_h = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)

    points_cam = (extr @ points_3d_h.T).T
    points_cam = points_cam[:, :3]

    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    colors = colors[valid]
    # Project to image plane
    uv = (K @ points_cam.T).T 
    uv = uv[:, :2] / uv[:, 2:3] 

    # Normalize homogeneous coordinates
    u = uv[:, 0].round().astype(np.uint32)
    v = uv[:, 1].round().astype(np.uint32)

    H, W = (288, 512)
    image = np.zeros((H, W, 3), dtype=colors.dtype)
    depth = np.full((H, W), float("inf"))

    # Z-buffer update
    for i in range(len(points_cam)):
        x, y, z = u[i], v[i], points_cam[i, 2]
        if 0 <= x < W and 0 <= y < H:
            if z < depth[y, x]:  # check if closer
                depth[y, x] = z
                image[y, x] = colors[i]
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"./debug_gaidosai/frame_{frame_num}.png", image)
    # depth_raw[np.isinf(depth_raw)] = 0
    # depth_img = (depth_raw - depth_raw.min()) / depth_raw.max()
    # cv2.imwrite(f"./debug_gaidosai/frame_{frame_num}_depth.png", depth_img * 255)

if __name__ == "__main__":
    path_to_sequence = "/storage/group/dataset_mirrors/uco3d/uco3d_preprocessed_new/bags_and_luggage/backpack/7-59262-87968/"

    cameras = np.load(f"{path_to_sequence}/camera_data.npz")
    loaded_points = np.load(f"{path_to_sequence}/pts3d.npz")

    points = loaded_points['pts_xyz'] / loaded_points['scale_factor']
    colors = loaded_points['pts_rgb']

    extrinsics = np.linalg.inv(cameras['camera_poses'])
    K = cameras['intrinsics']
    for i, extr in enumerate(extrinsics[:10]):
        project_points_to_image(i, K[i], extr, points, colors)
    # print(cameras['intrinsics'])
