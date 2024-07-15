import cv2
import numpy as np
import glob

# 相机内参，SUN3D Portland_hotel
K = np.array([[570.342224, 0, 320.0],
              [0, 570.342224, 240.0],
              [0, 0, 1]])
principal_point = (320.0, 240.0)
focal_length = 570

def find_feature_matches(img_1, img_2, feature_type="ORB"):
    # 初始化
    if feature_type == "ORB":
        detector = cv2.ORB_create()
    else:
        print(f"Unsupported feature type: {feature_type}")
        return [], [], []

    keypoints_1, descriptors_1 = detector.detectAndCompute(img_1, None)
    keypoints_2, descriptors_2 = detector.detectAndCompute(img_2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors_1, descriptors_2)

    # 筛选匹配点对
    min_dist = min(match.distance for match in matches)
    max_dist = max(match.distance for match in matches)

    good_matches = [m for m in matches if m.distance <= max(2 * min_dist, 30.0)]
    return keypoints_1, keypoints_2, good_matches

def pose_estimation_2d2d(keypoints_1, keypoints_2, matches):
    points1 = np.array([keypoints_1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints_2[m.trainIdx].pt for m in matches])

    essential_matrix, _ = cv2.findEssentialMat(points1, points2, focal_length, principal_point)
    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, K)
    return R, t

def triangulation(keypoints_1, keypoints_2, matches, R, t):
    pts_1 = np.array([pixel2cam(keypoints_1[m.queryIdx].pt) for m in matches])
    pts_2 = np.array([pixel2cam(keypoints_2[m.trainIdx].pt) for m in matches])

    pts_4d_hom = cv2.triangulatePoints(np.hstack((np.eye(3), np.zeros((3, 1)))), np.hstack((R, t)), pts_1.T, pts_2.T)
    pts_4d = pts_4d_hom / pts_4d_hom[3]
    points = pts_4d[:3].T
    return points

def pixel2cam(point):
    return np.array([(point[0] - K[0, 2]) / K[0, 0], (point[1] - K[1, 2]) / K[1, 1]])

def compute_metrics(depth_map, depth_gt):
    valid_mask = depth_gt > 0
    depth_map = depth_map[valid_mask] / 1000.0
    depth_gt = depth_gt[valid_mask] / 1000.0

    abs_error = np.mean(np.abs(depth_map - depth_gt))
    rmse = np.sqrt(np.mean((depth_map - depth_gt) ** 2))

    depth_map[depth_map < 0] = 0
    rmse_log = np.sqrt(np.mean((np.log(depth_gt + 1.0) - np.log(depth_map + 1.0)) ** 2))

    return abs_error, rmse, rmse_log

def process_images(feature_type="ORB", img_dir="", depth_dir="", interval=20, num_pairs=10):
    img_files = sorted(glob.glob(img_dir + "/*.jpg"))  
    depth_files = sorted(glob.glob(depth_dir + "/*.png")) 

    total_metrics = {"abs_error": 0.0, "rmse": 0.0, "rmse_log": 0.0}
    total = 0

    for i in range(num_pairs):
        print(f"Processing pair {i}")
        img_1 = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
        img_2 = cv2.imread(img_files[i + interval], cv2.IMREAD_COLOR)
        depth_gt = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)

        keypoints_1, keypoints_2, matches = find_feature_matches(img_1, img_2, feature_type)
        print(f"Found {len(matches)} matches")

        if len(matches) < 10:
            continue

        R, t = pose_estimation_2d2d(keypoints_1, keypoints_2, matches)
        points = triangulation(keypoints_1, keypoints_2, matches, R, t)

        depth_map = np.zeros(depth_gt.shape, dtype=np.float64)
        for pt, kp in zip(points, keypoints_1):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                depth_map[y, x] = pt[2]

        abs_error, rmse, rmse_log = compute_metrics(depth_map, depth_gt)
        total_metrics["abs_error"] += abs_error
        total_metrics["rmse"] += rmse
        total_metrics["rmse_log"] += rmse_log
        total += 1

    if total > 0:
        total_metrics = {k: v / total for k, v in total_metrics.items()}

    print(f"Total: {total}")
    print(f"Abs error: {total_metrics['abs_error']}")
    print(f"RMSE: {total_metrics['rmse']}")
    print(f"RMSE log: {total_metrics['rmse_log']}")

process_images(feature_type="ORB", img_dir="/media/lei/Data/2024/third-year/SLAM/pj/dataset/Portland_hotel/image", depth_dir="/media/lei/Data/2024/third-year/SLAM/pj/dataset/Portland_hotel/depth")
