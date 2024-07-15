#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp> 
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;

// 相机内参，TUM Freiburg2
// Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
// Point2d principal_point(325.1, 249.7);  // 相机主点
// int focal_length = 521;                 // 相机焦距
// 相机内参, SUN3D Portland_hotel
Mat K = (Mat_<double>(3, 3) << 570.342224, 0, 320.000000, 0, 570.342224, 240.000000, 0, 0, 1);
Point2d principal_point(320.0, 240.0);  // 相机主点
int focal_length = 570;                 // 相机焦距


void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches,
  const string &feature_type);

void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t);

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
);

// 作图用
inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// 像素坐标转相机归一化坐标
Point2f pixel2cam(const Point2d &p, const Mat &K);

struct Metrics {
  double abs_error;
  double rmse;
  double rmse_log;
};

// 计算 Abs、RMSE、RMSE log 指标
Metrics compute_metrics(const Mat &depth_map, const Mat &depth_gt);

// 计算单个文件的指标
Metrics get_single_file_metrics(String feature_type, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat depth_gt);

int main(int argc, char **argv) {
  String feature_type = "SIFT";
  String img_dir = "/media/lei/Data/2024/third-year/SLAM/pj/dataset/Portland_hotel/image";
  String depth_dir = "/media/lei/Data/2024/third-year/SLAM/pj/dataset/Portland_hotel/depth";
  int interval = 20;

  vector<String> img_files, depth_files;
  glob(img_dir, img_files);
  glob(depth_dir, depth_files);

  int total = 0;
  Metrics total_metrics = {0.0, 0.0, 0.0};
  for (int i = 0; i < 5000; i++) {
    cout << i << endl;
    Mat img_1 = imread(img_files[i], cv::IMREAD_COLOR);
    Mat img_2 = imread(img_files[i + interval], cv::IMREAD_COLOR);
    Mat depth_gt = imread(depth_files[i], cv::IMREAD_UNCHANGED);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches, feature_type);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;
    if (matches.size() < 10) {
      continue;
    }

    Metrics metrics = get_single_file_metrics(feature_type, keypoints_1, keypoints_2, matches, depth_gt);
    total_metrics.abs_error += metrics.abs_error;
    total_metrics.rmse += metrics.rmse;
    total_metrics.rmse_log += metrics.rmse_log;
    total++;
  }

  if (total > 0) {
    total_metrics.abs_error /= total;
    total_metrics.rmse /= total;
    total_metrics.rmse_log /= total;
  }

  cout << "Total: " << total << endl;
  cout << "Abs error: " << total_metrics.abs_error << endl;
  cout << "RMSE: " << total_metrics.rmse << endl;
  cout << "RMSE log: " << total_metrics.rmse_log << endl;

  return 0;
}

Metrics get_single_file_metrics(String feature_type, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<DMatch> matches, Mat depth_gt) {
  //-- 估计两张图像间运动
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //-- 三角化
  vector<Point3d> points;
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);

  //-- 验证三角化点与特征点的重投影关系
  Mat depth_map = Mat::zeros(depth_gt.size(), CV_64F);
  for (int i = 0; i < matches.size(); i++) {
    // 第一个图
    float depth1 = points[i].z;
    // cout << "depth: " << depth1 << endl;
    Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);

    int x = round(pt1_cam.x * K.at<double>(0, 0) + K.at<double>(0, 2));
    int y = round(pt1_cam.y * K.at<double>(1, 1) + K.at<double>(1, 2));
    depth_map.at<double>(y, x) = depth1;

    // 第二个图
    Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    float depth2 = pt2_trans.at<double>(2, 0);
  }

  Metrics metrics = compute_metrics(depth_map, depth_gt);
  return metrics;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches,
                          const string &feature_type) {
  //-- 初始化
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector;
  Ptr<DescriptorExtractor> descriptor;

  if (feature_type == "ORB") {
    detector = ORB::create();
    descriptor = ORB::create();
  } else if (feature_type == "SIFT") {
    detector = SIFT::create();
    descriptor = SIFT::create();
  } else if (feature_type == "SURF") {
    detector = cv::xfeatures2d::SURF::create();
    descriptor = cv::xfeatures2d::SURF::create();
  } else {
    cout << "Unsupported feature type: " << feature_type << endl;
    return;
  }

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");

  //-- 第一步:检测特征点位置
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //-- 第二步:计算描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //-- 第三步:匹配描述子，使用欧氏距离
  vector<DMatch> match;
  matcher->match(descriptors_1, descriptors_2, match);

  //-- 第四步:匹配点对筛选
  double min_dist = 10000, max_dist = 0;
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  // printf("-- Max dist : %f \n", max_dist);
  // printf("-- Min dist : %f \n", min_dist);

  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t) {
  
  //-- 把匹配点转换为vector<Point2f>的形式
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //-- 计算本质矩阵
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

  //-- 从本质矩阵中恢复旋转和平移信息.
  recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points) {
  Mat T1 = (Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
  Mat T2 = (Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );

  vector<Point2f> pts_1, pts_2;
  for (DMatch m:matches) {
    // 将像素坐标转换至相机坐标
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  // 转换成非齐次坐标
  for (int i = 0; i < pts_4d.cols; i++) {
    Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0);  // 归一化
    Point3d p(
      x.at<float>(0, 0),
      x.at<float>(1, 0),
      x.at<float>(2, 0)
    );
    points.push_back(p);
  }
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

Metrics compute_metrics(const Mat &depth_map, const Mat &depth_gt) {
  Metrics metrics = {0.0, 0.0, 0.0};
  int count = 0;

  for (int y = 0; y < depth_map.rows; y++) {
    for (int x = 0; x < depth_map.cols; x++) {
      double depth = depth_map.at<double>(y, x) / 1000.0;;
      double gt_depth = depth_gt.at<ushort>(y, x) / 1000.0;
      if (depth > 0) {
        metrics.abs_error += abs(depth - gt_depth);
        metrics.rmse += (depth - gt_depth) * (depth - gt_depth);
        metrics.rmse_log += (log(depth + 1.0) - log(gt_depth + 1.0)) * (log(depth + 1.0) - log(gt_depth + 1.0));
        count++;
      }
    }
  }

  if (count > 0) {
    metrics.abs_error /= count;
    metrics.rmse = sqrt(metrics.rmse / count);
    metrics.rmse_log = sqrt(metrics.rmse_log / count);
  }

  return metrics;
}