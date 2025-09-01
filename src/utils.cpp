#include <algorithm>
#include <vector>
#include <cmath>

#include "sort_backend/utils.hpp"
#include "hungarian/hungarian.hpp"


namespace sort
{

Vector4f convertBboxToZ(const Vector4f & bbox)
{
  float w = bbox[2] - bbox[0];
  float h = bbox[3] - bbox[1];
  float x = bbox[0] + w / 2.0f;
  float y = bbox[1] + h / 2.0f;
  float s = w * h;  // scale is area
  float r = w / h;  // aspect ratio

  return Vector4f(x, y, s, r);
}

VectorXf convertXToBbox(const VectorXf & x, float score)
{
  float w = std::sqrt(x[2] * x[3]);
  float h = x[2] / w;

  if (score < 0.0f) {
    // Return 4-element bbox
    VectorXf result(4);
    result << x[0] - w / 2.0f, x[1] - h / 2.0f, x[0] + w / 2.0f, x[1] + h / 2.0f;
    return result;
  } else {
    // Return 5-element bbox with score
    VectorXf result(5);
    result << x[0] - w / 2.0f, x[1] - h / 2.0f, x[0] + w / 2.0f, x[1] + h / 2.0f, score;
    return result;
  }
}

MatrixXf computeIouBatch(const MatrixXf & bb_test, const MatrixXf & bb_gt)
{
  int num_test = bb_test.rows();
  int num_gt = bb_gt.rows();
  MatrixXf iou_matrix(num_test, num_gt);

  for (int i = 0; i < num_test; ++i) {
    for (int j = 0; j < num_gt; ++j) {
      // Get bounding box coordinates
      float x1_test = bb_test(i, 0), y1_test = bb_test(i, 1);
      float x2_test = bb_test(i, 2), y2_test = bb_test(i, 3);
      float x1_gt = bb_gt(j, 0), y1_gt = bb_gt(j, 1);
      float x2_gt = bb_gt(j, 2), y2_gt = bb_gt(j, 3);

      // Compute intersection
      float xx1 = std::max(x1_test, x1_gt);
      float yy1 = std::max(y1_test, y1_gt);
      float xx2 = std::min(x2_test, x2_gt);
      float yy2 = std::min(y2_test, y2_gt);

      float w = std::max(0.0f, xx2 - xx1);
      float h = std::max(0.0f, yy2 - yy1);
      float intersection = w * h;

      // Compute union
      float area_test = (x2_test - x1_test) * (y2_test - y1_test);
      float area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt);
      float union_area = area_test + area_gt - intersection;

      // Compute IoU
      if (union_area > 0.0f) {
        iou_matrix(i, j) = intersection / union_area;
      } else {
        iou_matrix(i, j) = 0.0f;
      }
    }
  }

  return iou_matrix;
}

std::tuple<MatrixXf, std::vector<int>, std::vector<int>> associateDetectionsToTrackers(
  const MatrixXf & detections, const MatrixXf & trackers, float iou_threshold)
{
  // Handle case where there are no detections
  if (detections.rows() == 0) {
    // All trackers are unmatched
    std::vector<int> unmatched_trackers;
    for (int i = 0; i < trackers.rows(); ++i) {
      unmatched_trackers.push_back(i);
    }
    return {MatrixXf::Zero(0, 2), std::vector<int>(), unmatched_trackers};
  }

  if (trackers.rows() == 0) {
    // No trackers - all detections are unmatched
    std::vector<int> unmatched_dets;
    for (int i = 0; i < detections.rows(); ++i) {
      unmatched_dets.push_back(i);
    }
    return {MatrixXf::Zero(0, 2), unmatched_dets, std::vector<int>()};
  }

  // Compute IoU matrix
  MatrixXf iou_matrix = computeIouBatch(detections, trackers);

  // Convert to Hungarian algorithm format (double precision)
  MatrixXd iou_matrix_double = iou_matrix.cast<double>();

  // Solve assignment problem (maximize IoU)
  hungarian::Hungarian hungarian;
  VectorXi assignment;
  hungarian.solve(iou_matrix_double, assignment, false); // maximize IoU

  // Process assignment results
  std::vector<std::pair<int, int>> matched_pairs;
  std::vector<int> unmatched_detections;
  std::vector<int> unmatched_trackers;

  // Find matched pairs that meet IoU threshold
  for (int det_idx = 0; det_idx < assignment.size(); ++det_idx) {
    int trk_idx = assignment(det_idx);
    if (trk_idx >= 0) {
      float iou_value = iou_matrix(det_idx, trk_idx);
      if (iou_value >= iou_threshold) {
        matched_pairs.push_back({det_idx, trk_idx});
      } else {
        unmatched_detections.push_back(det_idx);
      }
    } else {
      unmatched_detections.push_back(det_idx);
    }
  }

  // Find unmatched trackers
  std::vector<bool> tracker_matched(trackers.rows(), false);
  for (const auto & pair : matched_pairs) {
    tracker_matched[pair.second] = true;
  }

  for (int trk_idx = 0; trk_idx < trackers.rows(); ++trk_idx) {
    if (!tracker_matched[trk_idx]) {
      unmatched_trackers.push_back(trk_idx);
    }
  }

  // Convert matches to matrix format
  MatrixXf matches(matched_pairs.size(), 2);
  for (size_t i = 0; i < matched_pairs.size(); ++i) {
    matches(i, 0) = matched_pairs[i].first;
    matches(i, 1) = matched_pairs[i].second;
  }

  return {matches, unmatched_detections, unmatched_trackers};
}

} // namespace sort
