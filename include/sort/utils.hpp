///////////////////////////////////////////////////////////////////////////////
// utils.hpp: Utility functions for SORT tracking
// Bounding box conversions, IoU calculations, and matrix operations
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Dense>

namespace sort
{

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;
using Vector4f = Eigen::Vector4f;

/**
 * @brief Convert bounding box from [x1,y1,x2,y2] to [x,y,s,r] format
 * @param bbox Bounding box in format [x1, y1, x2, y2]
 * @return Vector in format [x, y, s, r] where:
 *         x,y = center coordinates
 *         s = scale (area)
 *         r = aspect ratio (width/height)
 */
Vector4f convertBboxToZ(const Vector4f& bbox);

/**
 * @brief Convert from [x,y,s,r] format back to [x1,y1,x2,y2] bounding box
 * @param x State vector in format [x, y, s, r, ...]
 * @param score Optional confidence score to append
 * @return Bounding box in format [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
 */
VectorXf convertXToBbox(const VectorXf& x, float score = -1.0f);

/**
 * @brief Compute IoU (Intersection over Union) between detection and tracker bboxes
 * @param bb_test Detection bounding boxes, each row is [x1, y1, x2, y2, score]
 * @param bb_gt Tracker bounding boxes, each row is [x1, y1, x2, y2]
 * @return IoU matrix where element (i,j) is IoU between detection i and tracker j
 */
MatrixXf computeIouBatch(const MatrixXf& bb_test, const MatrixXf& bb_gt);

/**
 * @brief Associate detections to trackers using IoU and Hungarian algorithm
 * @param detections Detection bounding boxes [N x 5] format [x1, y1, x2, y2, score]
 * @param trackers Tracker bounding boxes [M x 4] format [x1, y1, x2, y2]
 * @param iou_threshold Minimum IoU threshold for valid matches
 * @return Tuple of (matches, unmatched_detections, unmatched_trackers)
 *         matches: [K x 2] matrix where each row is [detection_idx, tracker_idx]
 *         unmatched_detections: vector of detection indices
 *         unmatched_trackers: vector of tracker indices
 */
std::tuple<MatrixXf, std::vector<int>, std::vector<int>>
  associateDetectionsToTrackers(const MatrixXf& detections,
      const MatrixXf& trackers,
      float iou_threshold = 0.3f);

} // namespace sort
