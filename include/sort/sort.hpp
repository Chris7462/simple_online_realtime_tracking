///////////////////////////////////////////////////////////////////////////////
// sort.hpp: Simple Online and Realtime Tracking (SORT) algorithm
// Multi-object tracker using Kalman filters and Hungarian algorithm
// Tracks objects across frames using bounding box detections
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>

#include "sort/kalman_box_tracker.hpp"


namespace sort
{

using MatrixXf = Eigen::MatrixXf;
using VectorXf = Eigen::VectorXf;

class Sort
{
public:
  /**
   * @brief Initialize SORT tracker with parameters
   * @param max_age Maximum number of frames to keep tracker without detections
   * @param min_hits Minimum hits before tracker is considered confirmed
   * @param iou_threshold Minimum IoU for detection-tracker association
   */
  explicit Sort(int max_age = 1, int min_hits = 3, float iou_threshold = 0.3f);

  /**
   * @brief Update tracker with new detections
   * @param detections Detection matrix where each row is [x1, y1, x2, y2, score]
   *                   Use empty matrix (0 rows, 5 cols) for frames without detections
   * @return Tracking results where each row is [x1, y1, x2, y2, track_id]
   *         Note: Number of returned tracks may differ from input detections
   */
  MatrixXf update(const MatrixXf& detections = MatrixXf::Zero(0, 5));

  /**
   * @brief Get current frame count
   * @return Number of frames processed
   */
  int getFrameCount() const { return frame_count_; }

  /**
   * @brief Get number of active trackers
   * @return Current number of trackers
   */
  size_t getTrackerCount() const { return trackers_.size(); }

  /**
   * @brief Reset tracker state (clear all trackers)
   */
  void reset();

private:
  // SORT parameters
  int max_age_;           // Maximum frames without detection before deletion
  int min_hits_;          // Minimum hits before tracker is confirmed
  float iou_threshold_;   // IoU threshold for association

  // Tracker state
  std::vector<std::unique_ptr<KalmanBoxTracker>> trackers_;
  int frame_count_;       // Current frame number

  /**
   * @brief Remove trackers that have invalid predictions (NaN values)
   * @param predicted_tracks Matrix of predicted tracker states
   * @return Indices of trackers to remove
   */
  std::vector<int> findInvalidTrackers(const MatrixXf& predicted_tracks);

  /**
   * @brief Build output matrix from confirmed trackers
   * @return Matrix where each row is [x1, y1, x2, y2, track_id]
   */
  MatrixXf buildOutputTracks();
};

} // namespace sort
