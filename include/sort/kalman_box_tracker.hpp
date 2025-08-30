///////////////////////////////////////////////////////////////////////////////
// kalman_box_tracker.hpp: Bounding box tracker using Kalman filter
// Tracks individual objects using constant velocity motion model
// State vector: [x, y, s, r, dx, dy, ds] where (x,y) is center, s=area, r=aspect_ratio
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Dense>
#include <vector>

#include "sort/kalman_filter.hpp"


namespace sort
{

using Vector4f = Eigen::Vector4f;
using VectorXf = Eigen::VectorXf;

class KalmanBoxTracker
{
public:
  /**
   * @brief Initialize tracker with initial bounding box
   * @param bbox Initial bounding box in format [x1, y1, x2, y2]
   */
  explicit KalmanBoxTracker(const Vector4f& bbox);

  /**
   * @brief Update tracker with new detection
   * @param bbox New bounding box observation [x1, y1, x2, y2]
   */
  void update(const Vector4f& bbox);

  /**
   * @brief Predict next state and return predicted bounding box
   * @return Predicted bounding box [x1, y1, x2, y2]
   */
  Vector4f predict();

  /**
   * @brief Get current state as bounding box
   * @return Current bounding box estimate [x1, y1, x2, y2]
   */
  Vector4f getState() const;

  /**
   * @brief Get unique tracker ID
   * @return Tracker ID (starts from 1)
   */
  int getId() const { return id_; }

  /**
   * @brief Get number of frames since last update
   * @return Frames since last detection association
   */
  int getTimeSinceUpdate() const { return time_since_update_; }

  /**
   * @brief Get current hit streak (consecutive detections)
   * @return Number of consecutive frames with detections
   */
  int getHitStreak() const { return hit_streak_; }

  /**
   * @brief Get total age of tracker
   * @return Total number of frames tracker has existed
   */
  int getAge() const { return age_; }

private:
  static int count_;  // Global counter for unique IDs

  KalmanFilter kf_;   // Kalman filter instance

  // Tracker state
  int id_;                    // Unique tracker ID
  int time_since_update_;     // Frames since last update
  int hit_streak_;            // Current consecutive hit count
  int age_;                   // Total frames since creation

  /**
   * @brief Initialize Kalman filter matrices for constant velocity model
   */
  void initializeKalmanMatrices();
};

} // namespace sort
