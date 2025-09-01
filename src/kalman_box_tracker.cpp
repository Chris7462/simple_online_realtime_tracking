#include "sort/kalman_box_tracker.hpp"
#include "sort/utils.hpp"


namespace sort
{

int KalmanBoxTracker::count_ = 0;

KalmanBoxTracker::KalmanBoxTracker(const Vector4f & bbox)
: kf_(7, 4), // 7 state variables, 4 measurements
  id_(++count_),
  time_since_update_(0),
  hit_streak_(0),
  age_(0)
{

  initializeKalmanMatrices();

  // Set initial state from bounding box
  Vector4f z = convertBboxToZ(bbox);
  VectorXf initial_state = VectorXf::Zero(7);
  initial_state.head<4>() = z;
  kf_.setState(initial_state);
}

void KalmanBoxTracker::initializeKalmanMatrices()
{
  // State transition matrix F (constant velocity model)
  // State: [x, y, s, r, dx, dy, ds]
  kf_.F << 1, 0, 0, 0, 1, 0, 0,  // x = x + dx
    0, 1, 0, 0, 0, 1, 0,  // y = y + dy
    0, 0, 1, 0, 0, 0, 1,  // s = s + ds
    0, 0, 0, 1, 0, 0, 0,  // r = r (constant aspect ratio)
    0, 0, 0, 0, 1, 0, 0,  // dx = dx (constant velocity)
    0, 0, 0, 0, 0, 1, 0,  // dy = dy (constant velocity)
    0, 0, 0, 0, 0, 0, 1;  // ds = ds (constant scale velocity)

  // Observation matrix H (observe position, scale, aspect ratio)
  kf_.H << 1, 0, 0, 0, 0, 0, 0,  // observe x
    0, 1, 0, 0, 0, 0, 0,  // observe y
    0, 0, 1, 0, 0, 0, 0,  // observe s
    0, 0, 0, 1, 0, 0, 0;  // observe r

  // Measurement noise covariance R
  kf_.R = Eigen::MatrixXf::Identity(4, 4);
  kf_.R(2, 2) *= 10.0f;  // Higher uncertainty for scale
  kf_.R(3, 3) *= 10.0f;  // Higher uncertainty for aspect ratio

  // Process noise covariance Q
  kf_.Q = Eigen::MatrixXf::Identity(7, 7);
  kf_.Q(6, 6) *= 0.01f;  // Lower process noise for aspect ratio velocity
  kf_.Q.block<3, 3>(4, 4) *= 0.01f;  // Lower process noise for velocities

  // Initial state covariance P
  Eigen::MatrixXf & P = const_cast<Eigen::MatrixXf &>(kf_.getCovariance());
  P = Eigen::MatrixXf::Identity(7, 7);
  P.block<3, 3>(4, 4) *= 1000.0f;  // High uncertainty for initial velocities
  P *= 10.0f;  // Overall higher initial uncertainty
}

void KalmanBoxTracker::update(const Vector4f & bbox)
{
  time_since_update_ = 0;
  hit_streak_++;

  // Convert bbox to measurement vector
  Vector4f z = convertBboxToZ(bbox);
  kf_.update(z);
}

Vector4f KalmanBoxTracker::predict()
{
  // Handle negative scale prediction
  VectorXf state = kf_.getState();
  if ((state[6] + state[2]) <= 0) {
    // If predicted scale change would make scale negative, set velocity to zero
    state[6] = 0.0f;
    kf_.setState(state);
  }

  kf_.predict();
  age_++;

  if (time_since_update_ > 0) {
    hit_streak_ = 0;
  }
  time_since_update_++;

  // Return predicted bounding box directly
  return convertXToBbox(kf_.getState()).head<4>();
}

Vector4f KalmanBoxTracker::getState() const
{
  return convertXToBbox(kf_.getState()).head<4>();
}

int KalmanBoxTracker::getId() const
{
  return id_;
}

int KalmanBoxTracker::getTimeSinceUpdate() const
{
  return time_since_update_;
}

int KalmanBoxTracker::getHitStreak() const
{
  return hit_streak_;
}

int KalmanBoxTracker::getAge() const
{
  return age_;
}

} // namespace sort
