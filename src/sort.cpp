#include <algorithm>
#include <cmath>

#include "sort/sort.hpp"
#include "sort/utils.hpp"


namespace sort
{

Sort::Sort(int max_age, int min_hits, float iou_threshold)
: max_age_(max_age), min_hits_(min_hits), iou_threshold_(iou_threshold), frame_count_(0)
{
}

MatrixXf Sort::update(const MatrixXf & detections)
{
  frame_count_++;

  // Get predictions from existing trackers
  MatrixXf predicted_tracks(trackers_.size(), 5);
  predicted_tracks.setZero();

  for (size_t i = 0; i < trackers_.size(); ++i) {
    Vector4f pos = trackers_[i]->predict();
    predicted_tracks.row(i) << pos[0], pos[1], pos[2], pos[3], 0.0f;
  }

  // Remove trackers with invalid predictions (NaN values)
  std::vector<int> to_delete = findInvalidTrackers(predicted_tracks);

  // Remove invalid trackers (in reverse order to maintain indices)
  for (auto it = to_delete.rbegin(); it != to_delete.rend(); ++it) {
    trackers_.erase(trackers_.begin() + *it);
  }

  // Rebuild predicted_tracks matrix after removing invalid trackers
  predicted_tracks.resize(trackers_.size(), 5);
  for (size_t i = 0; i < trackers_.size(); ++i) {
    Vector4f pos = trackers_[i]->getState();
    predicted_tracks.row(i) << pos[0], pos[1], pos[2], pos[3], 0.0f;
  }

  // Associate detections to trackers
  auto [matches, unmatched_dets, unmatched_trks] =
    associateDetectionsToTrackers(detections, predicted_tracks.leftCols(4), iou_threshold_);

  // Update matched trackers with assigned detections
  for (int i = 0; i < matches.rows(); ++i) {
    int det_idx = static_cast<int>(matches(i, 0));
    int trk_idx = static_cast<int>(matches(i, 1));

    Vector4f bbox = detections.row(det_idx).head<4>();
    trackers_[trk_idx]->update(bbox);
  }

  // Create new trackers for unmatched detections
  for (int det_idx : unmatched_dets) {
    Vector4f bbox = detections.row(det_idx).head<4>();
    auto tracker = std::make_unique<KalmanBoxTracker>(bbox);
    trackers_.push_back(std::move(tracker));
  }

  // Build output from confirmed trackers and remove dead ones
  MatrixXf result = buildOutputTracks();

  // Remove dead trackers (in reverse order)
  for (int i = static_cast<int>(trackers_.size()) - 1; i >= 0; --i) {
    if (trackers_[i]->getTimeSinceUpdate() > max_age_) {
      trackers_.erase(trackers_.begin() + i);
    }
  }

  return result;
}

std::vector<int> Sort::findInvalidTrackers(const MatrixXf & predicted_tracks)
{
  std::vector<int> to_delete;

  for (int i = 0; i < predicted_tracks.rows(); ++i) {
    // Check for NaN values in predicted position
    Vector4f pos = predicted_tracks.row(i).head<4>();
    bool has_nan = false;
    for (int j = 0; j < 4; ++j) {
      if (std::isnan(pos[j])) {
        has_nan = true;
        break;
      }
    }

    if (has_nan) {
      to_delete.push_back(i);
    }
  }

  return to_delete;
}

MatrixXf Sort::buildOutputTracks()
{
  std::vector<VectorXf> output_tracks;

  //for (size_t i = 0; i < trackers_.size(); ++i) {
  for (int i = static_cast<int>(trackers_.size()) - 1; i >= 0; --i) {
    auto& tracker = trackers_[i];

    // Only return tracks that meet confirmation criteria
    bool is_confirmed = (tracker->getTimeSinceUpdate() < 1) &&
      (tracker->getHitStreak() >= min_hits_ || frame_count_ <= min_hits_);

    if (is_confirmed) {
      Vector4f bbox = tracker->getState();
      VectorXf track(5);
      track << bbox[0], bbox[1], bbox[2], bbox[3], static_cast<float>(tracker->getId());
      output_tracks.push_back(track);
    }
  }

  // Convert to matrix format
  if (output_tracks.empty()) {
    return MatrixXf::Zero(0, 5);
  }

  MatrixXf result(output_tracks.size(), 5);
  for (size_t i = 0; i < output_tracks.size(); ++i) {
    result.row(i) = output_tracks[i].transpose();
  }

  return result;
}

void Sort::reset()
{
  trackers_.clear();
  frame_count_ = 0;
}

} // namespace sort
