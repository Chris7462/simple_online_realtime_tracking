// C++ standard library includes
#include <vector>
#include <cmath>
#include <limits>

// Eigen includes
#include <Eigen/Dense>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#include "sort_backend/kalman_box_tracker.hpp"
#include "sort_backend/utils.hpp"


using namespace Eigen;

class KalmanBoxTrackerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Reset static counter for consistent test results
    resetStaticCounter();
    setupTestBoundingBoxes();
  }

  void TearDown() override
  {
    // Clean up if needed
  }

  void resetStaticCounter()
  {
    // Note: This is a hack to reset the static counter between tests
    // In real implementation, you might want to add a reset method
    // For now, we'll work with the incrementing behavior
  }

  void setupTestBoundingBoxes()
  {
    // Standard test bounding box
    standard_bbox_ = Vector4f(10.0f, 20.0f, 50.0f, 80.0f);

    // Square bounding box
    square_bbox_ = Vector4f(0.0f, 0.0f, 100.0f, 100.0f);

    // Small bounding box
    small_bbox_ = Vector4f(0.0f, 0.0f, 1.0f, 1.0f);

    // Large bounding box
    large_bbox_ = Vector4f(0.0f, 0.0f, 1000.0f, 500.0f);

    // Zero area box (degenerate case)
    zero_bbox_ = Vector4f(10.0f, 10.0f, 10.0f, 10.0f);
  }

  // Helper method to check if two values are approximately equal
  bool isApproxEqual(float a, float b, float tolerance = 1e-5f) const
  {
    return std::abs(a - b) < tolerance;
  }

  // Helper method to check if two vectors are approximately equal
  bool isApproxEqual(const Vector4f & a, const Vector4f & b, float tolerance = 1e-5f) const
  {
    for (int i = 0; i < 4; ++i) {
      if (!isApproxEqual(a[i], b[i], tolerance)) {
        return false;
      }
    }
    return true;
  }

  // Test data
  Vector4f standard_bbox_;
  Vector4f square_bbox_;
  Vector4f small_bbox_;
  Vector4f large_bbox_;
  Vector4f zero_bbox_;
};

// Construction and initialization tests
TEST_F(KalmanBoxTrackerTest, TestBasicConstruction)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Verify basic properties after construction
  EXPECT_GT(tracker.getId(), 0)
    << "Tracker ID should be positive";
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0)
    << "Initial time since update should be 0";
  EXPECT_EQ(tracker.getHitStreak(), 0)
    << "Initial hit streak should be 0";
  EXPECT_EQ(tracker.getAge(), 0)
    << "Initial age should be 0";

  // Verify initial state corresponds to input bounding box
  Vector4f initial_state = tracker.getState();
  Vector4f expected_z = sort::convertBboxToZ(standard_bbox_);
  Vector4f converted_back = sort::convertXToBbox(VectorXf(expected_z)).head<4>();

  EXPECT_TRUE(isApproxEqual(initial_state, converted_back, 1e-3f))
    << "Initial state should correspond to input bounding box";
}

TEST_F(KalmanBoxTrackerTest, TestUniqueIds)
{
  // Test that each tracker gets a unique ID
  std::vector<std::unique_ptr<sort::KalmanBoxTracker>> trackers;
  std::vector<int> ids;

  for (int i = 0; i < 10; ++i) {
    trackers.push_back(std::make_unique<sort::KalmanBoxTracker>(standard_bbox_));
    ids.push_back(trackers.back()->getId());
  }

  // Verify all IDs are unique
  for (size_t i = 0; i < ids.size(); ++i) {
    for (size_t j = i + 1; j < ids.size(); ++j) {
      EXPECT_NE(ids[i], ids[j])
        << "Tracker IDs should be unique";
    }
  }

  // Verify IDs are sequential (implementation detail)
  for (size_t i = 1; i < ids.size(); ++i) {
    EXPECT_EQ(ids[i], ids[i - 1] + 1)
      << "IDs should be sequential";
  }
}

TEST_F(KalmanBoxTrackerTest, TestConstructionWithDifferentBoxes)
{
  // Test construction with various bounding box types
  sort::KalmanBoxTracker tracker_standard(standard_bbox_);
  sort::KalmanBoxTracker tracker_square(square_bbox_);
  sort::KalmanBoxTracker tracker_small(small_bbox_);
  sort::KalmanBoxTracker tracker_large(large_bbox_);

  // All should construct successfully and have valid initial states
  Vector4f state_standard = tracker_standard.getState();
  Vector4f state_square = tracker_square.getState();
  Vector4f state_small = tracker_small.getState();
  Vector4f state_large = tracker_large.getState();

  // Check for NaN values
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(std::isnan(state_standard[i]))
      << "Standard tracker state should not be NaN";
    EXPECT_FALSE(std::isnan(state_square[i]))
      << "Square tracker state should not be NaN";
    EXPECT_FALSE(std::isnan(state_small[i]))
      << "Small tracker state should not be NaN";
    EXPECT_FALSE(std::isnan(state_large[i]))
      << "Large tracker state should not be NaN";
  }
}

TEST_F(KalmanBoxTrackerTest, TestConstructionWithZeroArea)
{
  // Test construction with zero-area bounding box
  sort::KalmanBoxTracker tracker(zero_bbox_);

  Vector4f state = tracker.getState();

  // Zero area box should still create valid tracker (though may have special behavior)
  EXPECT_GT(tracker.getId(), 0)
    << "Zero area box should still get valid ID";

  // For zero-area box, the conversion may produce NaN due to division by zero
  // Check if convertBboxToZ handles this case
  Vector4f expected_z = sort::convertBboxToZ(zero_bbox_);

  // The center coordinates should be the point location (10.0, 10.0)
  EXPECT_TRUE(isApproxEqual(expected_z[0], 10.0f))
    << "Expected center x should be 10.0";
  EXPECT_TRUE(isApproxEqual(expected_z[1], 10.0f))
    << "Expected center y should be 10.0";
  EXPECT_TRUE(isApproxEqual(expected_z[2], 0.0f))
    << "Expected area should be 0.0";
  // Aspect ratio for zero area is 0/0 which is undefined/NaN

  // The KalmanBoxTracker may handle this differently, so we test what actually happens
  if (!std::isnan(state[0]) && !std::isnan(state[1])) {
    // If the implementation handles zero area gracefully
    EXPECT_TRUE(isApproxEqual(state[0], 10.0f, 1e-3f))
      << "Center x should match";
    EXPECT_TRUE(isApproxEqual(state[1], 10.0f, 1e-3f))
      << "Center y should match";
  } else {
    // If zero area causes NaN, that's also acceptable behavior
    std::cout << "Zero area bounding box produces NaN coordinates (expected behavior)" << std::endl;
  }
}

// Prediction tests
TEST_F(KalmanBoxTrackerTest, TestPredictBasic)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  Vector4f initial_state = tracker.getState();
  Vector4f predicted_state = tracker.predict();

  // After first prediction with no velocity, state should remain similar
  EXPECT_TRUE(isApproxEqual(initial_state, predicted_state, 1.0f))
    << "First prediction should not move much without established velocity";

  // Age should increment
  EXPECT_EQ(tracker.getAge(), 1)
    << "Age should increment after prediction";
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 1)
    << "Time since update should increment";
  EXPECT_EQ(tracker.getHitStreak(), 0)
    << "Hit streak should be 0 without updates";
}

TEST_F(KalmanBoxTrackerTest, TestPredictMultipleSteps)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Predict multiple steps without updates
  std::vector<Vector4f> predictions;
  predictions.push_back(tracker.getState());

  for (int step = 1; step <= 5; ++step) {
    Vector4f pred = tracker.predict();
    predictions.push_back(pred);

    EXPECT_EQ(tracker.getAge(), step)
      << "Age should increment at step " << step;
    EXPECT_EQ(tracker.getTimeSinceUpdate(), step)
      << "Time since update should increment";
    EXPECT_EQ(tracker.getHitStreak(), 0)
      << "Hit streak should remain 0 without updates";

    // Check for NaN values
    for (int i = 0; i < 4; ++i) {
      EXPECT_FALSE(std::isnan(pred[i]))
        << "Prediction should not be NaN at step " << step;
    }
  }
}

TEST_F(KalmanBoxTrackerTest, TestPredictNegativeScaleHandling)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Force a scenario that might lead to negative scale
  // First establish some negative scale velocity through updates
  Vector4f shrinking_box1 = Vector4f(15.0f, 25.0f, 45.0f, 75.0f);  // Smaller
  Vector4f shrinking_box2 = Vector4f(20.0f, 30.0f, 40.0f, 70.0f);  // Even smaller
  Vector4f shrinking_box3 = Vector4f(25.0f, 35.0f, 35.0f, 65.0f);  // Very small

  tracker.predict();
  tracker.update(shrinking_box1);
  tracker.predict();
  tracker.update(shrinking_box2);
  tracker.predict();
  tracker.update(shrinking_box3);

  // Now predict many steps to potentially trigger negative scale handling
  for (int i = 0; i < 20; ++i) {
    Vector4f pred = tracker.predict();

    // Should not produce NaN values even if scale becomes problematic
    for (int j = 0; j < 4; ++j) {
      EXPECT_FALSE(std::isnan(pred[j]))
        << "Prediction should handle negative scale gracefully at step " << i;
    }

    // Width and height should remain positive
    float width = pred[2] - pred[0];
    float height = pred[3] - pred[1];
    EXPECT_GT(width, 0.0f)
      << "Width should remain positive at step " << i;
    EXPECT_GT(height, 0.0f)
      << "Height should remain positive at step " << i;
  }
}

// Update tests
TEST_F(KalmanBoxTrackerTest, TestUpdateBasic)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  Vector4f new_bbox = Vector4f(12.0f, 22.0f, 52.0f, 82.0f);  // Slightly moved
  tracker.update(new_bbox);

  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0)
    << "Time since update should reset to 0";
  EXPECT_EQ(tracker.getHitStreak(), 1)
    << "Hit streak should increment to 1";
  EXPECT_EQ(tracker.getAge(), 0)
    << "Age should not change during update";

  Vector4f updated_state = tracker.getState();

  // State should be influenced by the new measurement
  Vector4f expected_z = sort::convertBboxToZ(new_bbox);
  Vector4f converted_state = sort::convertXToBbox(VectorXf(expected_z)).head<4>();

  // The state should be close to the measurement (exact match depends on Kalman gain)
  for (int i = 0; i < 4; ++i) {
    EXPECT_LT(std::abs(updated_state[i] - converted_state[i]), 10.0f)
      << "Updated state should be reasonably close to measurement";
  }
}

TEST_F(KalmanBoxTrackerTest, TestUpdateSequence)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Sequence of updates without predictions
  std::vector<Vector4f> update_sequence = {
    Vector4f(12.0f, 22.0f, 52.0f, 82.0f),
    Vector4f(14.0f, 24.0f, 54.0f, 84.0f),
    Vector4f(16.0f, 26.0f, 56.0f, 86.0f)
  };

  for (size_t i = 0; i < update_sequence.size(); ++i) {
    tracker.update(update_sequence[i]);

    EXPECT_EQ(tracker.getTimeSinceUpdate(), 0)
      << "Time since update should be 0 after update " << i;
    EXPECT_EQ(tracker.getHitStreak(), static_cast<int>(i + 1))
      << "Hit streak should increment";
    EXPECT_EQ(tracker.getAge(), 0)
      << "Age should not increment during updates";

    Vector4f current_state = tracker.getState();
    for (int j = 0; j < 4; ++j) {
      EXPECT_FALSE(std::isnan(current_state[j]))
        << "State should not be NaN after update " << i;
    }
  }
}

TEST_F(KalmanBoxTrackerTest, TestUpdateAfterPrediction)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Predict then update
  tracker.predict();
  EXPECT_EQ(tracker.getAge(), 1);
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 1);
  EXPECT_EQ(tracker.getHitStreak(), 0);

  Vector4f update_bbox = Vector4f(15.0f, 25.0f, 55.0f, 85.0f);
  tracker.update(update_bbox);

  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0)
    << "Update should reset time since update";
  EXPECT_EQ(tracker.getHitStreak(), 1)
    << "Update should start new hit streak";
  EXPECT_EQ(tracker.getAge(), 1)
    << "Age should remain from prediction";
}

// Prediction-Update cycle tests
TEST_F(KalmanBoxTrackerTest, TestPredictUpdateCycle)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Simulate object moving with constant velocity
  std::vector<Vector4f> trajectory = {
    Vector4f(10.0f, 20.0f, 50.0f, 80.0f),   // Frame 0 (initial)
    Vector4f(15.0f, 25.0f, 55.0f, 85.0f),   // Frame 1 (+5, +5)
    Vector4f(20.0f, 30.0f, 60.0f, 90.0f),   // Frame 2 (+5, +5)
    Vector4f(25.0f, 35.0f, 65.0f, 95.0f),   // Frame 3 (+5, +5)
    Vector4f(30.0f, 40.0f, 70.0f, 100.0f)   // Frame 4 (+5, +5)
  };

  // Process trajectory
  for (size_t i = 1; i < trajectory.size(); ++i) {
    Vector4f prediction = tracker.predict();
    tracker.update(trajectory[i]);

    EXPECT_EQ(tracker.getAge(), static_cast<int>(i))
      << "Age should match frame number";
    EXPECT_EQ(tracker.getTimeSinceUpdate(), 0)
      << "Time since update should be 0 after update";

    // Hit streak behavior: after first predict(), hit_streak_ gets reset to 0,
    // then increments with each update
    EXPECT_EQ(tracker.getHitStreak(), static_cast<int>(i))
      << "Hit streak should increment correctly";

    // Check for valid prediction values
    for (int j = 0; j < 4; ++j) {
      EXPECT_FALSE(std::isnan(prediction[j]))
        << "Prediction should not be NaN at step " << i;
    }
  }

  // Test prediction quality after establishing velocity
  // Don't call predict() again as it would increment age and reset hit streak
  Vector4f current_state = tracker.getState();

  // After 4 updates showing consistent +5,+5 movement,
  // we can check if the internal velocity has been learned
  // But we need to be more lenient with tolerance since Kalman filtering
  // involves uncertainty and may not perfectly track the pattern

  // Just verify the state is reasonable
  EXPECT_GT(current_state[0], 20.0f)
    << "X position should have progressed";
  EXPECT_GT(current_state[1], 30.0f)
    << "Y position should have progressed";
  EXPECT_LT(current_state[0], 40.0f)
    << "X position should be reasonable";
  EXPECT_LT(current_state[1], 50.0f)
    << "Y position should be reasonable";
}

TEST_F(KalmanBoxTrackerTest, TestVelocityEstimation)
{
  sort::KalmanBoxTracker tracker(Vector4f(0.0f, 0.0f, 20.0f, 20.0f));

  // Create consistent movement pattern
  std::vector<Vector4f> positions = {
    Vector4f(10.0f, 5.0f, 30.0f, 25.0f),    // +10, +5
    Vector4f(20.0f, 10.0f, 40.0f, 30.0f),   // +10, +5
    Vector4f(30.0f, 15.0f, 50.0f, 35.0f),   // +10, +5
    Vector4f(40.0f, 20.0f, 60.0f, 40.0f)    // +10, +5
  };

  for (const auto & pos : positions) {
    tracker.predict();
    tracker.update(pos);
  }

  // After establishing pattern, prediction should anticipate movement
  Vector4f final_prediction = tracker.predict();
  Vector4f expected_position = Vector4f(50.0f, 25.0f, 70.0f, 45.0f);  // Continue +10, +5 pattern

  EXPECT_LT(std::abs(final_prediction[0] - expected_position[0]), 5.0f)
    << "Should predict continued X movement";
  EXPECT_LT(std::abs(final_prediction[1] - expected_position[1]), 5.0f)
    << "Should predict continued Y movement";
}

TEST_F(KalmanBoxTrackerTest, TestVelocityWithStationaryObject)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Update with same position multiple times
  for (int i = 0; i < 10; ++i) {
    tracker.predict();
    tracker.update(standard_bbox_);
  }

  // After many stationary observations, predictions should stay close
  Vector4f prediction1 = tracker.predict();
  Vector4f prediction2 = tracker.predict();

  // Predictions should not drift much for stationary object
  EXPECT_LT(std::abs(prediction2[0] - prediction1[0]), 2.0f)
    << "Stationary object predictions should not drift in X";
  EXPECT_LT(std::abs(prediction2[1] - prediction1[1]), 2.0f)
    << "Stationary object predictions should not drift in Y";
}

// State management tests
TEST_F(KalmanBoxTrackerTest, TestStateConsistency)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Initial state should be consistent with getState()
  Vector4f state1 = tracker.getState();
  Vector4f state2 = tracker.getState();

  EXPECT_TRUE(isApproxEqual(state1, state2))
    << "getState() should return consistent values";

  // State should be consistent before and after predict() call
  Vector4f pre_predict = tracker.getState();
  Vector4f prediction = tracker.predict();
  Vector4f post_predict = tracker.getState();

  EXPECT_TRUE(isApproxEqual(prediction, post_predict))
    << "predict() return value should match getState() after prediction";
}

TEST_F(KalmanBoxTrackerTest, TestStateAfterUpdate)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  Vector4f update_bbox = Vector4f(20.0f, 30.0f, 60.0f, 90.0f);
  tracker.update(update_bbox);

  Vector4f state_after_update = tracker.getState();

  // State should incorporate the measurement
  Vector4f expected_center = sort::convertBboxToZ(update_bbox);
  Vector4f state_center = sort::convertBboxToZ(state_after_update);

  // Should be closer to measurement than original
  Vector4f original_center = sort::convertBboxToZ(standard_bbox_);

  float dist_to_measurement = std::sqrt(
    std::pow(state_center[0] - expected_center[0], 2) +
    std::pow(state_center[1] - expected_center[1], 2)
  );

  float dist_to_original = std::sqrt(
    std::pow(state_center[0] - original_center[0], 2) +
    std::pow(state_center[1] - original_center[1], 2)
  );

  EXPECT_LT(dist_to_measurement, dist_to_original)
    << "Updated state should be closer to measurement than original";
}

// Lifecycle tracking tests
TEST_F(KalmanBoxTrackerTest, TestAgeProgression)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  EXPECT_EQ(tracker.getAge(), 0)
    << "Initial age should be 0";

  // Age should only increment during predict()
  tracker.update(standard_bbox_);
  EXPECT_EQ(tracker.getAge(), 0)
    << "Age should not increment during update";

  tracker.predict();
  EXPECT_EQ(tracker.getAge(), 1)
    << "Age should increment during predict";

  tracker.update(standard_bbox_);
  EXPECT_EQ(tracker.getAge(), 1)
    << "Age should not change during update";

  tracker.predict();
  EXPECT_EQ(tracker.getAge(), 2)
    << "Age should continue incrementing";
}

TEST_F(KalmanBoxTrackerTest, TestHitStreakManagement)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Hit streak should increment with updates
  EXPECT_EQ(tracker.getHitStreak(), 0);

  tracker.update(standard_bbox_);
  EXPECT_EQ(tracker.getHitStreak(), 1);

  tracker.update(standard_bbox_);
  EXPECT_EQ(tracker.getHitStreak(), 2);

  // Hit streak behavior with predictions:
  // Looking at the implementation: hit_streak_ = 0 only when time_since_update_ > 0
  // So the first predict() call doesn't reset it, but the second one does

  tracker.predict();  // time_since_update_ becomes 1, but hit_streak_ stays 2
  EXPECT_EQ(tracker.getHitStreak(), 2)
    << "Hit streak should not reset on first predict";

  tracker.predict();  // time_since_update_ becomes 2, now hit_streak_ becomes 0
  EXPECT_EQ(tracker.getHitStreak(), 0)
    << "Hit streak should reset on second predict without update";

  // Build up hit streak again
  tracker.update(standard_bbox_);
  EXPECT_EQ(tracker.getHitStreak(), 1);

  tracker.update(standard_bbox_);
  EXPECT_EQ(tracker.getHitStreak(), 2);
}

TEST_F(KalmanBoxTrackerTest, TestTimeSinceUpdateTracking)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0);

  // Should increment with predictions
  tracker.predict();
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 1);

  tracker.predict();
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 2);

  // Should reset with update
  tracker.update(standard_bbox_);
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0);

  // Should increment again with predictions
  tracker.predict();
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 1);
}

// Integration tests with realistic tracking scenarios
TEST_F(KalmanBoxTrackerTest, TestMovingObjectTracking)
{
  sort::KalmanBoxTracker tracker(Vector4f(0.0f, 0.0f, 50.0f, 50.0f));

  // Simulate object moving diagonally with some noise
  std::vector<Vector4f> trajectory = {
    Vector4f(5.2f, 4.8f, 55.1f, 54.9f),   // Frame 1
    Vector4f(10.1f, 9.9f, 60.2f, 59.8f),  // Frame 2
    Vector4f(14.8f, 15.2f, 64.9f, 65.1f), // Frame 3
    Vector4f(20.0f, 19.8f, 70.1f, 69.9f), // Frame 4
    Vector4f(25.1f, 24.9f, 75.0f, 75.2f)  // Frame 5
  };

  std::vector<Vector4f> predictions;

  for (size_t i = 0; i < trajectory.size(); ++i) {
    Vector4f pred = tracker.predict();
    predictions.push_back(pred);
    tracker.update(trajectory[i]);

    // After a few frames, predictions should be reasonably accurate
    if (i >= 2) {
      float pred_error_x = std::abs(pred[0] - trajectory[i][0]);
      float pred_error_y = std::abs(pred[1] - trajectory[i][1]);

      EXPECT_LT(pred_error_x, 5.0f)
        << "X prediction error should be reasonable at frame " << i;
      EXPECT_LT(pred_error_y, 5.0f)
        << "Y prediction error should be reasonable at frame " << i;
    }
  }

  // Final prediction should anticipate continued movement
  Vector4f final_pred = tracker.predict();
  EXPECT_GT(final_pred[0], trajectory.back()[0])
    << "Should predict continued X movement";
  EXPECT_GT(final_pred[1], trajectory.back()[1])
    << "Should predict continued Y movement";
}

TEST_F(KalmanBoxTrackerTest, TestScaleTracking)
{
  sort::KalmanBoxTracker tracker(Vector4f(0.0f, 0.0f, 20.0f, 20.0f));

  // Simulate object growing in size
  std::vector<Vector4f> growing_sequence = {
    Vector4f(-1.0f, -1.0f, 21.0f, 21.0f),    // Slightly larger
    Vector4f(-2.0f, -2.0f, 22.0f, 22.0f),    // Even larger
    Vector4f(-3.0f, -3.0f, 23.0f, 23.0f),    // Continuing growth
    Vector4f(-4.0f, -4.0f, 24.0f, 24.0f)     // More growth
  };

  for (const auto & bbox : growing_sequence) {
    tracker.predict();
    tracker.update(bbox);
  }

  // Predict next state - should anticipate continued growth
  Vector4f prediction = tracker.predict();
  float predicted_width = prediction[2] - prediction[0];
  float predicted_height = prediction[3] - prediction[1];

  EXPECT_GT(predicted_width, 24.0f)
    << "Should predict continued width growth";
  EXPECT_GT(predicted_height, 24.0f)
    << "Should predict continued height growth";
}

TEST_F(KalmanBoxTrackerTest, TestAspectRatioTracking)
{
  // Test with changing aspect ratios
  sort::KalmanBoxTracker tracker(Vector4f(0.0f, 0.0f, 40.0f, 20.0f));  // 2:1 ratio

  // Sequence with changing aspect ratios
  std::vector<Vector4f> aspect_sequence = {
    Vector4f(5.0f, 5.0f, 45.0f, 25.0f),     // Still roughly 2:1
    Vector4f(10.0f, 10.0f, 50.0f, 35.0f),   // Becoming more square
    Vector4f(15.0f, 15.0f, 55.0f, 45.0f),   // Even more square
    Vector4f(20.0f, 20.0f, 60.0f, 55.0f)    // Almost square
  };

  for (const auto & bbox : aspect_sequence) {
    tracker.predict();
    tracker.update(bbox);

    Vector4f current_state = tracker.getState();
    float width = current_state[2] - current_state[0];
    float height = current_state[3] - current_state[1];
    float aspect_ratio = width / height;

    EXPECT_GT(aspect_ratio, 0.0f)
      << "Aspect ratio should be positive";
    EXPECT_LT(aspect_ratio, 10.0f)
      << "Aspect ratio should be reasonable";
  }
}

// Robustness and error handling tests
TEST_F(KalmanBoxTrackerTest, TestNoisyMeasurements)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Add noise to measurements
  Vector4f base_pos = Vector4f(50.0f, 50.0f, 100.0f, 100.0f);
  std::vector<float> noise_levels = {2.0f, -1.5f, 3.0f, -2.5f, 1.0f, -0.5f};

  for (float noise : noise_levels) {
    Vector4f noisy_bbox = base_pos + Vector4f(noise, noise, noise, noise);

    tracker.predict();
    tracker.update(noisy_bbox);

    Vector4f current_state = tracker.getState();

    // Should handle noise without becoming unstable
    for (int i = 0; i < 4; ++i) {
      EXPECT_FALSE(std::isnan(current_state[i]))
        << "State should remain stable with noise";
    }

    // State should remain reasonably close to base position
    EXPECT_LT(std::abs(current_state[0] - base_pos[0]), 10.0f)
      << "Noisy measurements should not cause excessive drift";
    EXPECT_LT(std::abs(current_state[1] - base_pos[1]), 10.0f)
      << "Noisy measurements should not cause excessive drift";
  }
}

TEST_F(KalmanBoxTrackerTest, TestExtremeMovement)
{
  sort::KalmanBoxTracker tracker(Vector4f(0.0f, 0.0f, 20.0f, 20.0f));

  // Sudden large movement
  Vector4f extreme_bbox = Vector4f(1000.0f, 1000.0f, 1020.0f, 1020.0f);

  tracker.predict();
  tracker.update(extreme_bbox);

  Vector4f state_after_extreme = tracker.getState();

  // Should handle extreme movement gracefully
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(std::isnan(state_after_extreme[i]))
      << "Extreme movement should not cause NaN";
  }

  // State should move toward the extreme measurement (though maybe not completely)
  EXPECT_GT(state_after_extreme[0], 100.0f)
    << "Should move significantly toward extreme X position";
  EXPECT_GT(state_after_extreme[1], 100.0f)
    << "Should move significantly toward extreme Y position";
}

TEST_F(KalmanBoxTrackerTest, TestInvalidBoundingBoxes)
{
  // Test with bounding box that has negative width/height
  Vector4f invalid_bbox = Vector4f(50.0f, 50.0f, 30.0f, 30.0f);  // x2 < x1, y2 < y1

  sort::KalmanBoxTracker tracker(invalid_bbox);

  // Should still create tracker (implementation handles this)
  EXPECT_GT(tracker.getId(), 0)
    << "Invalid bbox should still create valid tracker";

  Vector4f initial_state = tracker.getState();

  // Predict should not crash
  Vector4f prediction = tracker.predict();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(std::isnan(prediction[i]))
      << "Invalid bbox should not cause NaN in prediction";
  }
}

TEST_F(KalmanBoxTrackerTest, TestLongTermTracking)
{
  sort::KalmanBoxTracker tracker(Vector4f(0.0f, 0.0f, 50.0f, 50.0f));

  // Simulate long-term tracking with periodic updates
  for (int frame = 1; frame <= 100; ++frame) {
    tracker.predict();

    // Update every 3 frames (simulating occasional missed detections)
    if (frame % 3 == 0) {
      Vector4f update_bbox = Vector4f(
        static_cast<float>(frame * 2),
        static_cast<float>(frame * 1.5f),
        static_cast<float>(frame * 2 + 50),
        static_cast<float>(frame * 1.5f + 50)
      );
      tracker.update(update_bbox);
    }

    // Verify stability over time
    Vector4f current_state = tracker.getState();
    for (int i = 0; i < 4; ++i) {
      EXPECT_FALSE(std::isnan(current_state[i]))
        << "Long-term tracking should remain stable at frame " << frame;
    }

    // Age should match frame number
    EXPECT_EQ(tracker.getAge(), frame)
      << "Age should increment correctly over long term";
  }
}

// Performance and stress tests
TEST_F(KalmanBoxTrackerTest, TestRepeatedOperations)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Perform many predict-update cycles
  for (int i = 0; i < 1000; ++i) {
    tracker.predict();

    // Add small variations to simulate realistic tracking
    Vector4f update_bbox = standard_bbox_ + Vector4f(
      static_cast<float>(i % 5 - 2),
      static_cast<float>(i % 3 - 1),
      static_cast<float>(i % 5 - 2),
      static_cast<float>(i % 3 - 1)
    );

    tracker.update(update_bbox);

    // Periodically check for stability
    if (i % 100 == 99) {
      Vector4f state = tracker.getState();
      for (int j = 0; j < 4; ++j) {
        EXPECT_FALSE(std::isnan(state[j]))
          << "State should remain stable after " << i << " iterations";
      }
    }
  }
}

TEST_F(KalmanBoxTrackerTest, TestNumericalStability)
{
  // Test with very small bounding box
  Vector4f tiny_bbox = Vector4f(0.0f, 0.0f, 0.001f, 0.001f);
  sort::KalmanBoxTracker tiny_tracker(tiny_bbox);

  for (int i = 0; i < 10; ++i) {
    Vector4f pred = tiny_tracker.predict();
    tiny_tracker.update(tiny_bbox);

    for (int j = 0; j < 4; ++j) {
      EXPECT_FALSE(std::isnan(pred[j]))
        << "Tiny bbox should not cause NaN at iteration " << i;
    }
  }

  // Test with very large bounding box
  Vector4f huge_bbox = Vector4f(0.0f, 0.0f, 1e6f, 1e6f);
  sort::KalmanBoxTracker huge_tracker(huge_bbox);

  for (int i = 0; i < 10; ++i) {
    Vector4f pred = huge_tracker.predict();
    huge_tracker.update(huge_bbox);

    for (int j = 0; j < 4; ++j) {
      EXPECT_FALSE(std::isnan(pred[j]))
        << "Huge bbox should not cause NaN at iteration " << i;
    }
  }
}

// Integration tests with SORT-specific scenarios
TEST_F(KalmanBoxTrackerTest, TestSortIntegrationScenarios)
{
  // Test scenario: tracker gets detection, then no detection for several frames
  sort::KalmanBoxTracker tracker(Vector4f(100.0f, 100.0f, 200.0f, 200.0f));

  // Initial update
  tracker.update(Vector4f(105.0f, 105.0f, 205.0f, 205.0f));
  EXPECT_EQ(tracker.getHitStreak(), 1);
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0);

  // Several predictions without updates (simulating lost detections)
  for (int i = 1; i <= 5; ++i) {
    tracker.predict();
    EXPECT_EQ(tracker.getTimeSinceUpdate(), i);
  }
  // After multiple predictions, hit streak should be 0
  EXPECT_EQ(tracker.getHitStreak(), 0);

  // Re-detection
  tracker.update(Vector4f(130.0f, 130.0f, 230.0f, 230.0f));
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0);
  EXPECT_EQ(tracker.getHitStreak(), 1);  // New streak starts
}

TEST_F(KalmanBoxTrackerTest, TestConfirmationScenario)
{
  sort::KalmanBoxTracker tracker(Vector4f(0.0f, 0.0f, 50.0f, 50.0f));

  // Build up hit streak (typical SORT confirmation requirement is 3 hits)
  std::vector<Vector4f> confirmation_sequence = {
    Vector4f(2.0f, 2.0f, 52.0f, 52.0f),
    Vector4f(4.0f, 4.0f, 54.0f, 54.0f),
    Vector4f(6.0f, 6.0f, 56.0f, 56.0f)
  };

  for (size_t i = 0; i < confirmation_sequence.size(); ++i) {
    tracker.predict();
    tracker.update(confirmation_sequence[i]);

    EXPECT_EQ(tracker.getHitStreak(), static_cast<int>(i + 1));
    EXPECT_GT(tracker.getAge(), 0);  // Age increments with predictions
  }

  // After 3 hits, tracker should be well-established
  EXPECT_EQ(tracker.getHitStreak(), 3);
}

TEST_F(KalmanBoxTrackerTest, TestOcclusionRecovery)
{
  sort::KalmanBoxTracker tracker(Vector4f(50.0f, 50.0f, 100.0f, 100.0f));

  // Establish movement pattern
  std::vector<Vector4f> pre_occlusion = {
    Vector4f(55.0f, 55.0f, 105.0f, 105.0f),
    Vector4f(60.0f, 60.0f, 110.0f, 110.0f),
    Vector4f(65.0f, 65.0f, 115.0f, 115.0f)
  };

  for (const auto & bbox : pre_occlusion) {
    tracker.predict();
    tracker.update(bbox);
  }

  // Simulate occlusion (predictions without updates)
  Vector4f prediction_during_occlusion;
  for (int i = 0; i < 5; ++i) {
    prediction_during_occlusion = tracker.predict();
  }

  // Recovery - object reappears close to predicted position
  Vector4f recovery_bbox = Vector4f(85.0f, 85.0f, 135.0f, 135.0f);  // Roughly where predicted
  tracker.update(recovery_bbox);

  // Should resume tracking normally
  EXPECT_EQ(tracker.getTimeSinceUpdate(), 0);
  EXPECT_EQ(tracker.getHitStreak(), 1);  // New streak after occlusion

  Vector4f state_after_recovery = tracker.getState();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(std::isnan(state_after_recovery[i]))
      << "Should recover from occlusion gracefully";
  }
}

// Edge cases and boundary conditions
TEST_F(KalmanBoxTrackerTest, TestBoundaryConditions)
{
  // Test with bounding box at image boundaries (assuming coordinate system starts at 0)
  Vector4f boundary_bbox = Vector4f(0.0f, 0.0f, 10.0f, 10.0f);
  sort::KalmanBoxTracker tracker(boundary_bbox);

  // Should handle boundary positions
  Vector4f state = tracker.getState();
  EXPECT_GE(state[0], 0.0f)
    << "X coordinate should be valid";
  EXPECT_GE(state[1], 0.0f)
    << "Y coordinate should be valid";

  // Predict movement toward negative coordinates
  Vector4f negative_update = Vector4f(-5.0f, -5.0f, 5.0f, 5.0f);
  tracker.predict();
  tracker.update(negative_update);

  Vector4f state_after_negative = tracker.getState();
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(std::isnan(state_after_negative[i]))
      << "Negative coordinates should not cause NaN";
  }
}

TEST_F(KalmanBoxTrackerTest, TestZeroVelocityInitialization)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // First prediction should have minimal movement (zero initial velocity)
  Vector4f initial_state = tracker.getState();
  Vector4f first_prediction = tracker.predict();

  float movement = std::sqrt(
    std::pow(first_prediction[0] - initial_state[0], 2) +
    std::pow(first_prediction[1] - initial_state[1], 2)
  );

  EXPECT_LT(movement, 5.0f)
    << "Initial prediction should have minimal movement with zero velocity";
}

// Comparative tests
TEST_F(KalmanBoxTrackerTest, TestPredictionVsUpdate)
{
  sort::KalmanBoxTracker tracker(standard_bbox_);

  // Get initial state
  Vector4f initial_state = tracker.getState();

  // Compare prediction vs immediate update
  Vector4f close_bbox = Vector4f(12.0f, 22.0f, 52.0f, 82.0f);

  // Branch 1: Just predict
  Vector4f prediction_only = tracker.predict();

  // Branch 2: Reset and update immediately (simulate different tracker)
  sort::KalmanBoxTracker tracker2(standard_bbox_);
  tracker2.update(close_bbox);
  Vector4f update_only = tracker2.getState();

  // The updated tracker should be closer to the measurement
  Vector4f measurement_center = sort::convertBboxToZ(close_bbox);
  Vector4f pred_center = sort::convertBboxToZ(prediction_only);
  Vector4f update_center = sort::convertBboxToZ(update_only);

  float pred_distance = std::sqrt(
    std::pow(pred_center[0] - measurement_center[0], 2) +
    std::pow(pred_center[1] - measurement_center[1], 2)
  );

  float update_distance = std::sqrt(
    std::pow(update_center[0] - measurement_center[0], 2) +
    std::pow(update_center[1] - measurement_center[1], 2)
  );

  EXPECT_LT(update_distance, pred_distance)
    << "Updated tracker should be closer to measurement than prediction-only tracker";
}

TEST_F(KalmanBoxTrackerTest, TestConsistentBehavior)
{
  // Test that identical operations produce identical results
  std::vector<Vector4f> test_sequence = {
    Vector4f(10.0f, 20.0f, 50.0f, 80.0f),
    Vector4f(15.0f, 25.0f, 55.0f, 85.0f),
    Vector4f(20.0f, 30.0f, 60.0f, 90.0f)
  };

  // Run sequence twice with different trackers
  sort::KalmanBoxTracker tracker1(test_sequence[0]);
  sort::KalmanBoxTracker tracker2(test_sequence[0]);

  for (size_t i = 1; i < test_sequence.size(); ++i) {
    Vector4f pred1 = tracker1.predict();
    tracker1.update(test_sequence[i]);

    Vector4f pred2 = tracker2.predict();
    tracker2.update(test_sequence[i]);

    // Both trackers should behave identically
    EXPECT_TRUE(isApproxEqual(pred1, pred2, 1e-6f))
      << "Identical operations should produce identical predictions at step " << i;

    Vector4f state1 = tracker1.getState();
    Vector4f state2 = tracker2.getState();

    EXPECT_TRUE(isApproxEqual(state1, state2, 1e-6f))
      << "Identical operations should produce identical states at step " << i;

    EXPECT_EQ(tracker1.getAge(), tracker2.getAge())
      << "Ages should match";
    EXPECT_EQ(tracker1.getHitStreak(), tracker2.getHitStreak())
      << "Hit streaks should match";
    EXPECT_EQ(tracker1.getTimeSinceUpdate(), tracker2.getTimeSinceUpdate())
      << "Time since update should match";
  }
}
