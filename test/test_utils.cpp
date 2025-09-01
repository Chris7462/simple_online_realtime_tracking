// C++ standard library includes
#include <vector>
#include <cmath>
#include <limits>

// Eigen includes
#include <Eigen/Dense>

// Google Test includes
#include <gtest/gtest.h>

// Local includes
#include "sort_backend/utils.hpp"


using namespace Eigen;

class UtilsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Common test data setup
    setupTestBoundingBoxes();
    setupTestStates();
  }

  void TearDown() override
  {
    // Clean up if needed
  }

  void setupTestBoundingBoxes()
  {
    // Standard bounding box [x1, y1, x2, y2]
    standard_bbox_ = Vector4f(10.0f, 20.0f, 50.0f, 80.0f);

    // Square bounding box
    square_bbox_ = Vector4f(0.0f, 0.0f, 100.0f, 100.0f);

    // Unit bounding box
    unit_bbox_ = Vector4f(0.0f, 0.0f, 1.0f, 1.0f);

    // Wide aspect ratio box
    wide_bbox_ = Vector4f(0.0f, 0.0f, 200.0f, 50.0f);

    // Tall aspect ratio box
    tall_bbox_ = Vector4f(0.0f, 0.0f, 25.0f, 100.0f);

    // Zero area box
    zero_bbox_ = Vector4f(10.0f, 10.0f, 10.0f, 10.0f);

    // Negative coordinates box
    negative_bbox_ = Vector4f(-50.0f, -30.0f, -10.0f, 10.0f);
  }

  void setupTestStates()
  {
    // Create state vectors from actual bbox conversions to ensure consistency
    Vector4f z_standard = sort::convertBboxToZ(standard_bbox_);
    standard_state_ = VectorXf(7);
    standard_state_.head<4>() = z_standard;
    standard_state_.tail<3>() = VectorXf::Zero(3);  // Zero velocities

    // State corresponding to square box
    Vector4f z_square = sort::convertBboxToZ(square_bbox_);
    square_state_ = VectorXf(7);
    square_state_.head<4>() = z_square;
    square_state_.tail<3>() = VectorXf::Zero(3);  // Zero velocities
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
  Vector4f unit_bbox_;
  Vector4f wide_bbox_;
  Vector4f tall_bbox_;
  Vector4f zero_bbox_;
  Vector4f negative_bbox_;

  VectorXf standard_state_;
  VectorXf square_state_;
};

// Tests for convertBboxToZ function
TEST_F(UtilsTest, TestConvertBboxToZBasic)
{
  Vector4f result = sort::convertBboxToZ(standard_bbox_);

  // Expected: center=(30, 50), area=1600, aspect_ratio=0.667
  EXPECT_TRUE(isApproxEqual(result[0], 30.0f)) << "Center x coordinate incorrect";
  EXPECT_TRUE(isApproxEqual(result[1], 50.0f)) << "Center y coordinate incorrect";
  EXPECT_TRUE(isApproxEqual(result[2], 2400.0f)) << "Scale (area) incorrect";
  EXPECT_TRUE(isApproxEqual(result[3], 0.6667f, 1e-3f)) << "Aspect ratio incorrect";
}

TEST_F(UtilsTest, TestConvertBboxToZSquare)
{
  Vector4f result = sort::convertBboxToZ(square_bbox_);

  // Expected: center=(50, 50), area=10000, aspect_ratio=1.0
  EXPECT_TRUE(isApproxEqual(result[0], 50.0f)) << "Square center x incorrect";
  EXPECT_TRUE(isApproxEqual(result[1], 50.0f)) << "Square center y incorrect";
  EXPECT_TRUE(isApproxEqual(result[2], 10000.0f)) << "Square area incorrect";
  EXPECT_TRUE(isApproxEqual(result[3], 1.0f)) << "Square aspect ratio should be 1.0";
}

TEST_F(UtilsTest, TestConvertBboxToZUnit)
{
  Vector4f result = sort::convertBboxToZ(unit_bbox_);

  // Expected: center=(0.5, 0.5), area=1.0, aspect_ratio=1.0
  EXPECT_TRUE(isApproxEqual(result[0], 0.5f)) << "Unit bbox center x incorrect";
  EXPECT_TRUE(isApproxEqual(result[1], 0.5f)) << "Unit bbox center y incorrect";
  EXPECT_TRUE(isApproxEqual(result[2], 1.0f)) << "Unit bbox area incorrect";
  EXPECT_TRUE(isApproxEqual(result[3], 1.0f)) << "Unit bbox aspect ratio incorrect";
}

TEST_F(UtilsTest, TestConvertBboxToZAspectRatios)
{
  // Test wide box
  Vector4f wide_result = sort::convertBboxToZ(wide_bbox_);
  EXPECT_TRUE(isApproxEqual(wide_result[3], 4.0f)) << "Wide bbox aspect ratio incorrect";

  // Test tall box
  Vector4f tall_result = sort::convertBboxToZ(tall_bbox_);
  EXPECT_TRUE(isApproxEqual(tall_result[3], 0.25f)) << "Tall bbox aspect ratio incorrect";
}

TEST_F(UtilsTest, TestConvertBboxToZZeroArea)
{
  Vector4f result = sort::convertBboxToZ(zero_bbox_);

  EXPECT_TRUE(isApproxEqual(result[0], 10.0f)) << "Zero area bbox center x incorrect";
  EXPECT_TRUE(isApproxEqual(result[1], 10.0f)) << "Zero area bbox center y incorrect";
  EXPECT_TRUE(isApproxEqual(result[2], 0.0f)) << "Zero area bbox should have zero scale";
  // Aspect ratio for zero area box is undefined (could be NaN or 0/0)
}

TEST_F(UtilsTest, TestConvertBboxToZNegativeCoordinates)
{
  Vector4f result = sort::convertBboxToZ(negative_bbox_);

  // Expected: center=(-30, -10), width=40, height=40, area=1600, aspect_ratio=1.0
  EXPECT_TRUE(isApproxEqual(result[0], -30.0f)) << "Negative coords center x incorrect";
  EXPECT_TRUE(isApproxEqual(result[1], -10.0f)) << "Negative coords center y incorrect";
  EXPECT_TRUE(isApproxEqual(result[2], 1600.0f)) << "Negative coords area incorrect";
  EXPECT_TRUE(isApproxEqual(result[3], 1.0f)) << "Negative coords aspect ratio incorrect";
}

// Tests for convertXToBbox function
TEST_F(UtilsTest, TestConvertXToBboxBasic)
{
  VectorXf result = sort::convertXToBbox(standard_state_);

  EXPECT_EQ(result.size(), 4) << "Should return 4-element bbox without score";

  // Verify round-trip conversion
  Vector4f original_bbox = result.head<4>();
  Vector4f z = sort::convertBboxToZ(original_bbox);

  EXPECT_TRUE(isApproxEqual(z[0], standard_state_[0])) << "Round-trip x failed";
  EXPECT_TRUE(isApproxEqual(z[1], standard_state_[1])) << "Round-trip y failed";
  EXPECT_TRUE(isApproxEqual(z[2], standard_state_[2])) << "Round-trip s failed";
  EXPECT_TRUE(isApproxEqual(z[3], standard_state_[3])) << "Round-trip r failed";
}

TEST_F(UtilsTest, TestConvertXToBboxWithScore)
{
  float test_score = 0.85f;
  VectorXf result = sort::convertXToBbox(standard_state_, test_score);

  EXPECT_EQ(result.size(), 5) << "Should return 5-element bbox with score";
  EXPECT_TRUE(isApproxEqual(result[4], test_score)) << "Score not preserved";
}

TEST_F(UtilsTest, TestConvertXToBboxSquare)
{
  VectorXf result = sort::convertXToBbox(square_state_);

  // Should produce a square bounding box
  float width = result[2] - result[0];
  float height = result[3] - result[1];
  EXPECT_TRUE(isApproxEqual(width, height)) << "Square state should produce square bbox";
  EXPECT_TRUE(isApproxEqual(width, 100.0f)) << "Square width incorrect";
}

TEST_F(UtilsTest, TestRoundTripConversion)
{
  // Test round-trip conversion for multiple bounding boxes
  std::vector<Vector4f> test_bboxes = {
    standard_bbox_, square_bbox_, unit_bbox_, wide_bbox_, tall_bbox_
  };

  for (const auto & bbox : test_bboxes) {
    // Skip zero area box as it may have undefined behavior
    if (isApproxEqual((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 0.0f)) {
      continue;
    }

    Vector4f z = sort::convertBboxToZ(bbox);

    Vector4f final_bbox = sort::convertXToBbox(z);

    EXPECT_TRUE(isApproxEqual(bbox, final_bbox, 1e-4f))
      << "Round-trip conversion failed for bbox: "
      << bbox.transpose();
  }
}

// Tests for computeIouBatch function
TEST_F(UtilsTest, TestComputeIouBatchIdentical)
{
  // Test identical bounding boxes (should have IoU = 1.0)
  MatrixXf detections(1, 4);
  detections.row(0) = standard_bbox_;

  MatrixXf trackers(1, 4);
  trackers.row(0) = standard_bbox_;

  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers);

  EXPECT_EQ(iou_matrix.rows(), 1);
  EXPECT_EQ(iou_matrix.cols(), 1);
  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), 1.0f)) << "Identical boxes should have IoU = 1.0";
}

TEST_F(UtilsTest, TestComputeIouBatchNoOverlap)
{
  // Test non-overlapping bounding boxes (should have IoU = 0.0)
  MatrixXf detections(1, 4);
  detections.row(0) = Vector4f(0.0f, 0.0f, 10.0f, 10.0f);

  MatrixXf trackers(1, 4);
  trackers.row(0) = Vector4f(20.0f, 20.0f, 30.0f, 30.0f);

  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers);

  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), 0.0f))
    << "Non-overlapping boxes should have IoU = 0.0";
}

TEST_F(UtilsTest, TestComputeIouBatchPartialOverlap)
{
  // Test boxes with known partial overlap
  MatrixXf detections(1, 4);
  detections.row(0) = Vector4f(0.0f, 0.0f, 20.0f, 20.0f);  // Area = 400

  MatrixXf trackers(1, 4);
  trackers.row(0) = Vector4f(10.0f, 10.0f, 30.0f, 30.0f);  // Area = 400

  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers);

  // Intersection: 10x10 = 100, Union: 400 + 400 - 100 = 700, IoU = 100/700 ≈ 0.143
  float expected_iou = 100.0f / 700.0f;
  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), expected_iou, 1e-3f))
    << "Partial overlap IoU incorrect. Expected: " << expected_iou
    << ", Got: " << iou_matrix(0, 0);
}

TEST_F(UtilsTest, TestComputeIouBatchMultiple)
{
  // Test multiple detections vs multiple trackers
  MatrixXf detections(3, 4);
  detections.row(0) = Vector4f(0.0f, 0.0f, 10.0f, 10.0f);    // Box A
  detections.row(1) = Vector4f(15.0f, 15.0f, 25.0f, 25.0f);  // Box B
  detections.row(2) = Vector4f(5.0f, 5.0f, 15.0f, 15.0f);    // Box C (overlaps A)

  MatrixXf trackers(2, 4);
  trackers.row(0) = Vector4f(0.0f, 0.0f, 10.0f, 10.0f);      // Same as Box A
  trackers.row(1) = Vector4f(20.0f, 20.0f, 30.0f, 30.0f);    // Different position

  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers);

  EXPECT_EQ(iou_matrix.rows(), 3);
  EXPECT_EQ(iou_matrix.cols(), 2);

  // Box A vs Tracker 0 (identical) should have IoU = 1.0
  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), 1.0f)) << "Identical boxes IoU should be 1.0";

  // Box A vs Tracker 1 (no overlap) should have IoU = 0.0
  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 1), 0.0f)) << "Non-overlapping IoU should be 0.0";
}

TEST_F(UtilsTest, TestComputeIouBatchEmpty)
{
  // Test with empty detection matrix
  MatrixXf empty_detections(0, 4);
  MatrixXf trackers(2, 4);
  trackers.row(0) = standard_bbox_;
  trackers.row(1) = square_bbox_;

  MatrixXf iou_matrix = sort::computeIouBatch(empty_detections, trackers);

  EXPECT_EQ(iou_matrix.rows(), 0);
  EXPECT_EQ(iou_matrix.cols(), 2);

  // Test with empty tracker matrix
  MatrixXf detections(2, 4);
  detections.row(0) = standard_bbox_;
  detections.row(1) = square_bbox_;
  MatrixXf empty_trackers(0, 4);

  iou_matrix = sort::computeIouBatch(detections, empty_trackers);

  EXPECT_EQ(iou_matrix.rows(), 2);
  EXPECT_EQ(iou_matrix.cols(), 0);
}

TEST_F(UtilsTest, TestComputeIouBatchZeroArea)
{
  // Test with zero-area bounding boxes
  MatrixXf detections(1, 4);
  detections.row(0) = zero_bbox_;

  MatrixXf trackers(1, 4);
  trackers.row(0) = standard_bbox_;

  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers);

  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), 0.0f)) << "Zero area box should have IoU = 0.0";
}

// Tests for associateDetectionsToTrackers function
TEST_F(UtilsTest, TestAssociateDetectionsToTrackersBasic)
{
  // Create detections with scores
  MatrixXf detections(2, 5);
  detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;   // High confidence
  detections.row(1) << 15.0f, 15.0f, 25.0f, 25.0f, 0.7f; // Medium confidence

  // Create trackers
  MatrixXf trackers(2, 4);
  trackers.row(0) = Vector4f(0.0f, 0.0f, 10.0f, 10.0f);    // Matches detection 0
  trackers.row(1) = Vector4f(15.0f, 15.0f, 25.0f, 25.0f);  // Matches detection 1

  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(detections, trackers, 0.5f);

  EXPECT_EQ(matches.rows(), 2) << "Should have 2 matches";
  EXPECT_TRUE(unmatched_dets.empty()) << "Should have no unmatched detections";
  EXPECT_TRUE(unmatched_trks.empty()) << "Should have no unmatched trackers";

  // Verify match assignments
  EXPECT_EQ(matches(0, 0), 0) << "Detection 0 should match";
  EXPECT_EQ(matches(0, 1), 0) << "Detection 0 should match tracker 0";
  EXPECT_EQ(matches(1, 0), 1) << "Detection 1 should match";
  EXPECT_EQ(matches(1, 1), 1) << "Detection 1 should match tracker 1";
}

TEST_F(UtilsTest, TestAssociateDetectionsToTrackersNoDetections)
{
  // Test with no detections
  MatrixXf empty_detections(0, 5);
  MatrixXf trackers(2, 4);
  trackers.row(0) = standard_bbox_;
  trackers.row(1) = square_bbox_;

  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(empty_detections, trackers, 0.3f);

  EXPECT_EQ(matches.rows(), 0) << "Should have no matches";
  EXPECT_TRUE(unmatched_dets.empty()) << "Should have no unmatched detections";
  EXPECT_EQ(unmatched_trks.size(), 2) << "Should have 2 unmatched trackers";
  EXPECT_EQ(unmatched_trks[0], 0) << "Tracker 0 should be unmatched";
  EXPECT_EQ(unmatched_trks[1], 1) << "Tracker 1 should be unmatched";
}

TEST_F(UtilsTest, TestAssociateDetectionsToTrackersNoTrackers)
{
  // Test with no trackers
  MatrixXf detections(2, 5);
  detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;
  detections.row(1) << 15.0f, 15.0f, 25.0f, 25.0f, 0.7f;

  MatrixXf empty_trackers(0, 4);

  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(detections, empty_trackers, 0.3f);

  EXPECT_EQ(matches.rows(), 0) << "Should have no matches";
  EXPECT_EQ(unmatched_dets.size(), 2) << "Should have 2 unmatched detections";
  EXPECT_TRUE(unmatched_trks.empty()) << "Should have no unmatched trackers";
  EXPECT_EQ(unmatched_dets[0], 0) << "Detection 0 should be unmatched";
  EXPECT_EQ(unmatched_dets[1], 1) << "Detection 1 should be unmatched";
}

TEST_F(UtilsTest, TestAssociateDetectionsToTrackersLowIoU)
{
  // Test with IoU threshold sensitivity - boxes with calculated overlap
  MatrixXf detections(1, 5);
  detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;  // Area = 100

  MatrixXf trackers(1, 4);
  trackers.row(0) = Vector4f(5.0f, 5.0f, 15.0f, 15.0f);  // Area = 100, overlap = 25

  // Calculate expected IoU: intersection=25, union=100+100-25=175, IoU=25/175≈0.143
  float expected_iou = 25.0f / 175.0f;  // ≈ 0.143

  // Verify IoU calculation first
  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers.leftCols(4));
  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), expected_iou, 1e-3f))
    << "Expected IoU: " << expected_iou << ", Got: " << iou_matrix(0, 0);

  // With high threshold (above actual IoU), should not match
  auto [matches_high, unmatched_dets_high, unmatched_trks_high] =
    sort::associateDetectionsToTrackers(detections, trackers, 0.2f);

  EXPECT_EQ(matches_high.rows(), 0) << "High threshold should prevent match";
  EXPECT_EQ(unmatched_dets_high.size(), 1) << "Detection should be unmatched";
  EXPECT_EQ(unmatched_trks_high.size(), 1) << "Tracker should be unmatched";

  // With low threshold (below actual IoU), should match
  auto [matches_low, unmatched_dets_low, unmatched_trks_low] =
    sort::associateDetectionsToTrackers(detections, trackers, 0.1f);

  EXPECT_EQ(matches_low.rows(), 1) << "Low threshold should allow match";
  EXPECT_TRUE(unmatched_dets_low.empty()) << "No unmatched detections with low threshold";
  EXPECT_TRUE(unmatched_trks_low.empty()) << "No unmatched trackers with low threshold";
}

TEST_F(UtilsTest, TestAssociateDetectionsToTrackersComplexScenario)
{
  // Complex scenario: 3 detections, 2 trackers, mixed matches
  MatrixXf detections(3, 5);
  detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;    // Should match tracker 0
  detections.row(1) << 50.0f, 50.0f, 60.0f, 60.0f, 0.8f;  // No good match (new object)
  detections.row(2) << 15.0f, 15.0f, 25.0f, 25.0f, 0.7f;  // Should match tracker 1

  MatrixXf trackers(2, 4);
  trackers.row(0) = Vector4f(1.0f, 1.0f, 11.0f, 11.0f);    // Close to detection 0
  trackers.row(1) = Vector4f(16.0f, 16.0f, 26.0f, 26.0f);  // Close to detection 2

  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(detections, trackers, 0.3f);

  EXPECT_EQ(matches.rows(), 2) << "Should have 2 matches";
  EXPECT_EQ(unmatched_dets.size(), 1) << "Should have 1 unmatched detection";
  EXPECT_TRUE(unmatched_trks.empty()) << "Should have no unmatched trackers";

  // Detection 1 (index 1) should be unmatched
  EXPECT_EQ(unmatched_dets[0], 1) << "Detection 1 should be unmatched";
}

TEST_F(UtilsTest, TestAssociateDetectionsToTrackersEdgeCases)
{
  // Test with zero IoU threshold
  MatrixXf detections(1, 5);
  detections.row(0) << 0.0f, 0.0f, 1.0f, 1.0f, 0.9f;

  MatrixXf trackers(1, 4);
  trackers.row(0) = Vector4f(100.0f, 100.0f, 101.0f, 101.0f);  // No overlap

  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(detections, trackers, 0.0f);

  EXPECT_EQ(matches.rows(), 1) << "Zero threshold should match any assignment";

  // Test with very high IoU threshold
  auto [matches_high, unmatched_dets_high, unmatched_trks_high] =
    sort::associateDetectionsToTrackers(detections, trackers, 0.99f);

  EXPECT_EQ(matches_high.rows(), 0) << "High threshold should prevent match";
}

// Performance and stress tests
TEST_F(UtilsTest, TestComputeIouBatchPerformance)
{
  // Test with larger matrices to ensure reasonable performance
  const int num_detections = 50;
  const int num_trackers = 30;

  MatrixXf detections(num_detections, 4);
  MatrixXf trackers(num_trackers, 4);

  // Fill with random-ish bounding boxes
  for (int i = 0; i < num_detections; ++i) {
    float x1 = static_cast<float>(i * 10);
    float y1 = static_cast<float>(i * 5);
    detections.row(i) = Vector4f(x1, y1, x1 + 20.0f, y1 + 15.0f);
  }

  for (int i = 0; i < num_trackers; ++i) {
    float x1 = static_cast<float>(i * 12);
    float y1 = static_cast<float>(i * 7);
    trackers.row(i) = Vector4f(x1, y1, x1 + 18.0f, y1 + 12.0f);
  }

  // This should complete without throwing and in reasonable time
  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers);

  EXPECT_EQ(iou_matrix.rows(), num_detections);
  EXPECT_EQ(iou_matrix.cols(), num_trackers);

  // Verify all IoU values are in [0, 1] range
  for (int i = 0; i < iou_matrix.rows(); ++i) {
    for (int j = 0; j < iou_matrix.cols(); ++j) {
      float iou_val = iou_matrix(i, j);
      EXPECT_GE(iou_val, 0.0f) << "IoU should be >= 0";
      EXPECT_LE(iou_val, 1.0f) << "IoU should be <= 1";
    }
  }
}

TEST_F(UtilsTest, TestBboxConversionNumericalStability)
{
  // Test conversion with very small bounding boxes
  Vector4f tiny_bbox(1e-6f, 1e-6f, 2e-6f, 2e-6f);
  Vector4f z = sort::convertBboxToZ(tiny_bbox);

  EXPECT_FALSE(std::isnan(z[0])) << "Tiny bbox conversion should not produce NaN";
  EXPECT_FALSE(std::isnan(z[1])) << "Tiny bbox conversion should not produce NaN";
  EXPECT_FALSE(std::isnan(z[2])) << "Tiny bbox conversion should not produce NaN";
  EXPECT_FALSE(std::isnan(z[3])) << "Tiny bbox conversion should not produce NaN";

  // Test conversion with very large bounding boxes
  Vector4f large_bbox(0.0f, 0.0f, 1e6f, 1e6f);
  z = sort::convertBboxToZ(large_bbox);

  EXPECT_FALSE(std::isnan(z[0])) << "Large bbox conversion should not produce NaN";
  EXPECT_FALSE(std::isnan(z[1])) << "Large bbox conversion should not produce NaN";
  EXPECT_FALSE(std::isnan(z[2])) << "Large bbox conversion should not produce NaN";
  EXPECT_FALSE(std::isnan(z[3])) << "Large bbox conversion should not produce NaN";
}

TEST_F(UtilsTest, TestBboxConversionInvalidInputs)
{
  // Test with inverted coordinates (x2 < x1)
  Vector4f inverted_bbox(10.0f, 20.0f, 5.0f, 80.0f);
  Vector4f z = sort::convertBboxToZ(inverted_bbox);

  // Should handle gracefully (negative width)
  EXPECT_LT(z[2], 0.0f) << "Inverted bbox should produce negative area";

  // Test with inverted height
  Vector4f inverted_height_bbox(10.0f, 80.0f, 50.0f, 20.0f);
  z = sort::convertBboxToZ(inverted_height_bbox);

  EXPECT_LT(z[2], 0.0f) << "Inverted height should produce negative area";
}

TEST_F(UtilsTest, TestStateToBoxConversionEdgeCases)
{
  // Test with zero scale - this should produce NaN due to sqrt(0) and division by 0
  VectorXf zero_scale_state(7);
  zero_scale_state << 50.0f, 50.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f;

  VectorXf result = sort::convertXToBbox(zero_scale_state);

  // Zero scale will produce NaN because: w = sqrt(s * r) = sqrt(0 * 1) = 0, then h = s/w = 0/0 = NaN
  EXPECT_TRUE(std::isnan(result[1]) || std::isnan(result[3]))
    << "Zero scale should produce NaN in some coordinates";
  std::cout << "Zero scale produces NaN as expected (w=sqrt(0*r)=0, h=s/w=0/0)" << std::endl;

  // Test with negative scale
  VectorXf negative_scale_state(7);
  negative_scale_state << 50.0f, 50.0f, -100.0f, 1.0f, 0.0f, 0.0f, 0.0f;

  result = sort::convertXToBbox(negative_scale_state);

  // Negative scale will produce NaN because sqrt(negative) = NaN
  EXPECT_TRUE(std::isnan(result[0]) || std::isnan(result[1]) || std::isnan(result[2]) ||
    std::isnan(result[3]))
    << "Negative scale should produce NaN due to sqrt of negative number";
  std::cout << "Negative scale produces NaN as expected (sqrt of negative)" << std::endl;

  // Test with valid but very small scale
  VectorXf small_scale_state(7);
  small_scale_state << 50.0f, 50.0f, 1e-12f, 1.0f, 0.0f, 0.0f, 0.0f;

  result = sort::convertXToBbox(small_scale_state);

  // Very small scale should still produce valid (though tiny) bounding box
  EXPECT_FALSE(std::isnan(result[0])) << "Very small scale should not produce NaN";
  EXPECT_FALSE(std::isnan(result[1])) << "Very small scale should not produce NaN";
  EXPECT_FALSE(std::isnan(result[2])) << "Very small scale should not produce NaN";
  EXPECT_FALSE(std::isnan(result[3])) << "Very small scale should not produce NaN";
}

TEST_F(UtilsTest, TestAssociateDetectionsToTrackersAsymmetric)
{
  // Test asymmetric scenario: more detections than trackers
  MatrixXf detections(4, 5);
  detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;
  detections.row(1) << 20.0f, 20.0f, 30.0f, 30.0f, 0.8f;
  detections.row(2) << 40.0f, 40.0f, 50.0f, 50.0f, 0.7f;
  detections.row(3) << 60.0f, 60.0f, 70.0f, 70.0f, 0.6f;

  MatrixXf trackers(2, 4);
  trackers.row(0) = Vector4f(1.0f, 1.0f, 11.0f, 11.0f);    // Should match detection 0
  trackers.row(1) = Vector4f(21.0f, 21.0f, 31.0f, 31.0f);  // Should match detection 1

  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(detections, trackers, 0.3f);

  EXPECT_EQ(matches.rows(), 2) << "Should have 2 matches";
  EXPECT_EQ(unmatched_dets.size(), 2) << "Should have 2 unmatched detections";
  EXPECT_TRUE(unmatched_trks.empty()) << "Should have no unmatched trackers";

  // Test reverse scenario: more trackers than detections
  MatrixXf few_detections(2, 5);
  few_detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;
  few_detections.row(1) << 20.0f, 20.0f, 30.0f, 30.0f, 0.8f;

  MatrixXf many_trackers(4, 4);
  many_trackers.row(0) = Vector4f(1.0f, 1.0f, 11.0f, 11.0f);
  many_trackers.row(1) = Vector4f(21.0f, 21.0f, 31.0f, 31.0f);
  many_trackers.row(2) = Vector4f(100.0f, 100.0f, 110.0f, 110.0f);
  many_trackers.row(3) = Vector4f(200.0f, 200.0f, 210.0f, 210.0f);

  auto [matches2, unmatched_dets2, unmatched_trks2] =
    sort::associateDetectionsToTrackers(few_detections, many_trackers, 0.3f);

  EXPECT_EQ(matches2.rows(), 2) << "Should have 2 matches";
  EXPECT_TRUE(unmatched_dets2.empty()) << "Should have no unmatched detections";
  EXPECT_EQ(unmatched_trks2.size(), 2) << "Should have 2 unmatched trackers";
}

TEST_F(UtilsTest, TestAssociateDetectionsToTrackersConsistency)
{
  // Test that the same input produces consistent results
  MatrixXf detections(2, 5);
  detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;
  detections.row(1) << 20.0f, 20.0f, 30.0f, 30.0f, 0.8f;

  MatrixXf trackers(2, 4);
  trackers.row(0) = Vector4f(1.0f, 1.0f, 11.0f, 11.0f);
  trackers.row(1) = Vector4f(21.0f, 21.0f, 31.0f, 31.0f);

  // Run association multiple times
  for (int run = 0; run < 5; ++run) {
    auto [matches, unmatched_dets, unmatched_trks] =
      sort::associateDetectionsToTrackers(detections, trackers, 0.3f);

    EXPECT_EQ(matches.rows(), 2) << "Consistent results expected across runs";
    EXPECT_TRUE(unmatched_dets.empty()) << "Consistent unmatched detections expected";
    EXPECT_TRUE(unmatched_trks.empty()) << "Consistent unmatched trackers expected";
  }
}

// Test the mathematical properties of the conversion functions
TEST_F(UtilsTest, TestConversionMathematicalProperties)
{
  // Test that area is preserved in conversion
  std::vector<Vector4f> test_boxes = {
    Vector4f(0.0f, 0.0f, 10.0f, 20.0f),    // Area = 200
    Vector4f(5.0f, 5.0f, 15.0f, 25.0f),    // Area = 200
    Vector4f(-10.0f, -5.0f, 30.0f, 35.0f)  // Area = 1600
  };

  for (const auto & bbox : test_boxes) {
    float original_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
    if (original_area <= 0.0f) {
      continue;  // Skip invalid boxes
    }

    Vector4f z = sort::convertBboxToZ(bbox);
    EXPECT_TRUE(isApproxEqual(z[2], original_area, 1e-4f))
      << "Area not preserved in conversion for bbox: " << bbox.transpose();

    // Test aspect ratio preservation
    float original_aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]);
    EXPECT_TRUE(isApproxEqual(z[3], original_aspect, 1e-4f))
      << "Aspect ratio not preserved for bbox: " << bbox.transpose();
  }
}

// Test error handling and boundary conditions
TEST_F(UtilsTest, TestAssociationWithNaNValues)
{
  // Test detection with NaN values
  MatrixXf detections_with_nan(1, 5);
  detections_with_nan.row(0) << std::numeric_limits<float>::quiet_NaN(), 0.0f, 10.0f, 10.0f, 0.9f;

  MatrixXf trackers(1, 4);
  trackers.row(0) = standard_bbox_;

  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(detections_with_nan, trackers, 0.3f);

  // Should handle NaN gracefully (exact behavior depends on implementation)
  EXPECT_LE(matches.rows(), 1) << "Should not crash with NaN inputs";

  // Test tracker with NaN values
  MatrixXf detections(1, 5);
  detections.row(0) << 0.0f, 0.0f, 10.0f, 10.0f, 0.9f;

  MatrixXf trackers_with_nan(1, 4);
  trackers_with_nan.row(0) << std::numeric_limits<float>::quiet_NaN(), 0.0f, 10.0f, 10.0f;

  auto [matches2, unmatched_dets2, unmatched_trks2] =
    sort::associateDetectionsToTrackers(detections, trackers_with_nan, 0.3f);

  EXPECT_LE(matches2.rows(), 1) << "Should not crash with NaN tracker inputs";
}

// Test specific IoU calculation edge cases
TEST_F(UtilsTest, TestComputeIouBatchSpecificCases)
{
  // Test touching boxes (should have IoU = 0)
  MatrixXf detections(1, 4);
  detections.row(0) = Vector4f(0.0f, 0.0f, 10.0f, 10.0f);

  MatrixXf trackers(1, 4);
  trackers.row(0) = Vector4f(10.0f, 0.0f, 20.0f, 10.0f);  // Touching edge

  MatrixXf iou_matrix = sort::computeIouBatch(detections, trackers);
  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), 0.0f)) << "Touching boxes should have IoU = 0";

  // Test one box inside another
  trackers.row(0) = Vector4f(2.0f, 2.0f, 8.0f, 8.0f);  // Inside detection box
  iou_matrix = sort::computeIouBatch(detections, trackers);

  // IoU = area_smaller / area_larger = 36 / 100 = 0.36
  EXPECT_TRUE(isApproxEqual(iou_matrix(0, 0), 0.36f, 1e-3f))
    << "Contained box IoU incorrect. Got: " << iou_matrix(0, 0);
}

TEST_F(UtilsTest, TestConversionConsistencyWithMinimalState)
{
  // Test conversion with minimal state vector (4 elements)
  VectorXf minimal_state(4);
  minimal_state << 30.0f, 50.0f, 1600.0f, 0.667f;

  VectorXf bbox_4 = sort::convertXToBbox(minimal_state);
  VectorXf bbox_5 = sort::convertXToBbox(minimal_state, 0.9f);

  EXPECT_EQ(bbox_4.size(), 4) << "Should return 4-element bbox";
  EXPECT_EQ(bbox_5.size(), 5) << "Should return 5-element bbox with score";

  // First 4 elements should be identical
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(isApproxEqual(bbox_4[i], bbox_5[i]))
      << "Bbox coordinates should match regardless of score parameter";
  }
}

// Integration test combining multiple utility functions
TEST_F(UtilsTest, TestIntegratedWorkflow)
{
  // Simulate a typical workflow: detections -> association -> tracking

  // Create some detections
  MatrixXf detections(3, 5);
  detections.row(0) << 10.0f, 10.0f, 50.0f, 50.0f, 0.9f;   // Object 1
  detections.row(1) << 100.0f, 100.0f, 150.0f, 140.0f, 0.8f; // Object 2
  detections.row(2) << 200.0f, 200.0f, 220.0f, 240.0f, 0.7f; // Object 3

  // Convert detections to state format for tracking
  std::vector<Vector4f> detection_states;
  for (int i = 0; i < detections.rows(); ++i) {
    Vector4f bbox = detections.row(i).head<4>();
    Vector4f z = sort::convertBboxToZ(bbox);
    detection_states.push_back(z);
  }

  // Simulate predicted tracker positions (slightly offset)
  MatrixXf predicted_trackers(2, 4);
  predicted_trackers.row(0) = Vector4f(12.0f, 12.0f, 52.0f, 52.0f);   // Close to detection 0
  predicted_trackers.row(1) = Vector4f(102.0f, 102.0f, 152.0f, 142.0f); // Close to detection 1

  // Perform association
  auto [matches, unmatched_dets, unmatched_trks] =
    sort::associateDetectionsToTrackers(detections, predicted_trackers, 0.3f);

  // Verify expected associations
  EXPECT_EQ(matches.rows(), 2) << "Should match 2 detections to trackers";
  EXPECT_EQ(unmatched_dets.size(), 1) << "Should have 1 unmatched detection (new object)";
  EXPECT_TRUE(unmatched_trks.empty()) << "All trackers should be matched";

  // Verify unmatched detection is the third one
  EXPECT_EQ(unmatched_dets[0], 2) << "Detection 2 should be unmatched (new object)";

  // Test IoU calculations are reasonable
  MatrixXf iou_matrix = sort::computeIouBatch(detections, predicted_trackers);
  for (int i = 0; i < matches.rows(); ++i) {
    int det_idx = static_cast<int>(matches(i, 0));
    int trk_idx = static_cast<int>(matches(i, 1));
    float iou_val = iou_matrix(det_idx, trk_idx);
    EXPECT_GT(iou_val, 0.3f) << "Matched pairs should exceed IoU threshold";
  }
}
