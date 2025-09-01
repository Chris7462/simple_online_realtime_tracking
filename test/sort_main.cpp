#include "sort/sort.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;
using namespace sort;

struct Arguments
{
  std::string seq_path = "/home/yi-chen/ros2_ws/src/simple_online_realtime_tracking/data";
  std::string phase = "train";
  int max_age = 1;
  int min_hits = 3;
  float iou_threshold = 0.3f;
  bool display = false;
};

Arguments parseArgs(int argc, char * argv[])
{
  Arguments args;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--display") {
      args.display = true;
    } else if (arg == "--seq_path" && i + 1 < argc) {
      args.seq_path = argv[++i];
    } else if (arg == "--phase" && i + 1 < argc) {
      args.phase = argv[++i];
    } else if (arg == "--max_age" && i + 1 < argc) {
      args.max_age = std::stoi(argv[++i]);
    } else if (arg == "--min_hits" && i + 1 < argc) {
      args.min_hits = std::stoi(argv[++i]);
    } else if (arg == "--iou_threshold" && i + 1 < argc) {
      args.iou_threshold = std::stof(argv[++i]);
    }
  }

  return args;
}

// Load detection file in MOT format
std::vector<std::vector<float>> loadDetections(const std::string & filename)
{
  std::vector<std::vector<float>> detections;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      try {
        row.push_back(std::stof(cell));
      } catch (const std::exception & e) {
        // Skip invalid entries
        row.push_back(0.0f);
      }
    }

    if (row.size() >= 7) {  // Minimum required columns
      detections.push_back(row);
    }
  }

  return detections;
}

// Get detections for specific frame
Eigen::MatrixXf getFrameDetections(
  const std::vector<std::vector<float>> & all_detections, int frame)
{
  std::vector<std::vector<float>> frame_dets;

  for (const auto & det : all_detections) {
    if (std::abs(det[0] - frame) < 1e-6) {  // Frame number match (column 0)
      frame_dets.push_back(det);
    }
  }

  if (frame_dets.empty()) {
    return Eigen::MatrixXf::Zero(0, 5);
  }

  // Convert to Eigen matrix [x1, y1, x2, y2, score]
  Eigen::MatrixXf detections(frame_dets.size(), 5);

  for (size_t i = 0; i < frame_dets.size(); ++i) {
    const auto & det = frame_dets[i];
    // MOT format: frame, id, x, y, w, h, conf, ...
    // Convert [x, y, w, h] to [x1, y1, x2, y2]
    float x1 = det[2];
    float y1 = det[3];
    float w = det[4];
    float h = det[5];
    float x2 = x1 + w;
    float y2 = y1 + h;
    float conf = det[6];

    detections.row(i) << x1, y1, x2, y2, conf;
  }

  return detections;
}

// Find maximum frame number in detections
int getMaxFrame(const std::vector<std::vector<float>> & detections)
{
  int max_frame = 0;
  for (const auto & det : detections) {
    max_frame = std::max(max_frame, static_cast<int>(det[0]));
  }
  return max_frame;
}

// Write tracking results in MOT format
void writeResults(std::ofstream & out_file, int frame, const Eigen::MatrixXf & tracks)
{
  for (int i = 0; i < tracks.rows(); ++i) {
    float x1 = tracks(i, 0);
    float y1 = tracks(i, 1);
    float x2 = tracks(i, 2);
    float y2 = tracks(i, 3);
    int track_id = static_cast<int>(tracks(i, 4));

    // Convert back to [x, y, w, h] format
    float w = x2 - x1;
    float h = y2 - y1;

    // MOT output format: frame, id, x, y, w, h, conf, x, y, z
    out_file << frame << "," << track_id << ","
             << std::fixed << std::setprecision(2)
             << x1 << "," << y1 << "," << w << "," << h
             << ",1,-1,-1,-1\n";
  }
}

int main(int argc, char * argv[])
{
  Arguments args = parseArgs(argc, argv);

  std::cout << "SORT C++ Implementation\n";
  std::cout << "Parameters:\n";
  std::cout << "  seq_path: " << args.seq_path << "\n";
  std::cout << "  phase: " << args.phase << "\n";
  std::cout << "  max_age: " << args.max_age << "\n";
  std::cout << "  min_hits: " << args.min_hits << "\n";
  std::cout << "  iou_threshold: " << args.iou_threshold << "\n";
  std::cout << "  display: " << (args.display ? "true" : "false") << "\n\n";

  // Create output directory
  if (!fs::exists("output")) {
    fs::create_directories("output");
  }

  // Find all detection files
  std::string pattern_dir = args.seq_path + "/" + args.phase;
  std::vector<std::string> det_files;

  try {
    for (const auto & seq_dir : fs::directory_iterator(pattern_dir)) {
      if (seq_dir.is_directory()) {
        std::string det_file = seq_dir.path() / "det" / "det.txt";
        if (fs::exists(det_file)) {
          det_files.push_back(det_file);
        }
      }
    }
  } catch (const fs::filesystem_error & e) {
    std::cerr << "Error accessing directory: " << e.what() << "\n";
    return 1;
  }

  if (det_files.empty()) {
    std::cerr << "No detection files found in " << pattern_dir << "\n";
    return 1;
  }

  // Sort detection files for consistent processing order
  std::sort(det_files.begin(), det_files.end());

  double total_time = 0.0;
  int total_frames = 0;

  // Process each sequence
  for (const std::string & seq_det_file : det_files) {
    std::cout << "Processing " << seq_det_file << "\n";

    // Create SORT tracker instance
    Sort mot_tracker(args.max_age, args.min_hits, args.iou_threshold);

    // Load sequence detections
    std::vector<std::vector<float>> seq_dets;
    try {
      seq_dets = loadDetections(seq_det_file);
    } catch (const std::exception & e) {
      std::cerr << "Error loading " << seq_det_file << ": " << e.what() << "\n";
      continue;
    }

    // Extract sequence name from path
    fs::path seq_path(seq_det_file);
    std::string seq_name = seq_path.parent_path().parent_path().filename().string();

    // Create output file
    std::string output_file = "output/" + seq_name + ".txt";
    std::ofstream out_file(output_file);

    if (!out_file.is_open()) {
      std::cerr << "Cannot create output file: " << output_file << "\n";
      continue;
    }

    // Get maximum frame number
    int max_frame = getMaxFrame(seq_dets);

    // Process each frame
    for (int frame = 1; frame <= max_frame; ++frame) {
      // Get detections for this frame
      Eigen::MatrixXf dets = getFrameDetections(seq_dets, frame);

      total_frames++;

      // Track timing
      auto start_time = std::chrono::high_resolution_clock::now();

      // Update tracker
      Eigen::MatrixXf tracks = mot_tracker.update(dets);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      total_time += duration.count() / 1000000.0;  // Convert to seconds

      // Write results
      writeResults(out_file, frame, tracks);
    }

    out_file.close();
    std::cout << "  Output written to: " << output_file << "\n";
  }

  // Print timing statistics
  std::cout << "\n=== Performance Statistics ===\n";
  std::cout << "Total Tracking took: " << std::fixed << std::setprecision(3)
            << total_time << " seconds for " << total_frames << " frames\n";

  if (total_frames > 0) {
    double fps = total_frames / total_time;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(1) << fps << "\n";
  }

  if (args.display) {
    std::cout << "\nNote: to get real runtime results run without the option: --display\n";
  }

  return 0;
}
