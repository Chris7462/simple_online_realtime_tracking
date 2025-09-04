cmake_minimum_required(VERSION 3.16)

project(sort_backend
  VERSION 1.0.0
  DESCRIPTION "C++ implementation of Simple Online and Realtime Tracking (SORT) backend"
  LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(hungarian REQUIRED)

# Create library
add_library(${PROJECT_NAME} STATIC
  src/kalman_filter.cpp
  src/kalman_box_tracker.cpp
  src/utils.cpp
  src/sort_backend.cpp)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    Eigen3::Eigen
    hungarian::hungarian)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -Wall -Wextra -Wpedantic>
)

# Optionally install the library
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  #LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  #RUNTIME DESTINATION bin
  # LIBRARY and RUNTIME are unused for static libs but harmless
  INCLUDES DESTINATION include
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Export targets and dependencies
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 hungarian) # Build static library, so downstream packages need to find these

# Build example
if(BUILD_EXAMPLE)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

  add_executable(sort_main example/sort_main.cpp)

  target_link_libraries(sort_main
    PRIVATE
      ${PROJECT_NAME}
      Eigen3::Eigen
      ${OpenCV_LIBS})

  target_include_directories(sort_main
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${OpenCV_INCLUDE_DIRS})

  target_compile_features(sort_main PRIVATE cxx_std_17)

  target_compile_options(sort_main PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>: -Wall -Wextra -Wpedantic>
  )
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  set(test_sources
    test_utils
    test_kalman_filter
    test_kalman_box_tracker
    test_sort_backend)

  foreach(test_src ${test_sources})
    ament_add_gtest(${test_src} test/${test_src}.cpp)
    target_link_libraries(${test_src}
      ${PROJECT_NAME}
      Eigen3::Eigen)
  endforeach()
endif()

ament_package()
