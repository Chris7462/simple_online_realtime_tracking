cmake_minimum_required(VERSION 3.15)

project(sort_tracker
  VERSION 1.0.0
  DESCRIPTION "C++ implementation of SORT (Simple Online and Realtime Tracking) using Eigen"
  LANGUAGES CXX)

include(GNUInstallDirs)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(hungarian REQUIRED)  # Your Hungarian algorithm package

# Create library
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
add_library(${PROJECT_NAME}
  src/kalman_filter.cpp
  src/kalman_box_tracker.cpp
  src/utils.cpp
  src/sort.cpp
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
    hungarian::hungarian
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: -Wall -Wextra -Wpedantic>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Installation
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# install(EXPORT ${PROJECT_NAME}Targets
#   FILE ${PROJECT_NAME}Targets.cmake
#   NAMESPACE ${PROJECT_NAME}::
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
# )

# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#   cmake/Config.cmake.in
#   ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
#   INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
# )

# write_basic_package_version_file(
#   ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
#   VERSION ${PROJECT_VERSION}
#   COMPATIBILITY SameMajorVersion
# )

# install(FILES
#   ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
#   ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
# )

# # Testing
# option(BUILD_TESTS "Build unit tests" ON)
# if(BUILD_TESTS)
#   enable_testing()
#   find_package(GTest REQUIRED)

#   add_executable(test_sort_tracker
#     test/test_sort.cpp
#   )

#   target_link_libraries(test_sort_tracker
#     ${PROJECT_NAME}
#     GTest::gtest_main
#   )

#   gtest_discover_tests(test_sort_tracker)
# endif()

# Main executable for MOT dataset processing
add_executable(sort_main
    test/sort_main.cpp
)

target_link_libraries(sort_main
    ${PROJECT_NAME}
)

# Example executable (optional)
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
  add_executable(sort_example
    test/sort_example.cpp
  )

  target_link_libraries(sort_example
    ${PROJECT_NAME}
  )
endif()
