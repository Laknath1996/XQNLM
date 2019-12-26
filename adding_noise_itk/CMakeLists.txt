
cmake_minimum_required(VERSION 2.8)

project(add_noise)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(add_noise add_noise.cxx )

target_link_libraries(add_noise ${ITK_LIBRARIES})