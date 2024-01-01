cmake_minimum_required(VERSION 3.19)

project(vulkan_raytracing)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 CONFIG REQUIRED)

file(GLOB_RECURSE PROJECT_SOURCES "code/*.cpp")
file(GLOB_RECURSE PROJECT_HEADERS "code/*.hpp")
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS})

# Lib
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

# Include
target_include_directories(${PROJECT_NAME} PRIVATE $ENV{VULKAN_SDK}/Include)

# Define
target_compile_definitions(${PROJECT_NAME} PRIVATE
    "SHADER_DIR=std::string{\"${PROJECT_SOURCE_DIR}/shaders/\"}"
)

# Set startup project
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
endif()
