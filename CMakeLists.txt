cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(sms_spam_classifier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions("-Wall" "-g")

add_executable(${PROJECT_NAME}
  src/main.cpp
)
# target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)
