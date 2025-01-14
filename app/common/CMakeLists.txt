add_library(app.common STATIC cmdline.cpp)
target_link_libraries(app.common PRIVATE powerserve CLI11::CLI11)

target_include_directories(app.common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
