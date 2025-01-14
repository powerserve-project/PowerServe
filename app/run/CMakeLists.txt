add_executable(run run.cpp)
target_link_libraries(run PRIVATE powerserve app.common)

powerserve_add_binary(run)
