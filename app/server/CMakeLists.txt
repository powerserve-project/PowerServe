add_library(server.simple STATIC simple_server.cpp)
target_link_libraries(server.simple PUBLIC
    powerserve
    httplib::httplib
    nlohmann_json::nlohmann_json
    concurrentqueue
)
if (POWERSERVE_SERVER_MULTIMODEL)
    target_compile_definitions(server.simple PUBLIC POWERSERVE_SERVER_MULTIMODEL)
endif ()

add_library(server.local STATIC local_server.cpp)
target_link_libraries(server.local PUBLIC
    powerserve
    httplib::httplib
    nlohmann_json::nlohmann_json
    concurrentqueue
)
if (POWERSERVE_SERVER_MULTIMODEL)
    target_compile_definitions(server.local PUBLIC POWERSERVE_SERVER_MULTIMODEL)
endif ()

add_executable(server server.cpp)
target_link_libraries(server PRIVATE
    powerserve
    app.common
    server.simple
)

powerserve_add_binary(server)
