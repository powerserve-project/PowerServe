file(GLOB_RECURSE storage_source *.cpp)
target_sources(powerserve PRIVATE
    ${storage_source}
)
