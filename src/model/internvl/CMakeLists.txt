target_sources(powerserve PRIVATE
    "internvl_model.cpp"
)
target_link_libraries(powerserve PRIVATE stb_headers)
target_link_libraries(powerserve PRIVATE xtensor xtl)
