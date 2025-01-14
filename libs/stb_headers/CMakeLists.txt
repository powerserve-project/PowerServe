add_library(stb_headers INTERFACE
    "stb/stb_image.h"
    "stb/stb_image_resize2.h"
)
target_include_directories(stb_headers INTERFACE ./stb)
