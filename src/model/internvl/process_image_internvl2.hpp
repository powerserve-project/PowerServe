// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "core/logger.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

std::vector<float> pixel_values;

class InternVL2ImageProcessor {
public:
    InternVL2ImageProcessor(int input_size = 448, int max_num = 6) : input_size(input_size), max_num(max_num) {}

    std::vector<float> load_image(const std::string &image_path) {
        int width, height, channels;
        POWERSERVE_ASSERT(std::filesystem::exists(image_path));
        unsigned char *data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
        if (!data) {
            throw std::runtime_error("Failed to load image");
        }

        // if (channels != 3) {
        //     stbi_image_free(data);
        //     throw std::runtime_error("Image must be RGB");
        // }
        // if it is not RGB, convert it to RGB
        if (channels == 1) {
            unsigned char *data_rgb = new unsigned char[width * height * 3];
            for (int i = 0; i < width * height; i++) {
                data_rgb[i * 3]     = data[i];
                data_rgb[i * 3 + 1] = data[i];
                data_rgb[i * 3 + 2] = data[i];
            }
            stbi_image_free(data);
            data     = data_rgb;
            channels = 3;
        }
        // if it is RGBA, convert it to RGB
        if (channels == 4) {
            unsigned char *data_rgb = new unsigned char[width * height * 3];
            for (int i = 0; i < width * height; i++) {
                data_rgb[i * 3]     = data[i * 4];
                data_rgb[i * 3 + 1] = data[i * 4 + 1];
                data_rgb[i * 3 + 2] = data[i * 4 + 2];
            }
            stbi_image_free(data);
            data     = data_rgb;
            channels = 3;
        }

        auto processed_images = dynamic_preprocess(data, width, height);
        std::vector<xt::xarray<float>> pixel_values;

        for (auto &img : processed_images) {
            pixel_values.push_back(transform(img, input_size, input_size));
        }

        stbi_image_free(data);

        // std::vector<std::vector<std::vector<std::vector<float>>>> pixel_values_vector;
        // for (auto &img : pixel_values) {
        //     auto img_vector = std::vector<std::vector<std::vector<float>>>();
        //     for (int i = 0; i < img.shape()[0]; i++) {
        //         auto channel = std::vector<std::vector<float>>();
        //         for (int j = 0; j < img.shape()[1]; j++) {
        //             auto row = std::vector<float>();
        //             for (int k = 0; k < img.shape()[2]; k++) {
        //                 row.push_back(img(i, j, k));
        //             }
        //             channel.push_back(row);
        //         }
        //         img_vector.push_back(channel);
        //     }
        //     pixel_values_vector.push_back(img_vector);
        // }

        std::vector<float> pixel_values_vector;
        for (auto &img : pixel_values) {
            for (size_t i = 0; i < img.shape()[0]; i++) {
                for (size_t j = 0; j < img.shape()[1]; j++) {
                    for (size_t k = 0; k < img.shape()[2]; k++) {
                        pixel_values_vector.push_back(img(i, j, k));
                    }
                }
            }
        }

        return {pixel_values_vector};
    }

private:
    int input_size;
    int max_num;

    std::vector<std::vector<unsigned char>> dynamic_preprocess(unsigned char *data, int orig_width, int orig_height) {
        float aspect_ratio = static_cast<float>(orig_width) / orig_height;

        auto target_ratios       = calculate_target_ratios();
        auto target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height);

        int target_width  = input_size * target_aspect_ratio.first;
        int target_height = input_size * target_aspect_ratio.second;
        int blocks        = target_aspect_ratio.first * target_aspect_ratio.second;

        std::vector<unsigned char> resized_img(target_width * target_height * 3);
        stbir_resize_uint8_srgb(
            data, orig_width, orig_height, 0, resized_img.data(), target_width, target_height, 0, STBIR_RGB
        );

        std::vector<std::vector<unsigned char>> processed_images;
        for (int i = 0; i < blocks; ++i) {
            int x = (i % (target_width / input_size)) * input_size;
            int y = (i / (target_width / input_size)) * input_size;
            std::vector<unsigned char> split_img(input_size * input_size * 3);
            for (int yy = 0; yy < input_size; ++yy) {
                for (int xx = 0; xx < input_size; ++xx) {
                    for (int c = 0; c < 3; ++c) {
                        split_img[(yy * input_size + xx) * 3 + c] =
                            resized_img[((y + yy) * target_width + (x + xx)) * 3 + c];
                    }
                }
            }
            processed_images.push_back(split_img);
        }

        if (blocks != 1) {
            std::vector<unsigned char> thumbnail_img(input_size * input_size * 3);
            stbir_resize_uint8_srgb(
                data, orig_width, orig_height, 0, thumbnail_img.data(), input_size, input_size, 0, STBIR_RGB
            );
            processed_images.push_back(thumbnail_img);
        }

        return processed_images;
    }

    std::vector<std::pair<int, int>> calculate_target_ratios() {
        std::vector<std::pair<int, int>> target_ratios;
        for (int n = 1; n <= max_num; ++n) {
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= n; ++j) {
                    if (i * j <= max_num && i * j >= 1) {
                        target_ratios.emplace_back(i, j);
                    }
                }
            }
        }
        std::sort(target_ratios.begin(), target_ratios.end(), [](auto &a, auto &b) {
            return a.first * a.second < b.first * b.second;
        });
        return target_ratios;
    }

    std::pair<int, int> find_closest_aspect_ratio(
        float aspect_ratio, const std::vector<std::pair<int, int>> &target_ratios, int width, int height
    ) {
        float best_ratio_diff          = std::numeric_limits<float>::infinity();
        std::pair<int, int> best_ratio = {1, 1};
        int area                       = width * height;

        for (const auto &ratio : target_ratios) {
            float target_aspect_ratio = static_cast<float>(ratio.first) / ratio.second;
            float ratio_diff          = std::abs(aspect_ratio - target_aspect_ratio);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                best_ratio      = ratio;
            } else if (ratio_diff == best_ratio_diff) {
                if (area > 0.5 * input_size * input_size * ratio.first * ratio.second) {
                    best_ratio = ratio;
                }
            }
        }

        return best_ratio;
    }

    xt::xarray<float> transform(const std::vector<unsigned char> &img, int width, int height) {
        xt::xarray<float> tensor = xt::zeros<float>({3, height, width});
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    float pixel_value = static_cast<float>(img[(y * width + x) * 3 + c]) / 255.0f;
                    tensor(c, y, x)   = (pixel_value - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                }
            }
        }
        return tensor;
    }

    const std::vector<float> IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    const std::vector<float> IMAGENET_STD  = {0.229f, 0.224f, 0.225f};
};
