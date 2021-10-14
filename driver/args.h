/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef __ARGS_H
#define __ARGS_H

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

typedef struct {
    std::string long_name;
    char short_name;
    std::string value;
    std::string help_text;
    std::string type;
} args_input_t;

class args_t {
  public:
    args_t() {}
    void insert_arg(const std::string &_long_name, char _short_name,
                    const std::string &_value, const std::string &_help_text,
                    const std::string &_type) {
        args_input_t in;
        in.long_name = _long_name;
        in.short_name = _short_name;
        in.value = _value;
        in.help_text = _help_text;
        in.type = _type;

        if (input_map.count(_short_name) != 0) {
            printf("arg:%s(%c) already exist\n", _long_name.c_str(),
                   _short_name);
        } else {
            input_map[_short_name] = in;
        }
    }
    void print() {
        for (auto &content : input_map) {
            std::vector<std::string> help_text_lines;
            size_t pos = 0;
            for (size_t next_pos = content.second.help_text.find('\n', pos);
                 next_pos != std::string::npos;) {
                help_text_lines.push_back(
                    std::string(content.second.help_text.begin() + pos,
                                content.second.help_text.begin() + next_pos++));
                pos = next_pos;
                next_pos = content.second.help_text.find('\n', pos);
            }
            help_text_lines.push_back(
                std::string(content.second.help_text.begin() + pos,
                            content.second.help_text.end()));

            std::cout << std::setw(8) << "--" << content.second.long_name
                      << std::setw(20 - content.second.long_name.length())
                      << "-" << content.first << std::setw(8) << " "
                      << help_text_lines[0] << std::endl;

            for (auto help_next_line = std::next(help_text_lines.begin());
                 help_next_line != help_text_lines.end(); ++help_next_line) {
                std::cout << std::setw(37) << " " << *help_next_line
                          << std::endl;
            }
        }
    }
    void parse(int argc, char *argv[]) {
        if (argc <= 2) {
            // printf("not enough args\n");
            return;
        }
        for (int i = 2; i < argc; i++) {
            char *cur_arg = argv[i];
            if (cur_arg[0] != '-') {
                printf("illegal input\n");
                print();
                return;
            } else if (cur_arg[0] == '-' && cur_arg[1] == '-') {
                std::string long_name(cur_arg + 2);
                char short_name = find_short_name(long_name);
                input_map[short_name].value = argv[i + 1];
                i++;
            } else if (cur_arg[0] == '-' && cur_arg[1] == '?') {
                print();
                return;
            } else {
                char short_name = argv[i][1];
                if (input_map.count(short_name) == 0) {
                    printf("arg %c not found\n", short_name);
                    return;
                }
                input_map[short_name].value = argv[i + 1];
                i++;
            }
        }
    }
    char find_short_name(const std::string &long_name) const {
        for (auto &it : input_map) {
            if (it.second.long_name == long_name)
                return it.second.short_name;
        }
        printf("can't find short name for %s\n", long_name.c_str());
        return '\0';
    }
    std::string get_str(const std::string &long_name) const {
        char short_name = find_short_name(long_name);
        std::string value = input_map.at(short_name).value;
        return value;
    }

    int get_int(const std::string &long_name) const {
        char short_name = find_short_name(long_name);
        int value = atoi(input_map.at(short_name).value.c_str());
        return value;
    }

    uint64_t get_uint64(const std::string &long_name) const {
        char short_name = find_short_name(long_name);
        uint64_t value =
            strtoull(input_map.at(short_name).value.c_str(), nullptr, 10);
        return value;
    }

    double get_double(const std::string &long_name) const {
        char short_name = find_short_name(long_name);
        double value = atof(input_map.at(short_name).value.c_str());
        return value;
    }

  private:
    std::unordered_map<char, args_input_t> input_map;
};

static inline std::string create_base_args(int argc, char *argv[]) {
    if(argc < 2)
    {
        printf("Invalid Number of Input Arguments\n");
        exit(0);
    }

    std::string arg = argv[1];

    if(arg != "conv" && arg != "convfp16" && arg != "convint8" && arg != "convbfp16" && arg != "--version")
    {
        printf("Invalid Base Input Argument\n");
        exit(0);
    }
    else if(arg == "-h" || arg == "--help" || arg == "-?")
        exit(0);
    else
        return arg;
}

static inline args_t create_conv_args(int argc, char *argv[]) {
    const std::string base = create_base_args(argc, argv);
    if (argc >= 2 && argv[1] != base) {
        printf("not proper base arg name");
        exit(1);
    }

    args_t args;
    args.insert_arg("in_layout", 'I', "NCHW", "Input Layout (Default=NCHW)", "string");
    args.insert_arg("out_layout", 'O', "NCHW", "Output Layout (Default=NCHW)", "string");
    args.insert_arg("fil_layout", 'f', "NCHW", "Input Layout (Default=NCHW)", "string");
    args.insert_arg("spatial_dim", '_', "2",
                    "convolution spatial dimension (Default-2)", "int");
    args.insert_arg("forw", 'F', "0", "Flag enables fwd, bwd, wrw convolutions"
                                      "\n0 fwd+bwd+wrw (default)"
                                      "\n1 fwd only"
                                      "\n2 bwd only"
                                      "\n4 wrw only"
                                      "\n3 fwd+bwd"
                                      "\n5 fwd+wrw"
                                      "\n6 bwd+wrw",
                    "int");
    args.insert_arg("batchsize", 'n', "100", "Mini-batch size (Default=100)",
                    "int");
    args.insert_arg("in_channels", 'c', "3",
                    "Number of Input Channels (Default=3)", "int");
    args.insert_arg("in_d", '!', "32", "Input Depth (Default=32)", "int");
    args.insert_arg("in_h", 'H', "32", "Input Height (Default=32)", "int");
    args.insert_arg("in_w", 'W', "32", "Input Width (Default=32)", "int");
    args.insert_arg("out_channels", 'k', "32",
                    "Number of Output Channels (Default=32)", "int");
    args.insert_arg("fil_d", '@', "3", "Filter Depth (Default=3)", "int");
    args.insert_arg("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
    args.insert_arg("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
    args.insert_arg("conv_stride_d", '#', "1",
                    "Convolution Stride for Depth (Default=1)", "int");
    args.insert_arg("conv_stride_h", 'u', "1",
                    "Convolution Stride for Height (Default=1)", "int");
    args.insert_arg("conv_stride_w", 'v', "1",
                    "Convolution Stride for Width (Default=1)", "int");
    args.insert_arg("pad_d", '$', "0", "Zero Padding for Depth (Default=0)",
                    "int");
    args.insert_arg("pad_h", 'p', "0", "Zero Padding for Height (Default=0)",
                    "int");
    args.insert_arg("pad_w", 'q', "0", "Zero Padding for Width (Default=0)",
                    "int");
    args.insert_arg("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
    args.insert_arg("trans_output_pad_d", '%', "0",
                    "Zero Padding Output for Depth (Default=0)", "int");
    args.insert_arg("trans_output_pad_h", 'Y', "0",
                    "Zero Padding Output for Height (Default=0)", "int");
    args.insert_arg("trans_output_pad_w", 'X', "0",
                    "Zero Padding Output for Width (Default=0)", "int");
    args.insert_arg("iter", 'i', "10", "Number of Iterations (Default=10)",
                    "int");
    args.insert_arg("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    args.insert_arg(
        "verification_cache", 'C', "",
        "Use specified directory to cache verification data. Off by default.",
        "string");
    args.insert_arg("time", 't', "0", "Time Each Layer (Default=0)", "int");
    args.insert_arg(
        "wall", 'w', "0",
        "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
    args.insert_arg("search", 's', "0", "Search Kernel Config (Default=0)",
                    "int");
    args.insert_arg("printconv", 'P', "1",
                    "Print Convolution Dimensions (Default=1)", "int");
    args.insert_arg("dump_output", 'o', "0",
                    "Dumps the output buffers (Default=0)", "int");
    args.insert_arg("in_data", 'd', "", "Input data filename (Default=)",
                    "string");
    args.insert_arg("weights", 'e', "", "Input weights filename (Default=)",
                    "string");
    args.insert_arg("bias", 'b', "", "Use Bias (Default=0)", "int");
    args.insert_arg("mode", 'm', "conv",
                    "Convolution Mode (conv, trans) (Default=conv)", "str");

    args.insert_arg("pad_mode", 'z', "default",
                    "Padding Mode (same, valid, default) (Default=default)",
                    "str");
    args.insert_arg(
        "tensor_vect", 'Z', "0",
        "tensor vectorization type (none, vect_c, vect_n) (Default=0)", "int");
    args.insert_arg("dilation_d", '^', "1",
                    "Dilation of Filter Depth (Default=1)", "int");
    args.insert_arg("dilation_h", 'l', "1",
                    "Dilation of Filter Height (Default=1)", "int");
    args.insert_arg("dilation_w", 'j', "1",
                    "Dilation of Filter Width (Default=1)", "int");
    args.insert_arg("in_bias", 'a', "", "Input bias filename (Default=)",
                    "string");
    args.insert_arg("group_count", 'g', "1", "Number of Groups (Default=1)",
                    "int");
    args.insert_arg(
        "dout_data", 'D', "",
        "dy data filename for backward weight computation (Default=)",
        "string");
    args.insert_arg("solution", 'S', "-1",
                    "Use immediate mode, run solution with specified id."
                    "\nAccepts integer argument N:"
                    "\n=0 Immediate mode, build and run fastest solution"
                    "\n>0 Immediate mode, build and run solution_id = N"
                    "\n<0 Use Find() API (Default=-1)",
                    "int");
    args.parse(argc, argv);
    return args;
}

#endif