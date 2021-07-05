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
 * The above copyright notice and this permission notice shall be included in
 *all
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
#ifndef __CONFIG_PARSER_H
#define __CONFIG_PARSER_H

#include <assert.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

static inline std::string &sltrim(std::string &s) {
    size_t startpos = s.find_first_not_of(" \t\r\n\v\f");
    if (std::string::npos != startpos) {
        s = s.substr(startpos);
    }else{
        s = "";
    }
    return s;
}

static inline std::string &srtrim(std::string &s) {
    size_t endpos = s.find_last_not_of(" \t\r\n\v\f");
    if (std::string::npos != endpos) {
        s = s.substr(0, endpos + 1);
    }else{
        s = "";
    }
    return s;
}

static inline std::string &strim(std::string &s) { return sltrim(srtrim(s)); }

static inline std::string remove_trailing_comment(std::string & s)
{
    size_t pos = s.find_first_of(";#");
    if(std::string::npos != pos) {
        s = s.erase(pos, std::string::npos);
    }
    return s;
}

static inline std::vector<std::string> ssplit(const std::string &s,
                                              char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

enum class config_section_value_type_enum {
    config_section_value_type_int = 0,
    config_section_value_type_float = 1,
    config_section_value_type_range = 2,
    config_section_value_type_list_int = 3,
    config_section_value_type_list_float = 4,
    config_section_value_type_list_string = 5,
    config_section_value_type_string = 6,
    config_section_value_type_non = 7
};

class config_section_value_t;

template <config_section_value_type_enum value_type>
struct section_meta_value_t {
    typedef decltype(value_type) type;
    static type decode(const std::vector<uint8_t> &buffer) {
        return value_type;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        (void)buffer;
        (void)value;
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        return std::string("");
    }
};

template <>
struct section_meta_value_t<
    config_section_value_type_enum::config_section_value_type_int> {
    typedef int type;
    static type decode(const std::vector<uint8_t> &buffer) {
        assert(buffer.size() == 4);
        int res;
        memcpy(&res, buffer.data(), 4);
        return res;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        buffer.resize(4);
        int res = std::stoi(value);
        memcpy(buffer.data(), &res, 4);
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        int value = decode(buffer);
        return std::to_string(value);
    }
};

template <>
struct section_meta_value_t<
    config_section_value_type_enum::config_section_value_type_float> {
    typedef float type;
    static type decode(const std::vector<uint8_t> &buffer) {
        assert(buffer.size() == 4);
        float res;
        memcpy(&res, buffer.data(), 4);
        return res;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        buffer.resize(4);
        float res = std::stof(value);
        memcpy(buffer.data(), &res, 4);
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        float value = decode(buffer);
        return std::to_string(value);
    }
};

template <>
struct section_meta_value_t<
    config_section_value_type_enum::config_section_value_type_range> {
    typedef std::vector<int> type;
    static type decode(const std::vector<uint8_t> &buffer) {
        assert(buffer.size() != 0 && buffer.size() % 4 == 0);
        int range_index_length = buffer.size() / 4;
        std::vector<int> range_index;
        range_index.resize(range_index_length);
        for (int i = 0; i < range_index_length; i++) {
            int res;
            memcpy(&res, buffer.data()+ 4 * i, 4);
            range_index[i] = res;
        }
        return range_index;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        int start, end, step;
        std::string v = value.substr(1, value.length() - 2);
        std::vector<std::string> ranges = ssplit(v, ',');
        if (ranges.size() == 1) {
            start = 0;
            end = std::stoi(ranges[0]);
            step = 1;
        } else if (ranges.size() == 2) {
            start = std::stoi(ranges[0]);
            end = std::stoi(ranges[1]);
            step = 1;
        } else if (ranges.size() == 3) {
            start = std::stoi(ranges[0]);
            end = std::stoi(ranges[1]);
            step = std::stoi(ranges[2]);
        } else {
            assert(false);
        }

        std::vector<int> range_index;
        if (step >= 0)
            for (int i = start; i < end; i += step)
                range_index.push_back(i);
        else
            for (int i = start; i > end; i += step)
                range_index.push_back(i);

        buffer.resize(4 * range_index.size());
        for (int i = 0; i < (int)range_index.size(); i++) {
            int res = range_index[i];
            memcpy(buffer.data() + 4 * i, &res, 4);
        }
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        std::vector<int> value = decode(buffer);
        std::string str = "[";
        str += std::to_string(value[0]);
        for (int i = 1; i < (int)value.size(); i++) {
            str += std::string(",");
            str += std::to_string(value[i]);
        }
        str += "]";
        return str;
    }
};

template <>
struct section_meta_value_t<
    config_section_value_type_enum::config_section_value_type_list_int> {
    typedef std::vector<int> type;
    static type decode(const std::vector<uint8_t> &buffer) {
        assert(buffer.size() != 0 && buffer.size() % 4 == 0);
        std::vector<int> list_value;
        list_value.resize(buffer.size() / 4);
        for (int i = 0; i < (int)(buffer.size() / 4); i++) {
            int res;
            memcpy(&res, buffer.data()+ 4 * i, 4);
            list_value[i] = res;
        }
        return list_value;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        std::string v = value.substr(1, value.length() - 2);
        std::vector<std::string> list_string = ssplit(v, ',');
        buffer.resize(4 * list_string.size());
        for (int i = 0; i < (int)list_string.size(); i++) {
            int res = std::stoi(list_string[i]);
            memcpy(buffer.data() + 4 * i, &res, 4);
        }
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        std::vector<int> value = decode(buffer);
        std::string str = "[";
        str += std::to_string(value[0]);
        for (int i = 1; i < (int)value.size(); i++) {
            str += std::string(",");
            str += std::to_string(value[i]);
        }
        str += std::string("]");
        return str;
    }
};

template <>
struct section_meta_value_t<
    config_section_value_type_enum::config_section_value_type_list_float> {
    typedef std::vector<float> type;
    static type decode(const std::vector<uint8_t> &buffer) {
        assert(buffer.size() != 0 && buffer.size() % 4 == 0);
        std::vector<float> list_value;
        list_value.resize(buffer.size() / 4);
        for (int i = 0; i < (int)(buffer.size() / 4); i++) {
            float res;
            memcpy(&res, buffer.data() + 4 * i, 4);
            list_value[i] = res;
        }
        return list_value;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        std::string v = value.substr(1, value.length() - 2);
        std::vector<std::string> list_string = ssplit(v, ',');
        buffer.resize(4 * list_string.size());
        for (int i = 0; i < (int)list_string.size(); i++) {
            float res = std::stof(list_string[i]);
            memcpy(buffer.data() + 4 * i, &res, 4);
        }
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        std::vector<float> value = decode(buffer);
        std::string str = "[";
        str += std::to_string(value[0]);
        for (int i = 1; i < (int)value.size(); i++) {
            str += std::string(",");
            str += std::to_string(value[i]);
        }
        str += "]";
        return str;
    }
};

template <>
struct section_meta_value_t<
    config_section_value_type_enum::config_section_value_type_list_string> {
    typedef std::vector<std::string> type;
    static type decode(const std::vector<uint8_t> &buffer) {
        assert(buffer.size() != 0);
        std::vector<std::string> list_value;
        std::string v;
        for (int i = 0; i < (int)buffer.size(); i++) {
            if (buffer[i] != static_cast<uint8_t>('\0')) {
                v.append(1, static_cast<const char>(buffer[i]));
            } else {
                list_value.push_back(v);
                v.clear();
            }
        }
        return list_value;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        std::string v = value.substr(1, value.length() - 2);
        std::vector<std::string> list_string = ssplit(v, ',');
        for (int i = 0; i < (int)list_string.size(); i++) {
            std::string str =
                strim(list_string[i]).substr(1, list_string[i].length() - 2);
            strim(str);
            for (int j = 0; j < (int)str.size(); j++) {
                buffer.push_back(static_cast<uint8_t>(str[j]));
            }
            buffer.push_back(static_cast<uint8_t>('\0'));
        }
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        std::vector<std::string> value = decode(buffer);
        std::string str = "[";
        str += (std::string("\'") + value[0] + std::string("\'"));
        for (int i = 1; i < (int)value.size(); i++) {
            str += std::string(",");
            str += (std::string("\'") + value[i] + std::string("\'"));
        }
        str += "]";
        return str;
    }
};

template <>
struct section_meta_value_t<
    config_section_value_type_enum::config_section_value_type_string> {
    typedef std::string type;
    static type decode(const std::vector<uint8_t> &buffer) {
        assert(buffer.size() != 0);
        std::string value_string;
        value_string.resize(buffer.size());
        for (int i = 0; i < (int)buffer.size(); i++)
            value_string[i] = static_cast<const char>(buffer[i]);
        //value_string.back() = '\0';
        return value_string;
    }
    static void encode(std::vector<uint8_t> &buffer, std::string value) {
        std::string v = value.substr(1, value.length() - 2);
        strim(v);
        buffer.resize(v.size());
        for (int i = 0; i < (int)v.size(); i++) {
            buffer[i] = static_cast<uint8_t>(v[i]);
        }
        //buffer.back() = static_cast<uint8_t>('\0');
    }
    static std::string serialize(const std::vector<uint8_t> &buffer) {
        return std::string("\'") + decode(buffer) + std::string("\'");
    }
};

#define CDECODE(type_enum_trait)                                               \
    static typename section_meta_value_t<                                      \
        config_section_value_type_enum::                                       \
            config_section_value_type_##type_enum_trait>::type                 \
        decode_##type_enum_trait(const std::vector<uint8_t> &buffer) {         \
        return section_meta_value_t<                                           \
            config_section_value_type_enum::                                   \
                config_section_value_type_##type_enum_trait>::decode(buffer);  \
    }                                                                          \
    typename section_meta_value_t<                                             \
        config_section_value_type_enum::                                       \
            config_section_value_type_##type_enum_trait>::type                 \
        get_##type_enum_trait() const {                                        \
        return section_meta_value_t<                                           \
            config_section_value_type_enum::                                   \
                config_section_value_type_##type_enum_trait>::                 \
            decode(value_buffer);                                              \
    }

#define CENCODE(type_enum_trait)                                               \
    static void encode_##type_enum_trait(std::vector<uint8_t> &buffer,         \
                                         std::string value) {                  \
        section_meta_value_t<                                                  \
            config_section_value_type_enum::                                   \
                config_section_value_type_##type_enum_trait>::encode(buffer,   \
                                                                     value);   \
    }

class config_section_value_t {
  public:
    config_section_value_t() {}
    config_section_value_t(std::string value_,
                           config_section_value_type_enum type_)
        : value_string(value_), value_type(type_) {}
    config_section_value_t(const config_section_value_t &other) {
        this->value_string = other.value_string;
        this->value_type = other.value_type;
        this->value_buffer = other.value_buffer;
    }
    config_section_value_t(config_section_value_t &&other) {
        this->value_string = other.value_string;
        this->value_type = other.value_type;
        this->value_buffer = other.value_buffer;
    }
    config_section_value_t &operator=(const config_section_value_t &other) {
        this->value_string = other.value_string;
        this->value_type = other.value_type;
        this->value_buffer = other.value_buffer;
        return *this;
    }
    config_section_value_t &operator=(config_section_value_t &&other) {
        this->value_string = other.value_string;
        this->value_type = other.value_type;
        this->value_buffer = other.value_buffer;
        return *this;
    }

    static bool is_value_int(std::string v) {
        try {
            int i = std::stoi(v);
            (void)i;
            return true;
        } catch (...) {
            return false;
        }
    }
    static bool is_value_float(std::string v) {
        try {
            float f = std::stof(v);
            (void)f;
            return true;
        } catch (...) {
            return false;
        }
    }
    static bool is_value_string(std::string v) {
        if ((v[0] == '\'' && v.back() == '\'') ||
            (v[0] == '\"' && v.back() == '\"'))
            return true;
        return false;
    }
    static bool is_value_list_int(std::string v) {
        bool valid = true;
        if (v[0] == '[' && v.back() == ']') {
            std::string vv = v.substr(1, v.length() - 2);
            std::vector<std::string> vlist = ssplit(vv, ',');
            for (auto e : vlist) {
                strim(e);
                if (e.empty())
                    return false;
                valid &= is_value_int(e);
            }
            return valid;
        }
        return false;
    }
    static bool is_value_list_float(std::string v) {
        bool valid = true;
        if (v[0] == '[' && v.back() == ']') {
            std::string vv = v.substr(1, v.length() - 2);
            std::vector<std::string> vlist = ssplit(vv, ',');
            for (auto e : vlist) {
                strim(e);
                if (e.empty())
                    return false;
                valid &= is_value_float(e);
            }
            return valid;
        }
        return false;
    }
    static bool is_value_list_string(std::string v) {
        bool valid = true;
        if (v[0] == '[' && v.back() == ']') {
            std::string vv = v.substr(1, v.length() - 2);
            std::vector<std::string> vlist = ssplit(vv, ',');
            for (auto e : vlist) {
                strim(e);
                if (e.empty())
                    return false;
                valid &= is_value_string(e);
            }
            return valid;
        }
        return false;
    }
    static bool is_value_range(std::string v) {
        if (v[0] == '(' && v.back() == ')') {
            std::string vv = v.substr(1, v.length() - 2);
            std::vector<std::string> vlist = ssplit(vv, ',');
            for (auto e : vlist) {
                strim(e);
                if (e.empty())
                    return false;
            }
            if (vlist.size() == 1 || vlist.size() == 2 || vlist.size() == 3)
                return true;
            return false;
        }
        return false;
    }

    template <config_section_value_type_enum value_type>
    static typename section_meta_value_t<value_type>::type
    decode(const std::vector<char> &buffer) {
        return section_meta_value_t<value_type>::decode(buffer);
    }

    template <config_section_value_type_enum value_type>
    typename section_meta_value_t<value_type>::type get_value() const {
        return section_meta_value_t<value_type>::decode(value_buffer);
    }

    CDECODE(int)
    CDECODE(float)
    CDECODE(range)
    CDECODE(list_int)
    CDECODE(list_float)
    CDECODE(list_string)
    CDECODE(string)

    template <config_section_value_type_enum value_type>
    static void encode(std::vector<char> &buffer, std::string value) {
        (void)buffer;
        (void)value;
    }

    CENCODE(int)
    CENCODE(float)
    CENCODE(range)
    CENCODE(list_int)
    CENCODE(list_float)
    CENCODE(list_string)
    CENCODE(string)

    static config_section_value_t parse_value(std::string v) {
#define PARSE_VALUE(type_enum_trait)                                           \
    if (is_value_##type_enum_trait(v)) {                                       \
        config_section_value_t meta_value(                                     \
            v, config_section_value_type_enum::                                \
                   config_section_value_type_##type_enum_trait);               \
        config_section_value_t::encode_##type_enum_trait(                      \
            meta_value.get_buffer(), v);                                       \
        return meta_value;                                                     \
    }
        PARSE_VALUE(int)
        PARSE_VALUE(float)
        PARSE_VALUE(range)
        PARSE_VALUE(list_int)
        PARSE_VALUE(list_float)
        PARSE_VALUE(list_string)
        PARSE_VALUE(string)
        assert(false);
#undef PARSE_VALUE
        return config_section_value_t(
            "", config_section_value_type_enum::config_section_value_type_non);
    }

    std::vector<uint8_t> &get_buffer() { return value_buffer; }
    const std::vector<uint8_t> &get_buffer() const { return value_buffer; }
    std::string serialize() const {
        if (value_type ==
            config_section_value_type_enum::config_section_value_type_int)
            return section_meta_value_t<
                config_section_value_type_enum::config_section_value_type_int>::
                serialize(value_buffer);
        if (value_type ==
            config_section_value_type_enum::config_section_value_type_float)
            return section_meta_value_t<
                config_section_value_type_enum::
                    config_section_value_type_float>::serialize(value_buffer);
        if (value_type ==
            config_section_value_type_enum::config_section_value_type_range)
            return section_meta_value_t<
                config_section_value_type_enum::
                    config_section_value_type_range>::serialize(value_buffer);
        if (value_type ==
            config_section_value_type_enum::config_section_value_type_list_int)
            return section_meta_value_t<
                config_section_value_type_enum::
                    config_section_value_type_list_int>::
                serialize(value_buffer);
        if (value_type == config_section_value_type_enum::
                              config_section_value_type_list_float)
            return section_meta_value_t<
                config_section_value_type_enum::
                    config_section_value_type_list_float>::
                serialize(value_buffer);
        if (value_type == config_section_value_type_enum::
                              config_section_value_type_list_string)
            return section_meta_value_t<
                config_section_value_type_enum::
                    config_section_value_type_list_string>::
                serialize(value_buffer);
        if (value_type ==
            config_section_value_type_enum::config_section_value_type_string)
            return section_meta_value_t<
                config_section_value_type_enum::
                    config_section_value_type_string>::serialize(value_buffer);
        assert(false);
        return std::string("");
    }

  private:
    std::string value_string;
    std::vector<uint8_t> value_buffer;
    config_section_value_type_enum value_type;
};

class config_section_t {
  public:
    config_section_t(std::string name_) : name(name_) {}
    std::string get_name() const { return name; }
    std::unordered_map<std::string, config_section_value_t>::iterator begin() {
        return dict.begin();
    }
    std::unordered_map<std::string, config_section_value_t>::iterator end() {
        return dict.end();
    }
    std::unordered_map<std::string, config_section_value_t>::const_iterator
    begin() const {
        return dict.cbegin();
    }
    std::unordered_map<std::string, config_section_value_t>::const_iterator
    end() const {
        return dict.cend();
    }

    config_section_value_t &at(std::string key) { return this->dict[key]; }
    const config_section_value_t &at(std::string key) const {
        return this->dict.at(key);
    }
    size_t count(std::string key) const { return this->dict.count(key); }

  private:
    std::string name;
    std::unordered_map<std::string, config_section_value_t> dict;
};

class config_content_t {
  public:
    void add_section(config_section_t section) {
        this->sections.push_back(section);
    }
    config_section_t get_section(std::string sec_name) const {
        // return first section with name 'sec_name'
        for(auto &sec : sections){
            if(sec.get_name() == sec_name)
                return sec;
        }
        return config_section_t("sec_na");
    }
    std::vector<config_section_t>::iterator begin() { return sections.begin(); }
    std::vector<config_section_t>::iterator end() { return sections.end(); }
    std::vector<config_section_t>::const_iterator begin() const {
        return sections.cbegin();
    }
    std::vector<config_section_t>::const_iterator end() const {
        return sections.cend();
    }

    void dump() const {
        printf("total sections:%d\n", (int)sections.size());
        for (int i = 0; i < (int)sections.size(); i++) {
            const config_section_t &section = sections[i];
            printf("[%s]\n", section.get_name().c_str());
            for (const auto kv : section) {
                printf("  %s = %s\n", kv.first.c_str(),
                       kv.second.serialize().c_str());
            }
        }
    }

  private:
    std::vector<config_section_t> sections;
};

/*
* config file need to be in unix format.
* If parse fail and have strange behavior, consider run 'dos2unix'
*/
class config_parser_t {
  public:
    config_parser_t(std::string config_file_) : config_file(config_file_) {}

    config_content_t parse() {
        std::ifstream fs;
        config_content_t config_content;
        config_section_t *section = NULL;
        fs.open(config_file);
        if (!fs) {
            printf("fail to open file:%s, %s\n", config_file.c_str(),
                   strerror(errno));
            exit(-1);
        }
        for (std::string line; std::getline(fs, line);) {
            remove_trailing_comment(line);
            strim(line);
            // printf("[%d], line:\"%s\"\n", __LINE__, line.c_str());
            if (is_empty(line) || is_comment(line))
                continue;
            if (is_section(line)) {
                if (section) {
                    config_content.add_section(*section);
                    delete section;
                }
                section = new config_section_t(get_section_name(line));
            } else {
                if (!section) {
                    printf("no current section, should not happen\n");
                    exit(-1);
                }
                std::vector<std::string> toks = ssplit(line, '=');
                if (toks.size() != 2) {
                    printf("fail to parse current line:%s, not enough tokens\n",
                           line.c_str());
                    exit(-1);
                }
                for (int i = 0; i < (int)toks.size(); i++) {
                    std::string tok = toks[i];
                    strim(tok);
                    if (tok == "") {
                        printf("fail to parse current line:%s, token empty\n",
                               line.c_str());
                    }
                    toks[i] = tok;
                }
                std::string key = toks[0];
                std::string value = toks[1];
                if (section->count(key)) {
                    printf("duplicate key %s in current section\n",
                           key.c_str());
                    exit(-1);
                }
                section->at(key) = config_section_value_t::parse_value(value);
            }
        }
        if (section) {
            config_content.add_section(*section);
            delete section;
        }
        return config_content;
    }

  private:
    std::string config_file;

    bool is_empty(std::string line) { return line.empty(); }
    bool is_comment(std::string line) {
        if (line[0] == '#' || line[0] == ';')
            return true;
        return false;
    }
    bool is_section(std::string line) {
        if (line[0] == '[' && line.back() == ']')
            return true;
        return false;
    }
    std::string get_section_name(std::string line) {
        std::string sec_name = line.substr(1, line.length() - 2);
        strim(sec_name);
        return sec_name;
    }
};

#endif