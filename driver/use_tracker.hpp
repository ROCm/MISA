#pragma once

#include <assert.h>
#include <string>
#include <vector>

class if_class {
  public:
    if_class(std::function<void()> push_fn, std::function<void()> pop_fn)
        : push_fn_(std::move(push_fn)), pop_fn_(std::move(pop_fn)) {
        push_fn_();
    }

    ~if_class() { pop_fn_(); }

  private:
    std::function<void()> push_fn_;
    std::function<void()> pop_fn_;
};

#define SETUP_TRACKING()                                                       \
    bool report_IsValid =                                                      \
        std::string(env_get_str("IGEMM_KVALID_TARGET", const_cast<char*>(""))) != "";             \
    std::vector<std::string> layer_stack_;                                     \
    auto print_stack = [&]() {                                                 \
        std::cout << "\n";                                                     \
        for (int i = 0; i < layer_stack_.size(); i++)                          \
            std::cout << layer_stack_[i] << "\n";                              \
    };                                                                         \
    auto tracking_if_push_ = [&](auto expr) { layer_stack_.push_back(expr); }; \
    auto tracking_if_pop_ = [&]() {                                            \
        if (layer_stack_.size() >= 1) {                                        \
            layer_stack_.pop_back();                                           \
        }                                                                      \
    };                                                                         \
    auto make_if_tracker = [&](const std::string &expr_str) -> if_class {      \
        if (report_IsValid)                                                    \
            return if_class(                                                   \
                [&]() { tracking_if_push_("(" + expr_str + ")"); },            \
                [&]() { tracking_if_pop_(); });                                \
        else                                                                   \
            return if_class([&]() {}, [&]() {});                               \
    }

#define IF_CHECK(expr)                                                         \
    if (auto _if_guard_##__LINE__ = make_if_tracker(#expr); expr)

#define ELSE_IF_CHECK(expr)                                                    \
    else if (if_class _if_guard_##__LINE__ = make_if_tracker(#expr); expr)

#define ELSE_CHECK() else

#define TRACK_RETURN(val)                                                      \
    {                                                                          \
        if (report_IsValid) {                                                  \
            tracking_if_push_(std::string("\n\nIs_valid failed. Track End"));  \
            tracking_if_push_(std::string(" | File: ") + __FILE__ +            \
                              std::string(" | Line: ") +                       \
                              std::to_string(__LINE__) + std::string("\n"));   \
            print_stack();                                                     \
        }                                                                      \
        return val;                                                            \
    }
