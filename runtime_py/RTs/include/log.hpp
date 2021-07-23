#ifndef LOG_HPP__
#define LOG_HPP__ 1

#include <iostream>

enum class severity : uint
{
    FATAL   = 0,
    ERROR   = 1,
    WARNING = 2,
    INFO    = 3,
    DEBUG   = 4,
    TRACE   = 5,
};

class Logger { // generic functor
private:
    severity loglevel;
    std::ostream& os;
    std::ostream null_stream;

public:
    Logger(std::ostream& os)
        : loglevel(severity::WARNING)
        , os(os)
        , null_stream(0) {
        null_stream.setstate(std::ios::failbit);
    }
    
    void SetSeverity(severity l) { loglevel = l; }
    void SetSeverity(uint l) { loglevel = severity(l < 5 ? l : 5); }
    
    std::ostream& operator()(severity l, bool show_level = true);    
};

extern Logger LOG;


#endif