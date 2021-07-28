#include "log.hpp"

static const char* level_string[] = { "FATAL ERROR: ", "ERROR: ", "WARNING: ", "INFO: ", "DBG: ", "TRACEINFO: " };

Logger LOG(std::cout);

std::ostream& Logger::operator()(severity l, bool show_level)
{
	if (l > loglevel)
		return null_stream;

	if (show_level)
		return os << level_string[(size_t)l];

	return os;
}
