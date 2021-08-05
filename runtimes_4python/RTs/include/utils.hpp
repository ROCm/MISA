#ifndef UTILS_HPP__
#define UTILS_HPP__ 1

#include <vector>
#include <string>
using std::vector;
using std::string;
using std::size_t;
using std::ostream;
using std::istream;

int ExecuteProcess(const string& p, vector<string>& args, istream* in, ostream* out, ostream* err);
bool write_file(const char* path, const void* data, size_t size);
bool read_file(const char* path, void* data, size_t size);
bool read_file(const char* path, vector<char> &bin);
bool read_file(const char* path, string &str);

#ifdef WIN32
int setenv(const char* name, const char* value, int overwrite);
#endif

class TempFile
{
public:
	TempFile(const string& path_template);
	~TempFile();
	inline operator const string& () { return _path; }
	const char* path() { return _path.c_str(); }

private:
	string _path;
	int _fd;

	static const string GetTempDirectoryPath();
};

#endif