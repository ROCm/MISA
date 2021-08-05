#include "utils.hpp"
#include "log.hpp"
#include <cassert>
#include <sstream>
#include <fstream>

bool write_file(const char* path, const void* data, size_t size)
{
	std::ofstream os(path, std::ofstream::binary | std::ofstream::trunc);
	if (os)
		os.write((const char*)data, size);

	if (!os)
		LOG(severity::ERROR) << "unable to write file \"" << path << "\"\n";

	return os ? true : false;
}

bool read_file(const char* path, void* data, size_t size)
{
	std::ifstream in(path, std::ifstream::binary);
	if (in)
		in.read((char*)data, size);

	if (!in)
		LOG(severity::ERROR) << "unable to read file \"" << path << "\"\n";

	return in ? true : false;
}

bool read_file(const char* path, vector<char>& bin)
{
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	const auto size = file.tellg();
	bool read_failed = false;
	do {
		if (size < 0) { read_failed = true; break; }
		file.seekg(std::ios::beg);
		if (file.fail()) { read_failed = true; break; }
		bin.resize(size);
		if (file.rdbuf()->sgetn(bin.data(), size) != size) { read_failed = true; break; }
	} while (false);
	file.close();

	if (read_failed)
		LOG(severity::ERROR) << "unable to read file \"" << path << "\"\n";

	return !read_failed;
}

bool read_file(const char* path, string& str)
{
	std::ifstream in(path);
	if (!in)
	{
		LOG(severity::ERROR) << "unable to read file \"" << path << "\"\n";
		return false;
	}
	getline(in, str, string::traits_type::to_char_type(string::traits_type::eof()));
	in.close();

	return true;
}
