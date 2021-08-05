#ifndef WIN32

#include "utils.hpp"
#include "log.hpp"
#include <cassert>
#include <sstream>
#include <unistd.h>
#include <ext/stdio_filebuf.h>
#include <sys/wait.h>

class Pipe
{
public:
	Pipe()
		: _read_side_closed(true)
		, _write_side_closed(true)
	{
	}

	Pipe(Pipe&&) = delete;
	Pipe(Pipe&) = delete;
	Pipe& operator =(Pipe&) = delete;
	~Pipe() { Close(); }
	void CloseRead() { CloseSide(_read_side, _read_side_closed); }
	void CloseWrite() { CloseSide(_write_side, _write_side_closed); }
	int DupRead(int target_fd) { assert(!_read_side_closed); return dup2(_read_side, target_fd); }
	int DupWrite(int target_fd) { assert(!_write_side_closed); return dup2(_write_side, target_fd); }
	int GetReadFd() { return _read_side; }
	int GetWriteFd() { return _write_side; }

	void Close()
	{
		CloseRead();
		CloseWrite();
	}

	void Open()
	{
		if (pipe(_sides)) { LOG(severity::ERROR) << "pipe()\n"; exit(-1); }
		_read_side_closed = false;
		_write_side_closed = false;
	}

private:
	union
	{
		int _sides[2];
		struct
		{
			int _read_side;
			int _write_side;
		};
	};

	bool _read_side_closed;
	bool _write_side_closed;

	static void CloseSide(int fd, bool& closed)
	{
		if (closed) { return; }
		if (close(fd)) { LOG(severity::ERROR) << "Error closing pipe\n"; }
		//close(fd);
		closed = true;
	}
};


TempFile::TempFile(const string& path_template)
	: _path(GetTempDirectoryPath() + "/" + path_template + "-XXXXXX")
{
	_fd = mkstemp(&_path[0]);
	if (_fd == -1)
	{
		LOG(severity::ERROR) << "TempFile: mkstemp()\n";
		exit(-1);
	}
}

TempFile::~TempFile()
{
	const int remove_rc = std::remove(_path.c_str());
	const int close_rc = close(_fd);
	if (remove_rc != 0 || close_rc != 0)
		LOG(severity::ERROR) << "TempFile: On removal of '" << _path
		<< "', remove_rc = " << remove_rc << ", close_rc = " << close_rc << ".\n";
}

const string TempFile::GetTempDirectoryPath()
{
	return "/tmp";
}

int ExecuteProcess(const string& p, vector<string>& args, istream* in, ostream* out, ostream* err)
{
	Pipe asmsh_stdin;
	Pipe asmsh_stdout;
	Pipe asmsh_stderr;

	const auto redirect_stdin = (in != nullptr);
	const auto redirect_stdout = (out != nullptr);
	const auto redirect_stderr = (err != nullptr);

	if (redirect_stdin) { asmsh_stdin.Open(); }
	if (redirect_stdout) { asmsh_stdout.Open(); }
	if (redirect_stderr) { asmsh_stderr.Open(); }

	int wstatus;
	pid_t pid = fork();

	if (pid == 0) {
		string path(p); // to remove constness 
		vector<char*> c_args;
		c_args.push_back(&path[0]);
		for (auto& arg : args)
			c_args.push_back(&arg[0]);
		c_args.push_back(nullptr);

		if (redirect_stdin) {
			if (asmsh_stdin.DupRead(STDIN_FILENO) == -1) { std::exit(EXIT_FAILURE); }
			asmsh_stdin.Close();
		}

		if (redirect_stdout) {
			if (asmsh_stdout.DupWrite(STDOUT_FILENO) == -1) { std::exit(EXIT_FAILURE); }
			asmsh_stdout.Close();
		}

		if (redirect_stderr) {
			if (asmsh_stderr.DupWrite(STDERR_FILENO) == -1) { std::exit(EXIT_FAILURE); }
			asmsh_stderr.Close();
		}

		asmsh_stdout.Close();

		execvp(path.c_str(), c_args.data());
		LOG(severity::ERROR) << "EXT_PROC: unable to execute \"" << path << "\"\n";
		std::exit(EXIT_FAILURE);
	}
	else {
		if (pid == -1) { LOG(severity::ERROR) << "EXT_PROC: fork()\n"; exit(-1); }

		if (redirect_stdin)
		{
			asmsh_stdin.CloseRead();
			__gnu_cxx::stdio_filebuf<char> asmsh_stdin_buffer(asmsh_stdin.GetWriteFd(), std::ios::out);
			ostream asmsh_stdin_stream(&asmsh_stdin_buffer);
			asmsh_stdin_stream << in->rdbuf() << std::flush;
			asmsh_stdin.CloseWrite();
		}

		if (redirect_stdout)
		{
			asmsh_stdout.CloseWrite();
			__gnu_cxx::stdio_filebuf<char> asmsh_stdout_buffer(asmsh_stdout.GetReadFd(), std::ios::in);
			istream asmsh_stdin_stream(&asmsh_stdout_buffer);
			*out << asmsh_stdin_stream.rdbuf() << std::flush;
			asmsh_stdout.CloseRead();
		}

		if (redirect_stderr)
		{
			asmsh_stderr.CloseWrite();
			__gnu_cxx::stdio_filebuf<char> asmsh_stderr_buffer(asmsh_stderr.GetReadFd(), std::ios::in);
			istream asmsh_stdin_stream(&asmsh_stderr_buffer);
			*err << asmsh_stdin_stream.rdbuf() << std::flush;
			asmsh_stderr.CloseRead();
		}

		if (waitpid(pid, &wstatus, 0) != pid) { LOG(severity::ERROR) << "EXT_PROC: waitpid()\n"; exit(-1); }
	}

	if (WIFEXITED(wstatus)) {
		const int exit_status = WEXITSTATUS(wstatus);
		return exit_status;
	}
	else {
		LOG(severity::ERROR) << "EXT_PROC terminated abnormally\n";
		exit(-1);
	}
}

#endif