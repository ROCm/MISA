#ifdef WIN32

#include "utils.hpp"
#include "log.hpp"
#include <cassert>
#include <Windows.h>
#undef ERROR
#include <io.h>

TempFile::TempFile(const string& path_template)
	: _path(GetTempDirectoryPath() + "\\" + path_template + "-XXXXXX")
{
	if (_mktemp(&_path[0]) == NULL)
	{
		LOG(severity::ERROR) << "TempFile: GetTempFileName()\n";
		std::exit(EXIT_FAILURE);
	}
	// _mktemp returns the same path for consecutive calls if the file does not exist, however _mktemp does not create the file by itself, so we need to do it manually to "reserve" the path
	HANDLE file = CreateFile(_path.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (file == INVALID_HANDLE_VALUE)
	{
		LOG(severity::ERROR) << "TempFile: CreateFile()\n";
		std::exit(EXIT_FAILURE);
	}
	CloseHandle(file);
}

TempFile::~TempFile()
{
	const int remove_rc = std::remove(_path.c_str());
	if (remove_rc != 0)
		LOG(severity::ERROR) << "TempFile: On removal of '" << _path << "', remove_rc = " << remove_rc << ".\n";
}

const string TempFile::GetTempDirectoryPath()
{
	char tmp_dir_path[MAX_PATH];
	if (!GetTempPath(MAX_PATH, tmp_dir_path))
	{
		LOG(severity::ERROR) << "TempFile: GetTempPath()\n";
		std::exit(EXIT_FAILURE);
	}
	return std::string(tmp_dir_path);
}

void ExecuteProcessCreatePipe(HANDLE* pipe_read, HANDLE* pipe_write)
{
	SECURITY_ATTRIBUTES sa;
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.lpSecurityDescriptor = NULL;
	sa.bInheritHandle = true;

	if (!CreatePipe(pipe_read, pipe_write, &sa, 0) || !pipe_read || !pipe_write)
	{
		DWORD error = GetLastError();
		LOG(severity::ERROR) << "EXT_PROC: CreatePipe(), GetLastError() = " << error << "\n";
		std::exit(EXIT_FAILURE);
	}
}

int ExecuteProcess(const string& p, vector<string>& args, istream* in, ostream* out, ostream* err)
{
	HANDLE stdin_rd, stdin_wr, stdout_rd, stdout_wr, stderr_rd, stderr_wr;

	const auto redirect_stdin = (in != nullptr);
	const auto redirect_stdout = (out != nullptr);
	const auto redirect_stderr = (err != nullptr);

	if (redirect_stdin) { ExecuteProcessCreatePipe(&stdin_rd, &stdin_wr); SetHandleInformation(stdin_wr, HANDLE_FLAG_INHERIT, 0); }
	if (redirect_stdout) { ExecuteProcessCreatePipe(&stdout_rd, &stdout_wr); SetHandleInformation(stdout_rd, HANDLE_FLAG_INHERIT, 0); }
	if (redirect_stderr) { ExecuteProcessCreatePipe(&stderr_rd, &stderr_wr); SetHandleInformation(stderr_rd, HANDLE_FLAG_INHERIT, 0); }

	STARTUPINFO si;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	si.dwFlags |= STARTF_USESTDHANDLES;
	if (redirect_stdin) si.hStdInput = stdin_rd;
	if (redirect_stdout) si.hStdOutput = stdout_wr;
	if (redirect_stderr) si.hStdError = stderr_wr;

	PROCESS_INFORMATION pi;
	ZeroMemory(&pi, sizeof(pi));

	std::string cmd_line = p;
	for (auto& arg : args)
	{
		cmd_line.push_back(' ');
		cmd_line.push_back('"');
		cmd_line.append(arg);
		cmd_line.push_back('"');
	}

	if (!CreateProcess(
		NULL,                    // executable, set to null so we can redirect standard streams
		(LPSTR)cmd_line.c_str(), // command line
		NULL,                    // process security attributes
		NULL,                    // primary thread security attributes
		true,                    // inherit handles
		CREATE_NO_WINDOW,        // creation flags
		NULL,                    // use parent's environment
		NULL,                    // use parent's current directory
		&si, &pi))
	{
		LOG(severity::ERROR) << "EXT_PROC: unable to execute \"" << p << "\"\n";
		std::exit(EXIT_FAILURE);
	}

	if (redirect_stdin) CloseHandle(stdin_rd);
	if (redirect_stdout) CloseHandle(stdout_wr);
	if (redirect_stderr) CloseHandle(stderr_wr);

	if (redirect_stdin)
	{
		char buffer[4096];
		while (!in->eof())
		{
			in->read(buffer, 4096);
			DWORD bytes_read = in->gcount();
			if (!WriteFile(stdin_wr, buffer, bytes_read, NULL, NULL))
			{
				DWORD error = GetLastError();
				LOG(severity::ERROR) << "EXT_PROC: WriteFile(), GetLastError() = " << error << "\n";
				std::exit(EXIT_FAILURE);
			}
		}
		CloseHandle(stdin_wr);
	}

	if (redirect_stdout)
	{
		char buffer[4096];
		DWORD bytes_read = 0;
		while (ReadFile(stdout_rd, buffer, 4096, &bytes_read, NULL))
		{
			if (bytes_read > 0)
				out->write(buffer, bytes_read);
		}
		CloseHandle(stdout_rd);
	}

	if (redirect_stderr)
	{
		char buffer[4096];
		DWORD bytes_read = 0;
		while (ReadFile(stderr_rd, buffer, 4096, &bytes_read, NULL))
		{
			if (bytes_read > 0)
				err->write(buffer, bytes_read);
		}
		CloseHandle(stderr_rd);
	}

	DWORD exit_code;
	do
	{
		if (!GetExitCodeProcess(pi.hProcess, &exit_code))
		{
			DWORD error = GetLastError();
			LOG(severity::ERROR) << "EXT_PROC: GetExitCodeProcess(), GetLastError() = " << error << "\n";
			std::exit(EXIT_FAILURE);
		}
	} while (exit_code == STILL_ACTIVE);

	CloseHandle(pi.hThread);
	CloseHandle(pi.hProcess);

	return exit_code;
}

int setenv(const char* name, const char* value, int overwrite)
{
	assert(overwrite); // _putenv_s overwrites existing variables, matching the behavior of overwrite=0 would require checking if the variable exists first
	return _putenv_s(name, value);
}

#endif