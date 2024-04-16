//
// License: MIT
//
#include "lox/common/input_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace lox {
lox::InputFile::InputFile(const std::string &filename) : fd_(open(filename.c_str(), O_RDONLY)) {
  // use mmap to lazy load file
  if (fd_ == -1) {
    return;
  }
  auto file_size = lseek(fd_, 0, SEEK_END);
  if (file_size == -1) {
    SafeCloseFd();
    return;
  }
  auto *data = mmap(nullptr, 0, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (data == MAP_FAILED) {
    SafeCloseFd();
    return;
  }
  Reload(filename, static_cast<const char *>(data), file_size);
}
lox::InputFile::~InputFile() {
  if (fd_ != -1) {
    munmap(const_cast<char *>(Data()), Size());
    SafeCloseFd();
  }
}
void lox::InputFile::SafeCloseFd() {
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}
lox::InputString::InputString(const char *data, int64_t size, bool copy_in) : data_(data), size_(size) {
  if (copy_in) {
    holder_.assign(data, size);
    data_ = holder_.c_str();
  }
  Reload("string", data_, size_);
}
} // namespace lox
