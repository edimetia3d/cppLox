//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_INPUT_FILE_H_
#define CPPLOX_SRCS_LOX_INPUT_FILE_H_

#include <string>

namespace lox {
class CharStream {
public:
  CharStream() = default;
  virtual ~CharStream() = default;
  void Reload(const std::string &name, const char *data, int64_t size) {
    name_ = name;
    data_ = data;
    size_ = size;
  }
  char Read() { return data_[current_pos_++]; }

  char Peek() const { return data_[current_pos_]; }

  char At(const int64_t pos) const { return data_[pos]; }

  void Reset() { current_pos_ = 0; }

  void Seek(const int pos) { current_pos_ = pos; }

  void Skip(const int n) { current_pos_ += n; }

  const char *Data() const { return data_; }

  int64_t Size() const { return size_; }

  int64_t Pos() const { return current_pos_; }

  std::string Name() const { return name_; }

  bool IsAtEnd() { return current_pos_ >= size_; }

private:
  std::string name_;
  const char *data_ = nullptr;
  int64_t current_pos_ = 0;
  int64_t size_ = 0;
};

class InputFile final : public CharStream {
public:
  explicit InputFile(const std::string &filename);

  ~InputFile() override;

private:
  void SafeCloseFd();
  int fd_ = -1;
};

class InputString : public CharStream {
public:
  InputString(const char *data, int64_t size, bool copy_in = false);

private:
  const char *data_ = nullptr;
  int64_t size_ = 0;
  std::string holder_;
};
} // namespace lox
#endif // CPPLOX_SRCS_LOX_INPUT_FILE_H_
