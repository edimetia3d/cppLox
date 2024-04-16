//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_STRING_REF_H_
#define CPPLOX_SRCS_LOX_STRING_REF_H_

#include <cstring>
#include <string>

namespace lox {
class RefString {
public:
  RefString() : data_(nullptr), length_(0) {}
  RefString(const char *data, size_t length) : data_(data), length_(length) {}
  RefString(const char *beg, const char *end) : data_(beg), length_(end - beg) {}
  RefString(const char *c_str) : data_(c_str), length_(strlen(c_str)) {}
  RefString(const std::string &str) : data_(str.data()), length_(str.size()) {}

  const char *Data() const { return data_; }
  const char *End() const { return data_ + length_; }
  size_t Length() const { return length_; }

  bool operator==(const RefString &rhs) const {
    if (length_ != rhs.length_) {
      return false;
    }
    return std::equal(data_, data_ + length_, rhs.data_);
  }

  bool operator==(const char *c_str) {
    size_t len = strlen(c_str);
    if (length_ != len) {
      return false;
    }
    return std::equal(data_, data_ + length_, c_str);
  }

  bool operator<(const RefString &rhs) const {
    return std::lexicographical_compare(data_, data_ + length_, rhs.data_, rhs.data_ + rhs.length_);
  }

  std::string Str() const { return std::string(data_, length_); }

private:
  const char *data_ = nullptr;
  size_t length_ = 0;
};
} // namespace lox
#endif // CPPLOX_SRCS_LOX_STRING_REF_H_
