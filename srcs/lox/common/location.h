//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_LOCATION_H_
#define CPPLOX_SRCS_LOX_LOCATION_H_

#include <memory>
#include <string>

#include "lox/common/input_file.h"
namespace lox {

class Location {
public:
  Location(std::shared_ptr<CharStream> file, int64_t line, int64_t pos_in_file)
      : file_(std::move(file)), line_(line), pos_in_file_(pos_in_file) {}
  int64_t Line() const { return line_; }
  int64_t Column() const;
  int64_t PosInFile() const { return pos_in_file_; }
  std::string FileName() const {
    if (!file_) {
      return "Unknown";
    }
    return file_->Name();
  }

  std::string OneLineStr() const {
    // format "file:line:column"
    return FileName() + ":" + std::to_string(line_) + ":" + std::to_string(Column());
  }

private:
  std::shared_ptr<CharStream> file_;
  int64_t line_ = 0;
  int64_t pos_in_file_ = 0;
};
} // namespace lox
#endif // CPPLOX_SRCS_LOX_LOCATION_H_
