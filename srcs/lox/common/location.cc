//
// License: MIT
//
#include "lox/common/location.h"

namespace lox {

int64_t lox::Location::Column() const {
  // Column are computed by counting the number of characters from the beginning of the line
  // Some implementation will even compute the line number at runtime
  if (!file_) {
    return -1;
  }
  int64_t pos_new_line = pos_in_file_ - 1;
  while (pos_new_line >= 0 && file_->At(pos_new_line) != '\n') {
    --pos_new_line;
  }
  return pos_in_file_ - pos_new_line - 1;
}

} // namespace lox