//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_H_
#define CPPLOX_SRCS_LOX_H_

#include <string>

namespace lox {

class Error {};

class Lox {
 public:
  static std::string CLIHelpString();

  static Error RunFile(std::string file_path);

  static Error RunPrompt();
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_H_
