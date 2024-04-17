//
// LICENSE: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_LOX_H_
#define CPPLOX_INCLUDES_LOX_LOX_H_

#include "lox/common/input_file.h"
#include <memory>
#include <string>

namespace lox {
class BackEnd;
class LoxInterpreter {
 public:
  explicit LoxInterpreter();
  static std::string CLIHelpString();

  void RunFile(const std::string &file_path);

  void RunPrompt();

  void Eval(std::shared_ptr<CharStream> input);

private:
  std::shared_ptr<BackEnd> back_end_;
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_LOX_H_
