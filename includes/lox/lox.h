//
// LICENSE: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_LOX_H_
#define CPPLOX_INCLUDES_LOX_LOX_H_

#include <memory>
#include <string>

#include "lox/lox_error.h"

namespace lox {
class BackEnd;
class LoxInterpreter {
 public:
  explicit LoxInterpreter(const std::string &backend_name = "TreeWalker");
  static std::string CLIHelpString();

  LoxError RunFile(const std::string &file_path);

  LoxError RunPrompt();

  /**
   * Multi line exec
   * @param code
   * @return
   */
  LoxError Eval(const std::string &code);

 private:
  LoxError RunStream(std::istream *istream);
  std::shared_ptr<BackEnd> back_end_;
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_LOX_H_
