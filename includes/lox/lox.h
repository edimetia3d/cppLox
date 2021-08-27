//
// LICENSE: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_LOX_H_
#define CPPLOX_INCLUDES_LOX_LOX_H_

#include <memory>
#include <string>

namespace lox {
enum class InterpreterError { NO_ERROR, BACKEND_ERROR };
class BackEnd;
class LoxInterpreter {
 public:
  explicit LoxInterpreter(const std::string &backend_name = "TreeWalker");
  static std::string CLIHelpString();

  InterpreterError RunFile(const std::string &file_path);

  InterpreterError RunPrompt();

  /**
   * Multi line exec
   * @param code
   * @return
   */
  InterpreterError Eval(const std::string &code);

 private:
  InterpreterError RunStream(std::istream *istream);
  std::shared_ptr<BackEnd> back_end_;
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_LOX_H_
