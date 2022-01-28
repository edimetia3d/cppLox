//
// LICENSE: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_LOX_H_
#define CPPLOX_INCLUDES_LOX_LOX_H_

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

  void Eval(const std::string &code, const std::string &file_name = "Unknown file");

 private:
  void RunStream(std::istream *istream, const std::string &file_name);
  std::shared_ptr<BackEnd> back_end_;
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_LOX_H_
