//
// LICENSE: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_LOX_H_
#define CPPLOX_INCLUDES_LOX_LOX_H_

#include <string>

namespace lox {

class Error {
 public:
  int TOErrCode();
  void Append(const Error &new_err);
};

class Lox {
 public:
  static std::string CLIHelpString();

  Error RunFile(const std::string &file_path);

  Error RunPrompt();

  /**
   * Multi line exec
   * @param code
   * @return
   */
  Error Run(const std::string &code);

 private:
  Error RunStream(std::istream *istream, bool interactive_mode);
  Error Eval(const std::string &code, std::string *eval_output);
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_LOX_H_
