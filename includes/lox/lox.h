//
// LICENSE: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_LOX_H_
#define CPPLOX_INCLUDES_LOX_LOX_H_

#include <string>

#include "lox/error.h"
#include "lox/evaluator/eval_visitor.h"

namespace lox {

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
  Error Eval(const std::string &code, std::string *run_output);

 private:
  Error RunStream(std::istream *istream, bool interactive_mode);
  AstEvaluator evaluator_;
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_LOX_H_
