//
// LICENSE: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_LOX_H_
#define CPPLOX_INCLUDES_LOX_LOX_H_

#include <string>

#include "lox/error.h"

namespace lox {
class StmtEvaluator;
class Environment;
class Lox {
 public:
  Lox();
  static std::string CLIHelpString();

  Error RunFile(const std::string &file_path);

  Error RunPrompt();

  /**
   * Multi line exec
   * @param code
   * @return
   */
  void Eval(const std::string &code);

 private:
  Error RunStream(std::istream *istream);
  std::shared_ptr<StmtEvaluator> evaluator_;
  std::shared_ptr<Environment> global_env_;
};

}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_LOX_H_
