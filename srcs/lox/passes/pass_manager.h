//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_MANAGER_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_MANAGER_H_
#include "lox/ast/ast.h"
#include "pass.h"
#include "pass_runner.h"
namespace lox {
class PassManager {
 public:
  void Append(std::shared_ptr<Pass> pass) { passes_.push_back(pass); }
  std::vector<Stmt> Run(std::vector<Stmt> stmts);

 private:
  std::vector<std::shared_ptr<Pass>> passes_;
  std::shared_ptr<PassRunner> runner_{new PassRunner};
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_MANAGER_H_
