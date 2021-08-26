//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_MANAGER_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_MANAGER_H_
#include "lox/backend/tree_walker/ast/ast.h"
#include "lox/backend/tree_walker/passes/pass.h"
#include "lox/backend/tree_walker/passes/pass_runner.h"
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
