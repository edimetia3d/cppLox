//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
#include <cassert>

#include "lox/ast/ast.h"
#include "pass.h"
namespace lox {
class PassRunner {
 public:
  PassRunner() = default;
  void SetPass(PassSequence passes) { passes_ = passes; }
  lox::Pass::IsModified Run(ASTNode* node);

 private:
  lox::Pass::IsModified RunPass(ASTNode* node, Pass* pass);
  PassSequence passes_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_RUNNER_H_
