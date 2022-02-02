//
// LICENSE: MIT
//
#ifndef CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
#define CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_

#include "lox/passes/pass.h"
namespace lox {

class SemanticCheck : public Pass {
 public:
  explicit SemanticCheck() = default;
  IsModified PreNode(ASTNode* ast_node) override;
  IsModified PostNode(ASTNode* ast_node) override;

 protected:
  int while_loop_level = 0;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
