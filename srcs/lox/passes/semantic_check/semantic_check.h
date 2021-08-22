//
// Created by edi on 8/21/21.
//

#ifndef CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
#define CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_

#include "lox/ast/ast.h"
#include "lox/passes/pass.h"
namespace lox {

class SemanticCheck : public Pass {
 public:
  explicit SemanticCheck() = default;
  void PreNode(AstNode* ast_node) override;
  void PostNode(AstNode* ast_node) override;

 protected:
  int while_loop_level = 0;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_PASSES_SEMANTIC_CHECK_SEMANTIC_CHECK_H_
