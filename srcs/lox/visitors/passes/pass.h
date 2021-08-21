//
// Created by edi on 8/21/21.
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_

namespace lox {
struct Pass {
  virtual void PreNode(AstNode* ast_node) = 0;
  virtual void PostNode(AstNode* ast_node) = 0;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_