//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_
#include "lox/frontend/ast/ast.h"
namespace lox {
/**
 * A pass could modify ast inplace , or return a new node to replace current node.
 * eg1. You could modify all print-stmt node's PrintStmt::expression to modify the printed value, this one could be a
 * inplace one eg2. You could do code injection to replace all print-stmt nodes with some new block-stmt, so you could
 * do more thing before/after print, in this case ,you should return a new node by set the `replace_node`
 *
 * Note that:
 * A Modified ast node will be processed by current pass again. So all pass must avoid endless recursion.
 * eg1. after you modified one ast-node, the pass will run on the ast-node again, until there is no modification nor new
 * node return eg2. you returned a new ast-node to replace the current one, the pass will run on the returned node
 * again, until there is no modification nor new node return
 */
struct Pass {
  /**
   * @param ast_node the input original node
   * @param replace_node a new node that will replace the input original node `replace_node->get()` is equal to
   * `ast_node` by default (self replace self, so replace nothing)
   */
  virtual void PreNode(AstNode* ast_node, std::shared_ptr<AstNode>* replace_node) = 0;
  virtual void PostNode(AstNode* ast_node, std::shared_ptr<AstNode>* replace_node) = 0;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_
