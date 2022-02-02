//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_
#define CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_
#include "lox/ast/ast.h"
namespace lox {
/**
 * A pass will walk the AST and perform some action on each node.
 *
 * The walking order is a root-first style traversal, but every node will be visited twice. The second run could access
 * the Node after itself or it's child might had been modified.
 *
 * 1. We Run PreNode on the node first.
 * 2. We Run PreNode and PostNode on all children of the node recursively.
 * 3. We Run PostNode on the node last.
 *
 * Note that:
 *
 * If a pass modifies the node's child in PreNode, the new child will be visited. If pass modifies the node's child in
 * PostNode, the new child will never be visited.
 *
 */
struct Pass {
  enum class IsModified {
    NO,
    YES,
  };

  virtual IsModified PreNode(ASTNode* ast_node) = 0;
  virtual IsModified PostNode(ASTNode* ast_node) = 0;
};

using PassSequence = std::vector<std::shared_ptr<Pass>>;

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_PASSES_PASS_H_
