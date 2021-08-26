//
// License: MIT
//
//

#ifndef CPPLOX_SRCS_LOX_AST_EXPR_H
#define CPPLOX_SRCS_LOX_AST_EXPR_H

#include "lox/backend/tree_walker/ast/ast_node.h"
namespace lox {

class ExprBase : public AstNode {
 protected:
  using AstNode::AstNode;
};

template <class T>
concept SubclassOfExpr = std::is_base_of<ExprBase, T>::value;

template <SubclassOfExpr SubT, class... Args>
std::shared_ptr<SubT> MakeExpr(Args... args) {
  return AstNode::Make<SubT, Args...>(args...);
}

}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_EXPR_H
