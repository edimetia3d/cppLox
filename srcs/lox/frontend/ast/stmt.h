//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_STMT_H_
#define CPPLOX_SRCS_LOX_AST_STMT_H_

#include "lox/frontend/ast/ast_node.h"

namespace lox {

class StmtBase : public AstNode {
 protected:
  using AstNode::AstNode;
};

template <class T>
concept SubclassOfStmt = std::is_base_of<StmtBase, T>::value;

template <SubclassOfStmt SubT, class... Args>
std::shared_ptr<SubT> MakeStmt(Args... args) {
  return AstNode::Make<SubT, Args...>(args...);
}
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_AST_STMT_H_
