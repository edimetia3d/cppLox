//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_H_
#define CPPLOX_SRCS_LOX_AST_AST_H_
#include <cassert>
#include <stack>

#include "lox/ast/_ast_node.h"
#include "lox/ast/ast_nodes_decl.h.inc"

namespace lox {

class Module {
public:
  Module() = default;
  explicit Module(std::vector<StmtPtr> &&statements) {
    group_ = ASTNode::Make<lox::BlockStmt>(BlockStmtAttr{}, std::move(statements));
  }
  std::vector<StmtPtr> &Statements() { return ViewAsBlock()->statements; }

  BlockStmt *ViewAsBlock() { return group_->As<BlockStmt>(); }

private:
  StmtPtr group_;
};

#define VisitorReturn(arg)                                                                                             \
  do {                                                                                                                 \
    Return(arg);                                                                                                       \
    return;                                                                                                            \
  } while (0)
template <class T> class ASTNodeVisitor : public IASTNodeVisitor {
public:
  void Return(T &&new_ret) {
    assert(ret_stk_.size() == 0);
    ret_stk_.push(std::move(new_ret));
  }
  void Return(const T &new_ret) {
    assert(ret_stk_.size() == 0);
    ret_stk_.push(new_ret);
  }

  void NoValueVisit(ASTNode *node) { node->Accept(this); }

  void NoValueVisit(ASTNodePtr &node) { return NoValueVisit(node.get()); }

  T ValueVisit(ExprPtr &node) { return ValueVisit(node.get()); }

  T ValueVisit(ASTNode *node) {
    node->Accept(this);
    return PopRet();
  }

private:
  std::stack<T> ret_stk_; // though it should be a stack, for visitor , it will at most contains 1 element.
  T PopRet() {
    assert(ret_stk_.size() == 1);
    auto ret = ret_stk_.top();
    ret_stk_.pop();
    return ret;
  }
};
} // namespace lox
#endif // CPPLOX_SRCS_LOX_AST_AST_H_
