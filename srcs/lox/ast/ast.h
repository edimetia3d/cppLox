//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_AST_AST_H_
#define CPPLOX_SRCS_LOX_AST_AST_H_
#include <stack>

#include "lox/ast/ast_node.h"
#include "lox/ast/ast_nodes_decl.h.inc"

namespace lox {
#define VisitorReturn(arg) \
  Return(arg);             \
  return
template <class T>
class AstNodeVisitor : public IASTNodeVisitor {
 public:
  void Return(T&& new_ret) {
    assert(ret_stk_.size() == 0);
    ret_stk_.push(std::move(new_ret));
  }
  void Return(const T& new_ret) {
    assert(ret_stk_.size() == 0);
    ret_stk_.push(new_ret);
  }
  T PopRet() {
    assert(ret_stk_.size() == 1);
    auto ret = ret_stk_.top();
    ret_stk_.pop();
    return ret;
  }

 protected:
  std::stack<T> ret_stk_;  // though it should be a stack, for visitor , it will at most contains 1 element.
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_AST_AST_H_
