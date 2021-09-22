//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_AST_AST_NODE_VISITOR_H_
#define LOX_SRCS_LOX_AST_AST_NODE_VISITOR_H_
#include <stack>

#include "lox/ast/ast.h"

namespace lox {
#define VisitorReturn(arg) \
  _Return(arg);            \
  return;
template <class T>
class AstNodeVisitor : public IAstNodeVisitor {
 public:
  void _Return(T&& new_ret) {
    assert(ret_stk_.size() == 0);
    ret_stk_.push(std::move(new_ret));
  }
  void _Return(const T& new_ret) {
    assert(ret_stk_.size() == 0);
    ret_stk_.push(std::move(new_ret));
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
#endif  // LOX_SRCS_LOX_AST_AST_NODE_VISITOR_H_
