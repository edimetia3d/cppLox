//
// License: MIT
//
//

#ifndef CPPLOX_SRCS_LOX_AST_EXPR_H
#define CPPLOX_SRCS_LOX_AST_EXPR_H

#include <memory>
#include <vector>

#include "lox/lox_object/lox_object.h"
#include "lox/token.h"

namespace lox {

class ExprState;
class ExprVisitor;
class Expr {
 public:
  /**
   * This fn will take ownership of object state_ pointed to
   */
  explicit Expr(ExprState* state) : state_(state) {}
  explicit Expr(std::shared_ptr<ExprState> state) : state_(std::move(state)) {}
  bool IsValid() { return static_cast<bool>(state_); }

  object::LoxObject Accept(ExprVisitor* visitor);

  template <class T>
  T* DownCastState() {
    return dynamic_cast<T*>(state_.get());
  }

 private:
  std::shared_ptr<ExprState> state_;
};
}  // namespace lox

#ifdef DYNAMIC_GEN_DECL
#include "lox/ast/expr_decl_dynamic.h.inc"
#else
#include "lox/ast/expr_decl.h.inc"
#endif
#endif  // CPPLOX_SRCS_LOX_AST_EXPR_H
