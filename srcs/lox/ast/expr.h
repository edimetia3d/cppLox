//
// License: MIT
//
//

#ifndef CPPLOX_SRCS_LOX_AST_EXPR_H
#define CPPLOX_SRCS_LOX_AST_EXPR_H

#include <memory>

#include "lox/token.h"

namespace lox {

class ExprState;

class Expr {
 public:
  /**
   * This fn will take ownership of object state_ pointed to
   */
  Expr(){};
  explicit Expr(ExprState* state) : state_(state) {}
  explicit Expr(std::shared_ptr<ExprState> state) : state_(std::move(state)) {}
  explicit operator bool() { return static_cast<bool>(state_); }

  std::shared_ptr<ExprState> State() { return state_; }

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
