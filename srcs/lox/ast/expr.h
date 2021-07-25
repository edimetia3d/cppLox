//
// License: MIT
//
//

#ifndef CPPLOX_SRCS_LOX_AST_EXPR_H
#define CPPLOX_SRCS_LOX_AST_EXPR_H

#include <memory>

#include "lox/token.h"

namespace lox {

template <class RetT>
class Visitor;

namespace private_ns {
class ExprImpl;
}

class Expr {
 public:
  /**
   * This fn will take ownership of object impl pointed to
   */
  explicit Expr(lox::private_ns::ExprImpl* impl) : impl(impl) {}
  explicit Expr(std::shared_ptr<lox::private_ns::ExprImpl> impl) : impl(std::move(impl)) {}
  explicit operator bool() { return static_cast<bool>(impl); }

  template <class RetT>
  RetT Accept(Visitor<RetT>* v);

 private:
  std::shared_ptr<lox::private_ns::ExprImpl> impl;
};
}  // namespace lox

#ifdef DYNAMIC_GEN_EXPR_DECL
#include "lox/ast/expr_decl_dynamic.h.inc"
#else
#include "lox/ast/expr_decl.h.inc"
#endif
#endif  // CPPLOX_SRCS_LOX_AST_EXPR_H
