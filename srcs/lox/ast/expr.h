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
class Visitor;  // forward decl

/**
 * An abstract class
 */
class Expr {
 public:
  template <class RetT>
  static RetT CallAccept(const Visitor<RetT>& v, Expr& expr);

  template <class RetT>
  static RetT CallAccept(const Visitor<RetT>& v, const Expr& expr);

  virtual ~Expr() {
    // just make it virtual to support dynamic_cast
  }

 protected:
  template <class RetT>
  friend class Visitor;
  template <class RetT>
  RetT Accept(const Visitor<RetT>& v);

  template <class RetT>
  RetT Accept(const Visitor<RetT>& v) const;
};

using ExprPointer = std::shared_ptr<Expr>;
}  // namespace lox

#include "expr_decl.h.inc"

#endif  // CPPLOX_SRCS_LOX_AST_EXPR_H
