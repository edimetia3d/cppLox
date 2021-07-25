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
#define VIRTUAL_FUNCTION  // Just remind you that it is a virtual function
  /**
   * Note that expr.Accept()` will be called with the derived version.
   */
  template <class RetT>
  VIRTUAL_FUNCTION RetT Accept(const Visitor<RetT>& v);

  template <class RetT>
  VIRTUAL_FUNCTION RetT Accept(const Visitor<RetT>& v) const;
#undef VIRTUAL_FUNCTION

  virtual ~Expr() {
    // just make Expr a virtual class to support dynamic_cast
  }

 protected:
  template <class RetT>
  friend class Visitor;
  template <class RetT>
  RetT _Accept(const Visitor<RetT>& v);

  template <class RetT>
  RetT _Accept(const Visitor<RetT>& v) const;
};

using ExprPointer = std::shared_ptr<Expr>;
}  // namespace lox

#include "lox/ast/expr_decl.h.inc"

#endif  // CPPLOX_SRCS_LOX_AST_EXPR_H
