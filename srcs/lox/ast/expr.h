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
class ExprImpl {
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

  virtual ~ExprImpl() {
    // just make ExprImpl a virtual class to support dynamic_cast
  }

 protected:
  template <class RetT>
  friend class Visitor;
  template <class RetT>
  RetT _Accept(const Visitor<RetT>& v);

  template <class RetT>
  RetT _Accept(const Visitor<RetT>& v) const;
};

using Expr = std::shared_ptr<ExprImpl>;
}  // namespace lox

#ifdef DYNAMIC_GEN_EXPR_DECL
#include "lox/ast/expr_decl_dynamic.h.inc"
#else
#include "lox/ast/expr_decl.h.inc"
#endif
#endif  // CPPLOX_SRCS_LOX_AST_EXPR_H
