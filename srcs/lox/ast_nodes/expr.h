//
// License: MIT
//
//

#ifndef CPPLOX_EXPR_H
#define CPPLOX_EXPR_H

#include "lox/token.h"

namespace lox {
template <class RetT>
class Visitor {};
/**
 * An abstract class
 */
class Expr {
  template <class RetT>
  RetT accept(const Visitor<RetT>& v);
};

using ExprPointer = std::shared_ptr<Expr>;
}  // namespace lox

#include "lox/ast_nodes/expr_decl.h.inc"

#endif  // CPPLOX_EXPR_H
