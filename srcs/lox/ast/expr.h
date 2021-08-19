//
// License: MIT
//
//

#ifndef CPPLOX_SRCS_LOX_AST_EXPR_H
#define CPPLOX_SRCS_LOX_AST_EXPR_H

#include <memory>
#include <type_traits>
#include <vector>

#include "lox/lox_object/lox_object.h"
#include "lox/token.h"

namespace lox {

class ExprBase;
using Expr = std::shared_ptr<ExprBase>;
template <class T>
concept SubclassOfExprState = std::is_base_of<ExprBase, T>::value;

class ExprVisitor;
class ExprBase {
 public:
  template <SubclassOfExprState SubT, class... Args>
  static std::shared_ptr<SubT> Make(Args... args) {
    return std::shared_ptr<SubT>(new SubT(args...));
  }

  virtual ~ExprBase() {}

  virtual object::LoxObject Accept(ExprVisitor* visitor) = 0;

  template <SubclassOfExprState T>
  T* DownCastState() {
    return dynamic_cast<T*>(this);
  }
};

static inline bool IsValid(const Expr& expr) { return expr.get(); }

template <SubclassOfExprState SubT, class... Args>
std::shared_ptr<SubT> MakeExpr(Args... args) {
  return ExprBase::Make<SubT, Args...>(args...);
}

}  // namespace lox

#ifdef DYNAMIC_GEN_DECL
#include "lox/ast/expr_decl_dynamic.h.inc"
#else
#include "lox/ast/expr_decl.h.inc"
#endif
#endif  // CPPLOX_SRCS_LOX_AST_EXPR_H
