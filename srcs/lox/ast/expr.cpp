
#include "lox/ast/expr.h"

namespace lox {

object::LoxObject Expr::Accept(ExprVisitor *visitor) { return state_->Accept(visitor); }
}  // namespace lox
