
#include "lox/ast/stmt.h"

namespace lox {

object::LoxObject Stmt::Accept(StmtVisitor *visitor) { return state_->Accept(visitor); }
}  // namespace lox
