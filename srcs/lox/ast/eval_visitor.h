//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include <cassert>

#include "lox/ast/expr.h"
#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"
namespace lox {
class ExprEvaluator : public ExprVisitor {
 public:
  object::LoxObject Eval(Expr expr) {
    assert(expr.State());
    return expr.State()->Accept(this);
  }

 protected:
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
  object::LoxObject Visit(VariableState* state) override;
};

class StmtEvaluator : public StmtVisitor {
 public:
  object::LoxObject Eval(Stmt stmt) {
    assert(stmt.State());
    return stmt.State()->Accept(this);
  }

 protected:
  object::LoxObject Visit(PrintStmtState* state) override;
  object::LoxObject Visit(ExprStmtState* state) override;
  object::LoxObject Visit(VarDeclStmtState* state) override;

 private:
  ExprEvaluator expr_evaluator_;
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
