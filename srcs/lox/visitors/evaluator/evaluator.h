//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include <cassert>

#include "environment.h"
#include "lox/ast/expr.h"
#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"
namespace lox {
class ExprEvaluator : public ExprVisitor {
 public:
  explicit ExprEvaluator(std::shared_ptr<Environment> env) : work_env_(std::move(env)) {}
  object::LoxObject Eval(Expr expr) {
    assert(expr.IsValid());
    return expr.Accept(this);
  }
  std::shared_ptr<Environment>& WorkEnv() { return work_env_; }
  std::shared_ptr<Environment> WorkEnv(std::shared_ptr<Environment> new_env) {
    auto old_env = work_env_;
    work_env_ = new_env;
    return old_env;
  }

 protected:
  object::LoxObject Visit(LogicalState* state) override;
  object::LoxObject Visit(BinaryState* state) override;
  object::LoxObject Visit(GroupingState* state) override;
  object::LoxObject Visit(LiteralState* state) override;
  object::LoxObject Visit(UnaryState* state) override;
  object::LoxObject Visit(VariableState* state) override;
  object::LoxObject Visit(AssignState* state) override;
  object::LoxObject Visit(CallState* state) override;
  std::shared_ptr<Environment> work_env_;
};

class StmtEvaluator : public StmtVisitor {
 public:
  explicit StmtEvaluator(std::shared_ptr<Environment> env) : expr_evaluator_(env) {}
  object::LoxObject Eval(Stmt stmt) {
    assert(stmt.IsValid());
    return stmt.Accept(this);
  }

  std::shared_ptr<Environment>& WorkEnv() { return expr_evaluator_.WorkEnv(); }
  std::shared_ptr<Environment> WorkEnv(std::shared_ptr<Environment> new_env) {
    return expr_evaluator_.WorkEnv(new_env);
  }

 protected:
  object::LoxObject Visit(PrintStmtState* state) override;
  object::LoxObject Visit(WhileStmtState* state) override;
  object::LoxObject Visit(ExprStmtState* state) override;
  object::LoxObject Visit(BreakStmtState* state) override;
  object::LoxObject Visit(VarDeclStmtState* state) override;
  object::LoxObject Visit(BlockStmtState* state) override;
  object::LoxObject Visit(IfStmtState* state) override;
  ExprEvaluator expr_evaluator_;

  struct EnterNewScopeGuard {
    EnterNewScopeGuard(StmtEvaluator* ev) : evaluator(ev) {
      backup = evaluator->WorkEnv(std::make_shared<Environment>(evaluator->WorkEnv()));
    }
    ~EnterNewScopeGuard() { evaluator->WorkEnv(backup); }
    std::shared_ptr<Environment> backup;
    StmtEvaluator* evaluator;
  };
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
