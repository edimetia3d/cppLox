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
class StmtEvaluator;
class ExprEvaluator : public ExprVisitor {
 public:
  explicit ExprEvaluator(std::shared_ptr<Environment> env, StmtEvaluator* parent)
      : work_env_(std::move(env)), parent_(parent) {}
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
  StmtEvaluator* parent_;
};

class StmtEvaluator : public StmtVisitor {
 public:
  explicit StmtEvaluator(std::shared_ptr<Environment> env) : expr_evaluator_(env, this) {}
  object::LoxObject Eval(Stmt stmt) {
    assert(stmt.IsValid());
    return stmt.Accept(this);
  }

  std::shared_ptr<Environment>& WorkEnv() { return expr_evaluator_.WorkEnv(); }
  std::shared_ptr<Environment> WorkEnv(std::shared_ptr<Environment> new_env) {
    return expr_evaluator_.WorkEnv(new_env);
  }

  struct EnterNewScopeGuard {
    EnterNewScopeGuard(StmtEvaluator* ev, std::shared_ptr<Environment> new_work_env = nullptr) : evaluator(ev) {
      if (!new_work_env) {
        new_work_env = std::make_shared<Environment>(evaluator->WorkEnv());
      }
      backup = evaluator->WorkEnv(new_work_env);
    }
    ~EnterNewScopeGuard() { evaluator->WorkEnv(backup); }
    std::shared_ptr<Environment> backup;
    StmtEvaluator* evaluator;
  };

 protected:
  object::LoxObject Visit(PrintStmtState* state) override;
  object::LoxObject Visit(ReturnStmtState* state) override;
  object::LoxObject Visit(WhileStmtState* state) override;
  object::LoxObject Visit(ExprStmtState* state) override;
  object::LoxObject Visit(BreakStmtState* state) override;
  object::LoxObject Visit(VarDeclStmtState* state) override;
  object::LoxObject Visit(FunctionStmtState* state) override;
  object::LoxObject Visit(BlockStmtState* state) override;
  object::LoxObject Visit(IfStmtState* state) override;
  ExprEvaluator expr_evaluator_;
};

}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
