//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
#define CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_

#include <cassert>

#include "lox/ast/ast.h"
#include "lox/backend/tree_walker/evaluator/environment.h"
#include "lox/backend/tree_walker/evaluator/runtime_error.h"
#include "lox/object/object.h"
namespace lox::twalker {

/**
 * Evaluator's most evaluation are intuitive, only three utilities need some explanation:
 * 1. Environment: it is the runtime semantic scope. A newly created named value will be stored in the latest
 * environment. and the named value will be deleted when the environment is deleted. For Lox, only
 * ClassStmt/CallStmt/BlockStmt may create a new environment.
 * 2. Closure: All FunctionStmt will be converted to Closure at runtime, and every
 * Closure will close the env it was created, like a hard fork. When the closure is called, a new temporary env append
 * to the closed env to act as a new local env.
 * 3. Callable: Mainly to support native function and instance creation.
 *
 * Some feature are implemented based on c++ exception: `Return`,`Break` and `Continue`.
 *
 * Note that most semantic error had been caught by semantic checker, so evaluator will use static cast when
 * possible.
 */
class Evaluator : public AstNodeVisitor<ObjectPtr> {
 public:
  Evaluator();
  void LaunchScript(FunctionStmt* script);
  ObjectPtr Eval(ASTNode* node);
  void Error(const std::string& msg);
  EnvPtr WorkEnv() { return work_env_; }
  EnvPtr SwitchEnv(EnvPtr new_env);

 protected:
  void Visit(LogicalExpr* node) override;
  void Visit(BinaryExpr* node) override;
  void Visit(GroupingExpr* node) override;
  void Visit(LiteralExpr* state) override;
  void Visit(UnaryExpr* node) override;
  void Visit(VariableExpr* node) override;
  void Visit(AssignExpr* node) override;
  void Visit(CallExpr* node) override;
  void Visit(GetAttrExpr* node) override;
  void Visit(SetAttrExpr* node) override;
  void Visit(PrintStmt* node) override;
  void Visit(ReturnStmt* state) override;
  void Visit(WhileStmt* state) override;
  void Visit(ForStmt* node) override;
  void Visit(ExprStmt* state) override;
  void Visit(BreakStmt* state) override;
  void Visit(VarDeclStmt* state) override;
  void Visit(FunctionStmt* node) override;
  void Visit(ClassStmt* node) override;
  void Visit(BlockStmt* state) override;
  void Visit(IfStmt* state) override;

  void NumberBinaryOp(const BinaryExpr* node, ObjectPtr left, ObjectPtr right);
  void StringBinaryOp(const BinaryExpr* node, ObjectPtr left, ObjectPtr right);

  EnvPtr work_env_;
  ObjectPtr CreateClosure(FunctionStmt* function);
};

}  // namespace lox::twalker
#endif  // CPPLOX_SRCS_LOX_EVALUATOR_EVAL_VISITOR_H_
