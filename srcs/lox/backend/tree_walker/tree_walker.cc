//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/tree_walker.h"

#include "lox/backend/tree_walker/error.h"
#include "lox/backend/tree_walker/evaluator/callable_object.h"
#include "lox/backend/tree_walker/evaluator/environment.h"
#include "lox/backend/tree_walker/evaluator/evaluator.h"
#include "lox/backend/tree_walker/passes/env_resolve_pass/env_reslove_pass.h"
#include "lox/backend/tree_walker/passes/env_resolve_pass/resolve_map.h"
#include "lox/frontend/ast/ast_printer/ast_printer.h"
#include "lox/frontend/parser.h"
#include "lox/frontend/passes/pass_manager.h"
#include "lox/frontend/passes/semantic_check/semantic_check.h"
#include "lox/global_setting.h"

namespace lox {
TreeWalker::TreeWalker() {
  global_env_ = Environment::Make();
  for (auto it : BuiltinCallables()) {
    global_env_->Define(it.first, it.second);
  }
  resolve_map_ = std::make_shared<EnvResolveMap>();
  evaluator_ = std::make_shared<Evaluator>(global_env_);
  evaluator_->SetActiveResolveMap(resolve_map_);
}
LoxError TreeWalker::Run(Scanner &scanner) {
  std::vector<Token> tokens;
  auto err = scanner.ScanAll(&tokens);
  Parser parser(tokens);
  auto statements = parser.Parse();

  PassManager pass_mgr;
  pass_mgr.Append(std::make_shared<SemanticCheck>());
  pass_mgr.Append(std::make_shared<EnvResovlePass>(resolve_map_));
  try {
    statements = pass_mgr.Run(statements);
  } catch (std::exception &pass_err) {
    std::cout << pass_err.what() << std::endl;
    return LoxError();
  }

  try {
    for (auto &stmt : statements) {
      if (GlobalSetting().debug) {
        static AstPrinter printer;
        std::cout << "Debug Stmt: `" << printer.Print(stmt) << "`" << std::endl;
      }
      evaluator_->Eval(stmt);
    }
  } catch (RuntimeError &rt_err) {
    std::cout << rt_err.what() << std::endl;
    err.Merge(rt_err);
  }
  return LoxError();
}

}  // namespace lox
