//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/tree_walker.h"

#include "lox/backend/tree_walker/parser.h"
#include "lox/backend/tree_walker/passes/env_resolve_pass/env_reslove_pass.h"
#include "lox/backend/tree_walker/passes/env_resolve_pass/resolve_map.h"
#include "lox/backend/tree_walker/passes/pass_manager.h"
#include "lox/backend/tree_walker/passes/semantic_check/semantic_check.h"
#include "lox/backend/tree_walker/visitors/ast_printer/ast_printer.h"
#include "lox/backend/tree_walker/visitors/evaluator/callable_object.h"
#include "lox/backend/tree_walker/visitors/evaluator/environment.h"
#include "lox/backend/tree_walker/visitors/evaluator/evaluator.h"
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
BackEndErrCode TreeWalker::Run(Scanner &scanner) {
  std::vector<Token> tokens;
  auto err = scanner.ScanAll(&tokens);
  Parser parser(tokens);
  auto statements = parser.Parse();

  PassManager pass_mgr;
  pass_mgr.Append(std::make_shared<SemanticCheck>());
  pass_mgr.Append(std::make_shared<EnvResovlePass>(resolve_map_));
  try {
    statements = pass_mgr.Run(statements);
  } catch (std::exception &err) {
    std::cout << err.what() << std::endl;
    return BackEndErrCode::NO_ERROR;
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
    return BackEndErrCode::NO_ERROR;
  }
}

}  // namespace lox
