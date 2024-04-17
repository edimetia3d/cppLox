//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/tree_walker.h"

#include "lox/backend/tree_walker/evaluator/evaluator.h"

namespace lox::twalker {
TreeWalker::TreeWalker() { evaluator_ = std::make_shared<Evaluator>(); }
void TreeWalker::Run(Scanner &scanner) {
  auto lox_module = BuildASTModule(scanner);
  evaluator_->LaunchStmts(lox_module->Statements());
}

} // namespace lox::twalker
