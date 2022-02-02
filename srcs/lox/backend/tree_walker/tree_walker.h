//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_BACKEND_TREE_WALKER_TREE_WALKER_H_
#define CPPLOX_SRCS_LOX_BACKEND_TREE_WALKER_TREE_WALKER_H_
#include <memory>

#include "lox/backend/backend.h"
#include "lox/frontend/scanner.h"

namespace lox::twalker {
class Evaluator;
class TreeWalker : public BackEnd {
 public:
  TreeWalker();
  void Run(Scanner& scanner) override;

 private:
  std::shared_ptr<Evaluator> evaluator_;
};
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_BACKEND_TREE_WALKER_TREE_WALKER_H_
