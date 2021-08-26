//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_LOX_OBJECT_CALLABLE_OBJECT_H_
#define CPPLOX_SRCS_LOX_LOX_OBJECT_CALLABLE_OBJECT_H_

#include <map>
#include <vector>

#include "lox/backend/tree_walker/lox_object/lox_object.h"
#include "lox/backend/tree_walker/visitors/evaluator/evaluator.h"
namespace lox {
class LoxCallable : public object::LoxObjectBase {
 public:
  virtual int Arity() = 0;
  virtual object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) = 0;
};
const std::map<std::string, object::LoxObject>& BuiltinCallables();
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_LOX_OBJECT_CALLABLE_OBJECT_H_
