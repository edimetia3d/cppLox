//
// LICENSE: MIT
//

#ifndef CPPLOX_SRCS_LOX_LOX_OBJECT_CALLABLE_OBJECT_H_
#define CPPLOX_SRCS_LOX_LOX_OBJECT_CALLABLE_OBJECT_H_

#include <map>
#include <vector>

#include "lox/lox_object/lox_object.h"
#include "lox/visitors/evaluator/evaluator.h"
namespace lox {
class LoxCallableState : public object::LoxObjectState {
 public:
  virtual int Arity() = 0;
  virtual object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) = 0;
};

class LoxCallable : public object::LoxObject {
 public:
  explicit LoxCallable(const object::LoxObject obj) : object::LoxObject(obj) {}
  bool IsValid();
  int Arity();
  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments);

 protected:
  LoxCallableState* ObjectState();
};

const std::map<std::string, object::LoxObject>& BuiltinCallables();
}  // namespace lox
#endif  // CPPLOX_SRCS_LOX_LOX_OBJECT_CALLABLE_OBJECT_H_
