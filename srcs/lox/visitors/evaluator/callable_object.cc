//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/callable_object.h"

#include <time.h>

#include "lox/lox_object/lox_object_state.h"
namespace lox {

class Clock : public LoxCallableState {
 public:
  int Arity() override { return 0; }
  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) {
    double clk = clock();
    object::LoxObject();
    return object::LoxObject(clk);
  }
  std::string ToString() override;
};
std::string Clock::ToString() { return "builtin function Clock"; }

const std::map<std::string, object::LoxObject>& BuiltinCallables() {
  static std::map<std::string, object::LoxObject> map{{"Clock", object::LoxObject(new Clock)}};
  return map;
};
}  // namespace lox
