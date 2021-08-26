//
// LICENSE: MIT
//

#include "lox/backend/tree_walker/visitors/evaluator/callable_object.h"

#include <time.h>

namespace lox {

class Clock : public LoxCallable {
 public:
  using ClockDummyData = int;
  using RawValueT = ClockDummyData;
  int Arity() override { return 0; }
  object::LoxObject Call(Evaluator* evaluator, std::vector<object::LoxObject> arguments) {
    double clk = clock();
    object::LoxObject();
    return object::MakeLoxObject(clk);
  }
  std::string ToString() const override;
  LOX_OBJECT_CTOR_SHARED_PTR_ONLY(Clock);
};
std::string Clock::ToString() const { return "builtin function Clock"; }

const std::map<std::string, object::LoxObject>& BuiltinCallables() {
  static std::map<std::string, object::LoxObject> map{{"Clock", object::MakeLoxObject<Clock>(0)}};
  return map;
};
}  // namespace lox
