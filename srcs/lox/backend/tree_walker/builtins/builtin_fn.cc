//
// LICENSE: MIT
//

#include "builtin_fn.h"

namespace lox::twalker {

class Clock : public lox::Object, public ICallable {
 public:
  Clock() = default;
  int Arity() override { return 0; }
  ObjectPtr Call(Evaluator* evaluator, ObjectPtr this_in_sp, std::vector<ObjectPtr> arguments) override {
    double clk = clock();
    return Object::MakeShared<Number>(clk);
  }
  std::string Str() const override { return "<native fn>"; }
  std::vector<Object*> References() override { return {}; }
};

std::map<std::string, ObjectPtr> BuiltinCallables() {
  std::map<std::string, ObjectPtr> map{{"clock", Object::MakeShared<Clock>()}};
  return map;
}
}  // namespace lox::twalker
