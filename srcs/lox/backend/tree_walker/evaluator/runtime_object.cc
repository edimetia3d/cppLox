//
// LICENSE: MIT
//

#include "runtime_object.h"

#include "lox/backend/tree_walker/evaluator/evaluator.h"
#include "lox/common/finally.h"

namespace lox::twalker {

int Closure::Arity() { return data.function->attr->params.size(); }

ObjectPtr Closure::Call(Evaluator *evaluator, ObjectPtr this_in_sp, std::vector<ObjectPtr> arguments) {
  assert(this == this_in_sp->obj());
  // closure will always use the closed env as base, and create a dynamic new local env;
  auto local = Environment::Make(data.closed);
  auto bakup = evaluator->SwitchEnv(local);
  Finally finally([evaluator, bakup]() { evaluator->SwitchEnv(bakup); });
  int N = Arity();
  for (int i = 0; i < N; ++i) {
    evaluator->WorkEnv()->Define(data.function->attr->params[i]->lexeme, arguments[i]);
  }
  for (auto &stmt : data.function->body) {
    evaluator->Eval(stmt.get());
  }
  // if no return was thrown, we gen a default return value.
  // todo: use a pass to inject a return statement is better.
  auto default_ret = ObjectPtr();
  if (data.function->attr->name->lexeme == "init") {
    default_ret = evaluator->WorkEnv()->Get("this");
  }
  return default_ret;
}
std::string Closure::Str() const { return std::string("Function ") + data.function->attr->name->lexeme; }

int Klass::Arity() {
  auto init_fn = GetMethod(ObjectPtr(), "init");
  if (init_fn) {
    return init_fn->obj()->As<Closure>()->Arity();
  }
  return 0;
}
ObjectPtr Klass::Call(Evaluator *evaluator, ObjectPtr this_in_sp, std::vector<ObjectPtr> arguments) {
  assert(this_in_sp->obj() == this);
  InstanceData data{};
  data.klass = this_in_sp;
  auto ret = Object::MakeShared<Instance>(data);
  auto init_fn = GetMethod(ret, "init");
  if (init_fn) {
    init_fn->obj()->As<Closure>()->Call(evaluator, ret, arguments);
  }
  return ret;
}
std::string Klass::Str() const { return std::string("class ") + data.name; }

ObjectPtr Klass::GetMethod(ObjectPtr object_this, const std::string &name) {
  auto klass = this;
  if (!klass->data.methods.contains(name)) {
    klass = klass->data.superclass->obj()->As<Klass>();
  }
  if (klass) {
    auto closure = klass->data.methods[name]->obj()->As<Closure>();
    ClosureData data{.closed = Environment::Make(closure->data.closed), .function = closure->data.function};
    data.closed->Define("this", object_this);
    if (klass->data.superclass) {
      auto super_instance_data = object_this->obj()->DynAs<Instance>()->data;
      super_instance_data.klass = klass->data.superclass;
      data.closed->Define("super", Object::MakeShared<Instance>(super_instance_data));
    }
    return Object::MakeShared<Closure>(data);
  }

  return ObjectPtr();
}
void Klass::AddMethod(const std::string &name, ObjectPtr method) {
  assert(method->obj()->Is<Closure>());
  data.methods[name] = method;
}

std::string Instance::Str() const { return std::string("Instance of ") + data.klass->obj()->Str(); }

InstanceData::InstanceData() noexcept : dict_(std::make_shared<DictT>()) {}
}  // namespace lox::twalker