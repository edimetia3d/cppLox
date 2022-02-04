//
// LICENSE: MIT
//

#include "runtime_object.h"

#include "lox/backend/tree_walker/evaluator/evaluator.h"
#include "lox/common/finally.h"

namespace lox::twalker {

int Closure::Arity() {
  if (auto p = data.function->comma_expr_params.get()) {
    return p->DynAs<CommaExpr>()->elements.size();
  } else {
    return 0;
  }
}

ObjectPtr Closure::Call(Evaluator *evaluator, ObjectPtr this_in_sp, std::vector<ObjectPtr> arguments) {
  assert(this == this_in_sp->obj());
  // closure will always use the closed env as base, and create a dynamic new local env;
  auto local = Environment::Make(data.closed);
  auto bakup = evaluator->SwitchEnv(local);
  Finally finally([evaluator, bakup]() { evaluator->SwitchEnv(bakup); });
  int N = Arity();
  for (int i = 0; i < N; ++i) {
    auto argi = data.function->comma_expr_params->DynAs<CommaExpr>()->elements[i]->DynAs<VariableExpr>();
    evaluator->WorkEnv()->Define(argi->attr->name->lexeme, arguments[i]);
  }
  auto ret = NullObject();
  try {
    for (auto &stmt : data.function->body) {
      evaluator->Eval(stmt.get());
    }
  } catch (const ReturnValueException &e) {
    ret = e.ret;
  }
  if (!ret) {
    if (data.is_initializer) {
      ret = data.closed->Get("this");
    } else {
      ret = Object::MakeShared<Nil>();
    }
  }
  return ret;
}
std::string Closure::Str() const { return std::string("<fn ") + data.function->attr->name->lexeme + ">"; }

int Klass::Arity() {
  auto init_fn = GetMethod(NullObject(), "init");
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
    auto init_ret = init_fn->obj()->As<Closure>()->Call(evaluator, init_fn, arguments);
    assert(init_ret->obj() == ret->obj());
  }
  return ret;
}
std::string Klass::Str() const { return data.name; }

ObjectPtr Klass::GetMethod(ObjectPtr object_this, const std::string &name) {
  auto klass = this;
  while (klass && !klass->data.methods.contains(name)) {
    if (klass->data.superclass) {
      klass = klass->data.superclass->obj()->As<Klass>();
    } else {
      klass = nullptr;
    }
  }
  if (klass) {
    auto closure = klass->data.methods[name]->obj()->As<Closure>();
    ClosureData data{.is_initializer = closure->data.function->DynAs<FunctionStmt>()->attr->name->lexeme == "init",
                     .closed = Environment::Make(closure->data.closed),
                     .function = closure->data.function};
    if (object_this) {
      data.closed->Define("this", object_this);
      if (klass->data.superclass) {
        auto super_instance_data = object_this->obj()->DynAs<Instance>()->data;
        super_instance_data.klass = klass->data.superclass;
        data.closed->Define("super", Object::MakeShared<Instance>(super_instance_data));
      }
    }
    return Object::MakeShared<Closure>(data);
  }

  return NullObject();
}
void Klass::AddMethod(const std::string &name, ObjectPtr method) {
  assert(method->obj()->Is<Closure>());
  data.methods[name] = method;
}

std::string Instance::Str() const { return data.klass->obj()->Str() + " instance"; }

InstanceData::InstanceData() noexcept : dict_(std::make_shared<DictT>()) {}
std::string List::Str() const {
  std::string ret = "[";
  int count = 0;
  for (auto &item : data) {
    if (count > 0) {
      ret += ", ";
    }
    ret += item->obj()->Str();
    ++count;
  }
  ret += "]";
  return ret;
}

bool List::Equal(const Object *rhs) const {
  if (!rhs->DynAs<List>()) {
    return false;
  }
  if (rhs->As<List>()->data.size() != data.size()) {
    return false;
  }
  bool equal = true;

  for (int i = 0; i < data.size(); i++) {
    equal = equal && data[i]->obj()->Equal(rhs->As<List>()->data[i]->obj());
  }
  return equal;
}
}  // namespace lox::twalker