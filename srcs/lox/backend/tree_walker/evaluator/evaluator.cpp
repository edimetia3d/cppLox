//
// LICENSE: MIT
//

#include "evaluator.h"

#include <iostream>

#include "lox/ast/ast.h"
#include "lox/backend/tree_walker/bultins/builtin_fn.h"
#include "lox/backend/tree_walker/evaluator/runtime_object.h"

namespace lox::twalker {

class BreakException : public std::exception {};
class ContinueException : public std::exception {};

struct NewEnvGuard {
  NewEnvGuard(Evaluator* evaluator) : evaluator(evaluator) {
    auto new_env = Environment::Make(evaluator->WorkEnv());
    backup = evaluator->SwitchEnv(new_env);
  }
  ~NewEnvGuard() { evaluator->SwitchEnv(backup); }
  EnvPtr backup;
  Evaluator* evaluator;
};

void Evaluator::Visit(AssignExpr* node) {
  auto value = Eval(node->value.get());
  if (!WorkEnv()->Set(node->attr->name->lexeme, value)) {
    Error("Doesnt reference to a valid value.");
  }
  VisitorReturn(value);
}

void Evaluator::Visit(LiteralExpr* node) {
  // todo: add a pass to expand/fold constants
  switch (node->attr->value->type) {
    case TokenType::NUMBER:
      VisitorReturn(Object::MakeShared<Number>(std::stod(node->attr->value->lexeme)));
    case TokenType::STRING:
      VisitorReturn(Object::MakeShared<String>(
          std::string(node->attr->value->lexeme.begin() + 1, node->attr->value->lexeme.end() - 1)));
    case TokenType::TRUE_TOKEN:
      VisitorReturn(Object::MakeShared<Bool>(true));
    case TokenType::FALSE_TOKEN:
      VisitorReturn(Object::MakeShared<Bool>(false));
    case TokenType::NIL:
      VisitorReturn(Object::MakeShared<Nil>());
    default:
      Error("Not a valid Literal.");
  }
}
void Evaluator::Visit(GroupingExpr* node) { VisitorReturn(Eval(node->expression.get())); }

void Evaluator::Visit(UnaryExpr* node) {
  auto right = Eval(node->right.get());

  switch (node->attr->op->type) {
    case TokenType::MINUS:
      if (!right->obj()->DynAs<Number>()) {
        Error("Only number support minus");
      }
      VisitorReturn(Object::MakeShared<Number>(-1 * right->obj()->As<Number>()->data));
    case TokenType::BANG:
      VisitorReturn(Object::MakeShared<Bool>(!right->obj()->IsTrue()));
    default:
      Error("Not a valid Unary Op.");
  }
}

void Evaluator::Visit(LogicalExpr* node) {
  auto left_obj = Eval(node->left.get());
  auto left = left_obj->obj()->IsTrue();
  if (node->attr->op->type == TokenType::AND && left) {
    VisitorReturn(Eval(node->right.get()));
  }
  if (node->attr->op->type == TokenType::OR && !left) {
    VisitorReturn(Eval(node->right.get()));
  }
  VisitorReturn(left_obj);
}

void Evaluator::Visit(BinaryExpr* node) {
  auto left = Eval(node->left.get());
  auto right = Eval(node->right.get());
  if (!left->obj()->DynAs<Number>() || !right->obj()->DynAs<Number>()) {
    Error("Only Number support binary op");
  }
  auto left_num = left->obj()->As<Number>()->data;
  auto right_num = right->obj()->As<Number>()->data;
  switch (node->attr->op->type) {
    case TokenType::PLUS:
      VisitorReturn(Object::MakeShared<Number>(left_num + right_num));
    case TokenType::MINUS:
      VisitorReturn(Object::MakeShared<Number>(left_num - right_num));
    case TokenType::STAR:
      VisitorReturn(Object::MakeShared<Number>(left_num * right_num));
    case TokenType::SLASH:
      VisitorReturn(Object::MakeShared<Number>(left_num / right_num));
    case TokenType::EQUAL_EQUAL:
      VisitorReturn(Object::MakeShared<Number>(left_num == right_num));
    case TokenType::BANG_EQUAL:
      VisitorReturn(Object::MakeShared<Number>(left_num != right_num));
    case TokenType::LESS:
      VisitorReturn(Object::MakeShared<Number>(left_num < right_num));
    case TokenType::GREATER:
      VisitorReturn(Object::MakeShared<Number>(left_num > right_num));
    case TokenType::LESS_EQUAL:
      VisitorReturn(Object::MakeShared<Number>(left_num <= right_num));
    case TokenType::GREATER_EQUAL:
      VisitorReturn(Object::MakeShared<Number>(left_num >= right_num));
    default:
      Error("Not a valid Binary Op.");
  }
}
void Evaluator::Visit(VariableExpr* node) {
  auto ret = WorkEnv()->Get(node->attr->name->lexeme);
  if (!ret) {
    Error("Doesnt reference to a valid value.");
  }
  VisitorReturn(ret);
}

void Evaluator::Visit(GetAttrExpr* node) {
  // get attr may return normal object or bounded method
  auto object = Eval(node->src_object.get());
  const auto& attr_name = node->attr->attr_name->lexeme;

  Klass* klass = nullptr;
  ObjectPtr objcet_this;
  ObjectPtr ret;

  // try instance first
  if (auto instance = object->obj()->DynAs<Instance>()) {
    if (instance->data.dict().contains(attr_name)) {
      VisitorReturn(instance->data.dict()[attr_name]);
    }
    // ok, we need get method
    klass = instance->data.klass->obj()->As<Klass>();
    objcet_this = object;
  }
  // if klass not set, the object should be a klass
  if (!klass) {
    klass = object->obj()->DynAs<Klass>();
    objcet_this = WorkEnv()->Get("this");
    if (!objcet_this) {
      Error("Cannot use method out of method");  // todo: this should be a semantic error
    }
  }
  auto fn = klass->GetMethod(objcet_this, attr_name)->obj()->DynAs<Closure>();
  if (!ret) {
    Error("No attr found");
  }
  VisitorReturn(ret);
}

void Evaluator::Visit(SetAttrExpr* node) {
  auto object = Eval(node->src_object.get());
  auto ret = Eval(node->value.get());
  object->obj()->DynAs<Instance>()->data.dict()[node->attr->attr_name->lexeme] = ret;
  VisitorReturn(ret);
}

void Evaluator::Visit(CallExpr* node) {
  auto callee = Eval(node->callee.get());
  auto callable = dynamic_cast<ICallable*>(callee->obj());
  if (!callable) {
    Error("Not a callable object");
  }

  std::vector<ObjectPtr> arguments;
  for (auto& arg_expr : node->arguments) {
    arguments.push_back(Eval(arg_expr.get()));
  }

  if (arguments.size() != callable->Arity()) {
    Error("Wrong arg number");
  }

  NewEnvGuard guard(this);  // used as temporary local
  try {
    VisitorReturn(callable->Call(this, callee, arguments));
  } catch (const ReturnValueException& e) {
    VisitorReturn(e.ret);
  }
}

void Evaluator::Visit(FunctionStmt* node) {
  auto closure = CreateClosure(node);
  WorkEnv()->Define(node->attr->name->lexeme, closure);
  VisitorReturn(ObjectPtr());
}
void Evaluator::Visit(ClassStmt* node) {
  auto superclass = ObjectPtr();
  if (node->superclass) {
    superclass = Eval(node->superclass.get());
    if (!superclass->obj()->DynAs<Klass>()) {
      Error("Base must be a class");  // todo: should be a semantic error
    }
  }
  KlassData class_data{.name = node->attr->name->lexeme, .superclass = superclass};
  auto klass = Object::MakeShared<Klass>(class_data);
  WorkEnv()->Define(node->attr->name->lexeme, klass);
  // klass is defined in outer scope, while methods generation are execute in inner scope
  {
    NewEnvGuard(this);
    for (auto& method_stmt : node->methods) {
      auto method = method_stmt->As<FunctionStmt>();  // todo : semantic check, only function stmt is allowed
      klass->obj()->As<Klass>()->AddMethod(method->attr->name->lexeme, CreateClosure(method));
    }
  }
  VisitorReturn(ObjectPtr());
}

void Evaluator::Visit(PrintStmt* node) {
  auto ret_v = Eval(node->expression.get());
  std::cout << ret_v->obj()->Str() << std::endl;
  VisitorReturn(ObjectPtr());
}
void Evaluator::Visit(ExprStmt* state) {
  Eval(state->expression.get());
  VisitorReturn(ObjectPtr());
}

void Evaluator::Visit(VarDeclStmt* state) {
  auto value = ObjectPtr();
  if (state->initializer) {
    value = Eval(state->initializer.get());
  }
  WorkEnv()->Define(state->attr->name->lexeme, value);
  VisitorReturn(ObjectPtr());
}

void Evaluator::Visit(BlockStmt* state) {
  NewEnvGuard guard(this);
  for (auto& stmt : state->statements) {
    Eval(stmt.get());
  }
  VisitorReturn(ObjectPtr());
}

void Evaluator::Visit(IfStmt* state) {
  auto cond_obj = Eval(state->condition.get());
  if (cond_obj && cond_obj->obj()->IsTrue()) {
    Eval(state->thenBranch.get());
  } else if (state->elseBranch) {
    Eval(state->elseBranch.get());
  }
  VisitorReturn(ObjectPtr());
}

void Evaluator::Visit(WhileStmt* state) {
  while (Eval(state->condition.get())->obj()->IsTrue()) {
    try {
      Eval(state->body.get());
    } catch (BreakException& err) {
      break;
    } catch (ContinueException& err) {
      continue;
    }
  }
  VisitorReturn(ObjectPtr());
}

void Evaluator::Visit(ForStmt* node) {
  NewEnvGuard(this);  // for loop may create a new local variable
  if (node->initializer) {
    Eval(node->initializer.get());
  }
  for (; Eval(node->condition.get())->obj()->IsTrue(); node->increment ? Eval(node->increment.get()) : ObjectPtr()) {
    try {
      Eval(node->body.get());
    } catch (BreakException& err) {
      break;
    } catch (ContinueException& err) {
      continue;
    }
  }
  VisitorReturn(ObjectPtr());
}

void Evaluator::Visit(BreakStmt* state) {
  if (state->attr->src_token->type == TokenType::BREAK) {
    throw BreakException();
  }
  if (state->attr->src_token->type == TokenType::CONTINUE) {
    throw ContinueException();
  }
  Error("Not a valid break statement");  // todo: should be a semantic error
}

void Evaluator::Visit(ReturnStmt* state) {
  auto ret = ObjectPtr();
  if (state->value) {
    ret = Eval(state->value.get());  // todo: check only `this` is allowed to return when in `init`
  }
  throw ReturnValueException(ret);
}

ObjectPtr Evaluator::CreateClosure(FunctionStmt* function) {
  auto closed = WorkEnv();
  if (!closed->Parent()) {
    closed = EnvPtr();  // we do not close global to avoid object defined in global create cycle with env
  }
  ClosureData fn_data{
      .closed = closed,
      .function = function,
  };
  return Object::MakeShared<Closure>(fn_data);
}

Evaluator::Evaluator() : work_env_(Environment::Make()) {
  for (auto it : BuiltinCallables()) {
    work_env_->Define(it.first, it.second);
  }
}

ObjectPtr Evaluator::Eval(ASTNode* node) {
  assert(node);
  node->Accept(this);
  return PopRet();
}

void Evaluator::Error(const std::string& msg) { throw RuntimeError(msg); }

EnvPtr Evaluator::SwitchEnv(EnvPtr new_env) {
  auto old_env = work_env_;
  work_env_ = new_env;
  return old_env;
}
void Evaluator::LaunchScript(FunctionStmt* script) {
  Eval(script);  // define the <script>
  auto callee = WorkEnv()->Get("<script>");
  auto callable = dynamic_cast<ICallable*>(callee->obj());
  assert(callable);
  callable->Call(this, callee, {});
}
}  // namespace lox::twalker
