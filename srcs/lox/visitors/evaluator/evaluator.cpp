//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/evaluator.h"

#include <iostream>

#include "lox/error.h"
#include "lox/visitors/evaluator/callable_object.h"
#include "lox/visitors/evaluator/class_instance.h"
#include "lox/visitors/evaluator/lox_class.h"
#include "lox/visitors/evaluator/lox_function.h"
namespace lox {

object::LoxObject Evaluator::Visit(LiteralExpr* state) {
  switch (state->value.type_) {
    case TokenType::NUMBER:
      return object::MakeLoxObject(std::stod(state->value.lexeme_));
    case TokenType::STRING:
      return object::MakeLoxObject(std::string(state->value.lexeme_.begin() + 1, state->value.lexeme_.end() - 1));
    case TokenType::TRUE:
      return object::MakeLoxObject(true);
    case TokenType::FALSE:
      return object::MakeLoxObject(false);
    case TokenType::NIL:
      return object::VoidObject();
    default:
      throw RuntimeError(Error(state->value, "Not a valid Literal."));
  }
}
object::LoxObject Evaluator::Visit(GroupingExpr* state) { return Eval(state->expression); }
object::LoxObject Evaluator::Visit(UnaryExpr* state) {
  auto right = Eval(state->right);

  switch (state->op.type_) {
    case TokenType::MINUS:
      return -right;
    case TokenType::BANG:
      try {
        return object::MakeLoxObject(IsValueTrue(right));
      } catch (const char* msg) {
        throw RuntimeError(Error(state->op, msg));
      }

    default:
      throw RuntimeError(Error(state->op, "Not a valid Unary Op."));
  }
}

object::LoxObject Evaluator::Visit(LogicalExpr* state) {
  auto left = Eval(state->left);
  if (state->op.type_ == TokenType::AND) {
    if (IsValueTrue(left)) {
      return Eval(state->right);
    }

  } else if (!IsValueTrue(left)) {
    return Eval(state->right);
  }
  return left;
}

object::LoxObject Evaluator::Visit(BinaryExpr* state) {
  auto left = Eval(state->left);
  auto right = Eval(state->right);
  try {
    switch (state->op.type_) {
      case TokenType::PLUS:
        return left + right;
      case TokenType::MINUS:
        return left - right;
      case TokenType::STAR:
        return left * right;
      case TokenType::SLASH:
        return left / right;
      case TokenType::EQUAL_EQUAL:
        return left == right;
      case TokenType::BANG_EQUAL:
        return left != right;
      case TokenType::LESS:
        return left < right;
      case TokenType::GREATER:
        return left > right;
      case TokenType::LESS_EQUAL:
        return left <= right;
      case TokenType::GREATER_EQUAL:
        return left >= right;
      default:
        throw RuntimeError(Error(state->op, "Not a valid Binary Op."));
    }
  } catch (const char* msg) {
    throw RuntimeError(Error(state->op, msg));
  }
}
object::LoxObject Evaluator::Visit(VariableExpr* state) {
  auto ret = WorkEnv()->GetByDistance(active_map_->Get(state))->Get(state->name.lexeme_);
  if (!IsValid(ret)) {
    throw RuntimeError(Error(state->name, "Doesnt reference to a valid value."));
  }
  return ret;
}
object::LoxObject Evaluator::Visit(AssignExpr* state) {
  auto value = Eval(state->value);
  try {
    WorkEnv()->GetByDistance(active_map_->Get(state))->Set(state->name.lexeme_, value);
  } catch (const char* msg) {
    throw RuntimeError(Error(state->name, msg));
  }
  return value;
}
object::LoxObject Evaluator::Visit(CallExpr* state) {
  auto callee = Eval(state->callee);

  std::vector<object::LoxObject> arguments;
  for (Expr argument : state->arguments) {
    arguments.push_back(Eval(argument));
  }

  auto callable = callee->DownCast<LoxCallable>();
  if (!callable) {
    throw RuntimeError(Error(state->paren, "Not a callable object"));
  }
  if (arguments.size() != callable->Arity()) {
    throw RuntimeError(Error(state->paren, "Wrong arg number"));
  }
  try {
    return callable->Call(this, arguments);
  } catch (const char* msg) {
    throw RuntimeError(Error(state->paren, msg));
  }
}

object::LoxObject Evaluator::Visit(PrintStmt* state) {
  auto ret_v = Eval(state->expression);
  std::cout << ret_v->ToString() << std::endl;
  return object::VoidObject();
}
object::LoxObject Evaluator::Visit(ExprStmt* state) {
  Eval(state->expression);
  return object::VoidObject();
}
object::LoxObject Evaluator::Visit(VarDeclStmt* state) {
  auto value = object::VoidObject();
  if (IsValid(state->initializer)) {
    value = Eval(state->initializer);
  }
  WorkEnv()->Define(state->name.lexeme_, value);
  return object::VoidObject();
}
object::LoxObject Evaluator::Visit(BlockStmt* state) {
  EnterNewScopeGuard guard(this);
  for (auto& stmt : state->statements) {
    Eval(stmt);
  }
  return object::VoidObject();
}
object::LoxObject Evaluator::Visit(IfStmt* state) {
  if (IsValueTrue((Eval(state->condition)))) {
    Eval(state->thenBranch);
  } else if (IsValid(state->elseBranch)) {
    Eval(state->elseBranch);
  }
  return object::VoidObject();
}
object::LoxObject Evaluator::Visit(WhileStmt* state) {
  while (IsValueTrue((Eval(state->condition)))) {
    try {
      Eval(state->body);
    } catch (RuntimeError& err) {
      if (err.err.SourceToken().type_ == TokenType::BREAK) {
        break;
      }
      throw;
    }
  }
  return object::VoidObject();
}
object::LoxObject Evaluator::Visit(BreakStmt* state) { throw RuntimeError(Error(state->src_token, "Hit break")); }
object::LoxObject Evaluator::Visit(FunctionStmt* state) {
  LoxFunctionData fn_data{
      .is_init_method = false,
      .closure = WorkEnv(),
      .function = std::static_pointer_cast<FunctionStmt>(state->shared_from_this()),

  };
  auto fn = object::MakeLoxObject<LoxFunction>(fn_data);
  WorkEnv()->Define(state->name.lexeme_, fn);
  // FreezeEnv(); use freeze env to "copy" capture
  return fn;
}
object::LoxObject Evaluator::Visit(ReturnStmt* state) {
  auto ret = object::VoidObject();
  if (IsValid(state->value)) {
    ret = Eval(state->value);
  }
  throw ReturnValue(ret);
}
object::LoxObject Evaluator::Visit(ClassStmt* state) {
  WorkEnv()->Define(state->name.lexeme_, object::VoidObject());
  auto klass = object::VoidObject();
  {
    EnterNewScopeGuard(this, WorkEnv());
    auto superclass = object::VoidObject();
    if (IsValid(state->superclass)) {
      superclass = Eval(state->superclass);
      if (!superclass->DownCast<LoxClass>()) {
        throw RuntimeError(Error(state->name, "Base must be a class"));
      }
    }
    LoxClassData class_data{.name = state->name.lexeme_, .superclass = std::static_pointer_cast<LoxClass>(superclass)};
    klass = object::MakeLoxObject<LoxClass>(class_data);
    for (auto& method_stmt : state->methods) {
      auto method_state = method_stmt->DownCast<FunctionStmt>();
      LoxFunctionData fn_data{
          .is_init_method = method_state->name.lexeme_ == "init",
          .closure = WorkEnv(),
          .function = std::static_pointer_cast<FunctionStmt>(method_state->shared_from_this()),
      };
      auto method = object::MakeLoxObject<LoxFunction>(fn_data);
      klass->SetAttr(method_state->name.lexeme_, method);
    }
  }
  WorkEnv()->Set(state->name.lexeme_, klass);
  return object::VoidObject();
}
object::LoxObject Evaluator::Visit(GetAttrExpr* state) {
  auto object = Eval(state->src_object);
  const auto& attr_name = state->attr_name.lexeme_;
  auto ret = object::VoidObject();
  // Let's handle binding of this
  if (auto instance = object->DownCast<LoxClassInstance>()) {
    if (auto attr = instance->GetAttr(attr_name)) {
      return attr;
    }
    ret = instance->RawValue<LoxClassInstanceData>().klass->GetAttr(attr_name);
    if (auto function = ret->DownCast<LoxFunction>()) {
      ret = function->BindThis(instance->shared_from_this());
    }
  } else if (auto klass = object->DownCast<LoxClass>()) {
    ret = klass->GetAttr(attr_name);
    if (auto fn = ret->DownCast<LoxFunction>()) {
      if (auto objcet_this = WorkEnv()->Get("this")) {
        if (!(objcet_this->DownCast<LoxClassInstance>()->IsInstanceOf(klass))) {
          throw RuntimeError(Error(state->attr_name, klass->ToString() + " is not superclass"));
        }
        ret = fn->BindThis(objcet_this);
      } else {
        throw RuntimeError(Error(state->attr_name, "Cannot use method out of method"));
      }
    }
  } else {
    ret = object->GetAttr(attr_name);
  }

  if (!IsValid(ret)) {
    throw RuntimeError(Error(state->attr_name, "No attr found"));
  }
  return ret;
}
object::LoxObject Evaluator::Visit(SetAttrExpr* state) {
  auto object = Eval(state->src_object);
  auto ret = Eval(state->value);
  object->SetAttr(state->attr_name.lexeme_, Eval(state->value));
  return ret;
}
}  // namespace lox
