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

object::LoxObject Evaluator::Visit(LiteralState* state) {
  switch (state->value.type_) {
    case TokenType::NUMBER:
      return object::LoxObject(std::stod(state->value.lexeme_));
    case TokenType::STRING:
      return object::LoxObject(std::string(state->value.lexeme_.begin() + 1, state->value.lexeme_.end() - 1));
    case TokenType::TRUE:
      return object::LoxObject(true);
    case TokenType::FALSE:
      return object::LoxObject(false);
    default:
      throw RuntimeError(Error(state->value, "Not a valid Literal."));
  }
}
object::LoxObject Evaluator::Visit(GroupingState* state) { return Eval(state->expression); }
object::LoxObject Evaluator::Visit(UnaryState* state) {
  auto right = Eval(state->right);

  switch (state->op.type_) {
    case TokenType::MINUS:
      return -right;
    case TokenType::BANG:
      try {
        return object::LoxObject(right.IsValueTrue());
      } catch (const char* msg) {
        throw RuntimeError(Error(state->op, msg));
      }

    default:
      throw RuntimeError(Error(state->op, "Not a valid Unary Op."));
  }
}

object::LoxObject Evaluator::Visit(LogicalState* state) {
  auto left = Eval(state->left);
  if (state->op.type_ == TokenType::AND) {
    if (left.IsValueTrue()) {
      return Eval(state->right);
    }

  } else if (!left.IsValueTrue()) {
    return Eval(state->right);
  }
  return left;
}

object::LoxObject Evaluator::Visit(BinaryState* state) {
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
object::LoxObject Evaluator::Visit(VariableState* state) {
  auto ret = object::LoxObject::VoidObject();
  try {
    ret = WorkEnv()->GetByDistance(active_map_->Get(state))->Get(state->name.lexeme_);
  } catch (const char* msg) {
    throw RuntimeError(Error(state->name, msg));
  }
  if (!ret.IsValid()) {
    throw RuntimeError(Error(state->name, "Doesnt reference to a valid value."));
  }
  return ret;
}
object::LoxObject Evaluator::Visit(AssignState* state) {
  auto value = Eval(state->value);
  try {
    WorkEnv()->GetByDistance(active_map_->Get(state))->Set(state->name.lexeme_, value);
  } catch (const char* msg) {
    throw RuntimeError(Error(state->name, msg));
  }
  return value;
}
object::LoxObject Evaluator::Visit(CallState* state) {
  auto callee = Eval(state->callee);

  std::vector<object::LoxObject> arguments;
  for (Expr argument : state->arguments) {
    arguments.push_back(Eval(argument));
  }

  auto function = callee.DownCastState<LoxCallableState>();
  if (!function) {
    throw RuntimeError(Error(state->paren, "Not a callable object"));
  }
  if (arguments.size() != function->Arity()) {
    throw RuntimeError(Error(state->paren, "Wrong arg number"));
  }
  try {
    return function->Call(this, arguments);
  } catch (const char* msg) {
    throw RuntimeError(Error(state->paren, msg));
  }
}

object::LoxObject Evaluator::Visit(PrintStmtState* state) {
  auto ret_v = Eval(state->expression);
  std::cout << ret_v.ToString() << std::endl;
  return object::LoxObject::VoidObject();
}
object::LoxObject Evaluator::Visit(ExprStmtState* state) {
  Eval(state->expression);
  return object::LoxObject::VoidObject();
}
object::LoxObject Evaluator::Visit(VarDeclStmtState* state) {
  auto value = object::LoxObject::VoidObject();
  if (state->initializer.IsValid()) {
    value = Eval(state->initializer);
  }
  WorkEnv()->Define(state->name.lexeme_, value);
  return object::LoxObject::VoidObject();
}
object::LoxObject Evaluator::Visit(BlockStmtState* state) {
  EnterNewScopeGuard guard(this);
  for (auto& stmt : state->statements) {
    Eval(stmt);
  }
  return object::LoxObject::VoidObject();
}
object::LoxObject Evaluator::Visit(IfStmtState* state) {
  if ((Eval(state->condition)).IsValueTrue()) {
    Eval(state->thenBranch);
  } else if (state->elseBranch.IsValid()) {
    Eval(state->elseBranch);
  }
  return object::LoxObject::VoidObject();
}
object::LoxObject Evaluator::Visit(WhileStmtState* state) {
  while ((Eval(state->condition)).IsValueTrue()) {
    try {
      Eval(state->body);
    } catch (RuntimeError& err) {
      if (err.err.SourceToken().type_ == TokenType::BREAK) {
        break;
      }
      throw;
    }
  }
  return object::LoxObject::VoidObject();
}
object::LoxObject Evaluator::Visit(BreakStmtState* state) { throw RuntimeError(Error(state->src_token, "Hit break")); }
object::LoxObject Evaluator::Visit(FunctionStmtState* state) {
  auto fn = object::LoxObject(new LoxFunctionState(state, WorkEnv()));
  WorkEnv()->Define(state->name.lexeme_, fn);
  // FreezeEnv(); use freeze env to "copy" capture
  return fn;
}
object::LoxObject Evaluator::Visit(ReturnStmtState* state) {
  auto ret = object::LoxObject::VoidObject();
  if (state->value.IsValid()) {
    ret = Eval(state->value);
  }
  throw ReturnValue(ret);
}
object::LoxObject Evaluator::Visit(ClassStmtState* state) {
  WorkEnv()->Define(state->name.lexeme_, object::LoxObject::VoidObject());

  std::map<std::string, object::LoxObject> methods;
  for (auto& method_stmt : state->methods) {
    auto method_state = method_stmt.DownCastState<FunctionStmtState>();
    auto method =
        object::LoxObject(new LoxFunctionState(method_state, WorkEnv(), method_state->name.lexeme_ == "init"));
    methods[method_state->name.lexeme_] = method;
  }

  auto klass = object::LoxObject(new LoxClassState(state->name.lexeme_, methods));
  WorkEnv()->Set(state->name.lexeme_, klass);
  return object::LoxObject::VoidObject();
}
object::LoxObject Evaluator::Visit(GetAttrState* state) {
  auto object = Eval(state->src_object);
  if (auto p = object.DownCastState<LoxClassInstanceState>()) {
    auto ret = p->GetAttr(state->attr_name.lexeme_);
    if (!ret.IsValid()) {
      throw RuntimeError(Error(state->attr_name, "No attr found"));
    }
    return ret;
  }

  throw RuntimeError(Error(state->attr_name, "Only class instances have properties."));
}
object::LoxObject Evaluator::Visit(SetAttrState* state) {
  auto object = Eval(state->src_object);
  if (auto p = object.DownCastState<LoxClassInstanceState>()) {
    auto ret = Eval(state->value);
    p->SetAttr(state->attr_name.lexeme_, Eval(state->value));
    return ret;
  }

  throw RuntimeError(Error(state->attr_name, "Only class instances have properties."));
}
}  // namespace lox
