//
// LICENSE: MIT
//

#include "lox/visitors/evaluator/evaluator.h"

#include <iostream>

#include "lox/error.h"
#include "lox/visitors/evaluator/callable_object.h"
#include "lox/visitors/evaluator/lox_function.h"

namespace lox {

struct ReturnValue : public std::exception {
  explicit ReturnValue(object::LoxObject obj) : ret(std::move(obj)) {}

  object::LoxObject ret;
};

object::LoxObject ExprEvaluator::Visit(LiteralState* state) {
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
object::LoxObject ExprEvaluator::Visit(GroupingState* state) { return Eval(state->expression); }
object::LoxObject ExprEvaluator::Visit(UnaryState* state) {
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

object::LoxObject ExprEvaluator::Visit(LogicalState* state) {
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

object::LoxObject ExprEvaluator::Visit(BinaryState* state) {
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
object::LoxObject ExprEvaluator::Visit(VariableState* state) {
  auto ret = object::LoxObject::VoidObject();
  try {
    ret = WorkEnv()->Get(state->name.lexeme_);
  } catch (const char* msg) {
    throw RuntimeError(Error(state->name, msg));
  }
  if (!ret.IsValid()) {
    throw RuntimeError(Error(state->name, "Doesnt reference to a valid value."));
  }
  return ret;
}
object::LoxObject ExprEvaluator::Visit(AssignState* state) {
  auto value = Eval(state->value);
  try {
    WorkEnv()->Set(state->name.lexeme_, value);
  } catch (const char* msg) {
    throw RuntimeError(Error(state->name, msg));
  }
  return value;
}
object::LoxObject ExprEvaluator::Visit(CallState* state) {
  auto callee = Eval(state->callee);

  std::vector<object::LoxObject> arguments;
  for (Expr argument : state->arguments) {
    arguments.push_back(Eval(argument));
  }

  LoxCallable function = static_cast<LoxCallable>(callee);
  if (!function.IsValid()) {
    throw RuntimeError(Error(state->paren, "Not a callable object"));
  }
  if (arguments.size() != function.Arity()) {
    throw RuntimeError(Error(state->paren, "Wrong arg number"));
  }
  try {
    return function.Call(parent_, arguments);
  } catch (ReturnValue& ret) {
    return ret.ret;
  } catch (const char* msg) {
    throw RuntimeError(Error(state->paren, msg));
  }
}

object::LoxObject StmtEvaluator::Visit(PrintStmtState* state) {
  auto ret_v = expr_evaluator_.Eval(state->expression);
  std::cout << ret_v.ToString() << std::endl;
  return object::LoxObject::VoidObject();
}
object::LoxObject StmtEvaluator::Visit(ExprStmtState* state) {
  expr_evaluator_.Eval(state->expression);
  return object::LoxObject::VoidObject();
}
object::LoxObject StmtEvaluator::Visit(VarDeclStmtState* state) {
  auto value = object::LoxObject::VoidObject();
  if (state->initializer.IsValid()) {
    value = expr_evaluator_.Eval(state->initializer);
  }
  WorkEnv()->Define(state->name.lexeme_, value);
  return object::LoxObject::VoidObject();
}
object::LoxObject StmtEvaluator::Visit(BlockStmtState* state) {
  EnterNewScopeGuard guard(this);
  for (auto& stmt : state->statements) {
    Eval(stmt);
  }
  return object::LoxObject::VoidObject();
}
object::LoxObject StmtEvaluator::Visit(IfStmtState* state) {
  if ((expr_evaluator_.Eval(state->condition)).IsValueTrue()) {
    Eval(state->thenBranch);
  } else if (state->elseBranch.IsValid()) {
    Eval(state->elseBranch);
  }
  return object::LoxObject::VoidObject();
}
object::LoxObject StmtEvaluator::Visit(WhileStmtState* state) {
  while ((expr_evaluator_.Eval(state->condition)).IsValueTrue()) {
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
object::LoxObject StmtEvaluator::Visit(BreakStmtState* state) {
  throw RuntimeError(Error(state->src_token, "Hit break"));
}
object::LoxObject StmtEvaluator::Visit(FunctionStmtState* state) {
  auto fn = object::LoxObject(new LoxFunctionState(state, WorkEnv()));
  WorkEnv()->Define(state->name.lexeme_, fn);
  return fn;
}
object::LoxObject StmtEvaluator::Visit(ReturnStmtState* state) {
  auto ret = object::LoxObject::VoidObject();
  if (state->value.IsValid()) {
    ret = expr_evaluator_.Eval(state->value);
  }
  throw ReturnValue(ret);
}
}  // namespace lox
