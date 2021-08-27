//
// LICENSE: MIT
//

#include "evaluator.h"

#include <iostream>

#include "lox/backend/tree_walker/error.h"
#include "lox/backend/tree_walker/visitors/evaluator/callable_object.h"
#include "lox/backend/tree_walker/visitors/evaluator/class_instance.h"
#include "lox/backend/tree_walker/visitors/evaluator/lox_class.h"
#include "lox/backend/tree_walker/visitors/evaluator/lox_function.h"
namespace lox {

void Evaluator::Visit(LiteralExpr* state) {
  switch (state->value()->type) {
    case TokenType::NUMBER:
      VisitorReturn(object::MakeLoxObject(std::stod(state->value()->lexeme)));
    case TokenType::STRING:
      VisitorReturn(
          object::MakeLoxObject(std::string(state->value()->lexeme.begin() + 1, state->value()->lexeme.end() - 1)));
    case TokenType::TRUE:
      VisitorReturn(object::MakeLoxObject(true));
    case TokenType::FALSE:
      VisitorReturn(object::MakeLoxObject(false));
    case TokenType::NIL:
      VisitorReturn(object::VoidObject());
    default:
      throw RuntimeError(TreeWalkerError(state->value(), "Not a valid Literal."));
  }
}
void Evaluator::Visit(GroupingExpr* state) { VisitorReturn(Eval(state->expression())); }
void Evaluator::Visit(UnaryExpr* state) {
  auto right = Eval(state->right());

  switch (state->op()->type) {
    case TokenType::MINUS:
      VisitorReturn(-right);
    case TokenType::BANG:
      try {
        VisitorReturn(object::MakeLoxObject(IsValueTrue(right)));
      } catch (const char* msg) {
        throw RuntimeError(TreeWalkerError(state->op(), msg));
      }

    default:
      throw RuntimeError(TreeWalkerError(state->op(), "Not a valid Unary Op."));
  }
}

void Evaluator::Visit(LogicalExpr* state) {
  auto left = Eval(state->left());
  if (state->op()->type == TokenType::AND) {
    if (IsValueTrue(left)) {
      VisitorReturn(Eval(state->right()));
    }

  } else if (!IsValueTrue(left)) {
    VisitorReturn(Eval(state->right()));
  }
  VisitorReturn(left);
}

void Evaluator::Visit(BinaryExpr* state) {
  auto left = Eval(state->left());
  auto right = Eval(state->right());
  try {
    switch (state->op()->type) {
      case TokenType::PLUS:
        VisitorReturn(left + right);
      case TokenType::MINUS:
        VisitorReturn(left - right);
      case TokenType::STAR:
        VisitorReturn(left * right);
      case TokenType::SLASH:
        VisitorReturn(left / right);
      case TokenType::EQUAL_EQUAL:
        VisitorReturn(left == right);
      case TokenType::BANG_EQUAL:
        VisitorReturn(left != right);
      case TokenType::LESS:
        VisitorReturn(left < right);
      case TokenType::GREATER:
        VisitorReturn(left > right);
      case TokenType::LESS_EQUAL:
        VisitorReturn(left <= right);
      case TokenType::GREATER_EQUAL:
        VisitorReturn(left >= right);
      default:
        throw RuntimeError(TreeWalkerError(state->op(), "Not a valid Binary Op."));
    }
  } catch (const char* msg) {
    throw RuntimeError(TreeWalkerError(state->op(), msg));
  }
}
void Evaluator::Visit(VariableExpr* state) {
  auto ret = WorkEnv()->GetByDistance(active_map_->Get(state))->Get(state->name()->lexeme);
  if (!IsValid(ret)) {
    throw RuntimeError(TreeWalkerError(state->name(), "Doesnt reference to a valid value."));
  }
  VisitorReturn(ret);
}
void Evaluator::Visit(AssignExpr* state) {
  auto value = Eval(state->value());
  try {
    WorkEnv()->GetByDistance(active_map_->Get(state))->Set(state->name()->lexeme, value);
  } catch (const char* msg) {
    throw RuntimeError(TreeWalkerError(state->name(), msg));
  }
  VisitorReturn(value);
}
void Evaluator::Visit(CallExpr* state) {
  auto callee = Eval(state->callee());

  std::vector<object::LoxObject> arguments;
  for (Expr argument : state->arguments()) {
    arguments.push_back(Eval(argument));
  }

  auto callable = callee->DownCast<LoxCallable>();
  if (!callable) {
    throw RuntimeError(TreeWalkerError(state->paren(), "Not a callable object"));
  }
  if (arguments.size() != callable->Arity()) {
    throw RuntimeError(TreeWalkerError(state->paren(), "Wrong arg number"));
  }
  try {
    VisitorReturn(callable->Call(this, arguments));
  } catch (const char* msg) {
    throw RuntimeError(TreeWalkerError(state->paren(), msg));
  }
}

void Evaluator::Visit(PrintStmt* state) {
  auto ret_v = Eval(state->expression());
  std::cout << ret_v->ToString() << std::endl;
  VisitorReturn(object::VoidObject());
}
void Evaluator::Visit(ExprStmt* state) {
  Eval(state->expression());
  VisitorReturn(object::VoidObject());
}
void Evaluator::Visit(VarDeclStmt* state) {
  auto value = object::VoidObject();
  if (IsValid(state->initializer())) {
    value = Eval(state->initializer());
  }
  WorkEnv()->Define(state->name()->lexeme, value);
  VisitorReturn(object::VoidObject());
}
void Evaluator::Visit(BlockStmt* state) {
  EnterNewScopeGuard guard(this);
  for (auto& stmt : state->statements()) {
    Eval(stmt);
  }
  VisitorReturn(object::VoidObject());
}
void Evaluator::Visit(IfStmt* state) {
  if (IsValueTrue((Eval(state->condition())))) {
    Eval(state->thenBranch());
  } else if (IsValid(state->elseBranch())) {
    Eval(state->elseBranch());
  }
  VisitorReturn(object::VoidObject());
}
void Evaluator::Visit(WhileStmt* state) {
  while (IsValueTrue((Eval(state->condition())))) {
    try {
      Eval(state->body());
    } catch (RuntimeError& err) {
      if (err.err.SourceToken()->type == TokenType::BREAK) {
        break;
      }
      throw;
    }
  }
  VisitorReturn(object::VoidObject());
}
void Evaluator::Visit(BreakStmt* state) { throw RuntimeError(TreeWalkerError(state->src_token(), "Hit break")); }
void Evaluator::Visit(FunctionStmt* state) {
  LoxFunctionData fn_data{
      .is_init_method = false,
      .closure = WorkEnv(),
      .function = std::static_pointer_cast<FunctionStmt>(state->shared_from_this()),

  };
  auto fn = object::MakeLoxObject<LoxFunction>(fn_data);
  WorkEnv()->Define(state->name()->lexeme, fn);
  // FreezeEnv(); use freeze env to "copy" capture
  VisitorReturn(fn);
}
void Evaluator::Visit(ReturnStmt* state) {
  auto ret = object::VoidObject();
  if (IsValid(state->value())) {
    ret = Eval(state->value());
  }
  throw ReturnValue(ret);
}
void Evaluator::Visit(ClassStmt* state) {
  WorkEnv()->Define(state->name()->lexeme, object::VoidObject());
  auto klass = object::VoidObject();
  {
    EnterNewScopeGuard(this, WorkEnv());
    auto superclass = object::VoidObject();
    if (IsValid(state->superclass())) {
      superclass = Eval(state->superclass());
      if (!superclass->DownCast<LoxClass>()) {
        throw RuntimeError(TreeWalkerError(state->name(), "Base must be a class"));
      }
    }
    LoxClassData class_data{.name = state->name()->lexeme,
                            .superclass = std::static_pointer_cast<LoxClass>(superclass)};
    klass = object::MakeLoxObject<LoxClass>(class_data);
    for (auto& method_stmt : state->methods()) {
      auto method_state = method_stmt->DownCast<FunctionStmt>();
      LoxFunctionData fn_data{
          .is_init_method = method_state->name()->lexeme == "init",
          .closure = WorkEnv(),
          .function = std::static_pointer_cast<FunctionStmt>(method_state->shared_from_this()),
      };
      auto method = object::MakeLoxObject<LoxFunction>(fn_data);
      klass->SetAttr(method_state->name()->lexeme, method);
    }
  }
  WorkEnv()->Set(state->name()->lexeme, klass);
  VisitorReturn(object::VoidObject());
}
void Evaluator::Visit(GetAttrExpr* state) {
  auto object = Eval(state->src_object());
  const auto& attr_name = state->attr_name()->lexeme;
  auto ret = object::VoidObject();
  // Let's handle binding of this
  if (auto instance = object->DownCast<LoxClassInstance>()) {
    if (auto attr = instance->GetAttr(attr_name)) {
      VisitorReturn(attr);
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
          throw RuntimeError(TreeWalkerError(state->attr_name(), klass->ToString() + " is not superclass"));
        }
        ret = fn->BindThis(objcet_this);
      } else {
        throw RuntimeError(TreeWalkerError(state->attr_name(), "Cannot use method out of method"));
      }
    }
  } else {
    ret = object->GetAttr(attr_name);
  }

  if (!IsValid(ret)) {
    throw RuntimeError(TreeWalkerError(state->attr_name(), "No attr found"));
  }
  VisitorReturn(ret);
}
void Evaluator::Visit(SetAttrExpr* state) {
  auto object = Eval(state->src_object());
  auto ret = Eval(state->value());
  object->SetAttr(state->attr_name()->lexeme, Eval(state->value()));
  VisitorReturn(ret);
}
}  // namespace lox
