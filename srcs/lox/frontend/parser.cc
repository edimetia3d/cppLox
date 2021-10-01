//
// License: MIT
//

#include "lox/frontend/parser.h"

#include <iostream>

#include "lox/global_setting.h"
namespace lox {

class ParserError : public PrefixTokenError<ParserError> {
 public:
  using PrefixTokenError<ParserError>::PrefixTokenError;
  static std::string StrName() { return "ParserError"; }
};
template <TokenType type, TokenType... remained_types>
bool Parser::AdvanceIfMatchAny() {
  if (!Check(type)) {
    if constexpr (sizeof...(remained_types) > 0) {
      return AdvanceIfMatchAny<remained_types...>();
    } else {
      return false;
    }
  }
  Advance();
  return true;
}

template <lox::Expr (Parser::*HIGHER_PRECEDENCE_EXPRESSION)(), TokenType... MATCH_TYPES>
Expr Parser::BinaryExpression() {
  // This function is the state_ of  " left_expr (<binary_op> right_expr)* "

  // All token before this->current has been parsed to the expr
  auto expr = (this->*HIGHER_PRECEDENCE_EXPRESSION)();

  // if this->current is matched , we should parse all tokens after
  // this->current into a r_expr, because repeating is allowed, we do it
  // multiple times
  // if this->current is not matched, we could just return the expr
  while (AdvanceIfMatchAny<MATCH_TYPES...>()) {
    Token op = Previous();
    auto r_expr = (this->*HIGHER_PRECEDENCE_EXPRESSION)();
    expr = MakeExpr<BinaryExpr>(expr, op, r_expr);
  }
  // ok now it's done
  return expr;
}

Expr Parser::Or() {
  auto expr = And();

  while (AdvanceIfMatchAny<TokenType::OR>()) {
    Token op = Previous();
    auto r_expr = And();
    expr = MakeExpr<LogicalExpr>(expr, op, r_expr);
  }
  return expr;
}
Expr Parser::And() {
  auto expr = Equality();

  while (AdvanceIfMatchAny<TokenType::AND>()) {
    Token op = Previous();
    auto r_expr = Equality();
    expr = MakeExpr<LogicalExpr>(expr, op, r_expr);
  }
  return expr;
}

lox::Expr lox::Parser::Equality() {
  return BinaryExpression<&Parser::Comparison, TokenType::BANG_EQUAL, TokenType::EQUAL_EQUAL>();
}
lox::Expr lox::Parser::Comparison() {
  return BinaryExpression<&Parser::Term, TokenType::GREATER, TokenType::GREATER_EQUAL, TokenType::LESS,
                          TokenType::LESS_EQUAL>();
}
lox::Expr lox::Parser::Term() { return BinaryExpression<&Parser::Factor, TokenType::MINUS, TokenType::PLUS>(); }
lox::Expr lox::Parser::Factor() { return BinaryExpression<&Parser::Unary, TokenType::SLASH, TokenType::STAR>(); }
lox::Expr lox::Parser::Unary() {
  if (AdvanceIfMatchAny<TokenType::BANG, TokenType::MINUS>()) {
    Token op = Previous();
    auto right = this->Unary();
    return MakeExpr<UnaryExpr>(op, right);
  }
  return Call();
}
lox::Expr lox::Parser::Primary() {
  if (AdvanceIfMatchAny<TokenType::FALSE, TokenType::TRUE, TokenType::NIL, TokenType::NUMBER, TokenType::STRING>())
    return MakeExpr<LiteralExpr>(Previous());

  if (AdvanceIfMatchAny<TokenType::IDENTIFIER, TokenType::THIS>()) {
    return MakeExpr<VariableExpr>(Previous());
  }

  if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
    auto expr = ExpressionExpr();
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
    return MakeExpr<GroupingExpr>(expr);
  }
  throw Error(Peek(), "Primary get unknown token");
}
lox::Token lox::Parser::Consume(lox::TokenType type, const std::string& message) {
  if (Check(type)) return Advance();
  throw Error(Peek(), message);
}
void lox::Parser::Synchronize() {
  Advance();

  while (!IsAtEnd()) {
    if (Previous()->type == TokenType::SEMICOLON) return;

    switch (Peek()->type) {
      case TokenType::CLASS:
      case TokenType::FUN:
      case TokenType::VAR:
      case TokenType::FOR:
      case TokenType::IF:
      case TokenType::WHILE:
      case TokenType::PRINT:
      case TokenType::RETURN:
        return;
      default:
        break;
    }

    Advance();
  }
}
std::vector<lox::Stmt> lox::Parser::Parse() {
  std::vector<lox::Stmt> statements;
  while (!IsAtEnd()) {
    try {
      auto stmt = Declaration();
      statements.push_back(stmt);
    } catch (LoxError& error) {
      Synchronize();
      Parse();
    }
  }
  if (err_found) {
    return {};
  } else {
    return statements;
  }
}
lox::Stmt lox::Parser::Statement() {
  if (AdvanceIfMatchAny<TokenType::IF>()) return If();
  if (AdvanceIfMatchAny<TokenType::WHILE>()) return While();
  if (AdvanceIfMatchAny<TokenType::BREAK, TokenType::CONTINUE>()) return Break();
  if (AdvanceIfMatchAny<TokenType::FOR>()) return ForStmtSugar();
  if (AdvanceIfMatchAny<TokenType::PRINT>()) return Print();
  if (AdvanceIfMatchAny<TokenType::RETURN>()) return Return();
  if (AdvanceIfMatchAny<TokenType::LEFT_BRACE>()) return Block();

  return ExpressionStmt();
}
Stmt Parser::Print() {
  Expr value = ExpressionExpr();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  return MakeStmt<PrintStmt>(value);
}
Stmt Parser::ExpressionStmt() {
  Expr expr = ExpressionExpr();
  if (AdvanceIfMatchAny<TokenType::SEMICOLON>()) {
    return MakeStmt<ExprStmt>(expr);
  } else {
    if (GlobalSetting().interactive_mode) {
      return MakeStmt<PrintStmt>(expr);
    } else {
      Error(Peek(), "Non-interactive mode must have ; after expression");
    }
  }
  return Stmt(nullptr);
}
Stmt Parser::Declaration() {
  if (AdvanceIfMatchAny<TokenType::CLASS>()) return ClassDef();
  if (AdvanceIfMatchAny<TokenType::FUN>()) return FunctionDef("function");
  if (AdvanceIfMatchAny<TokenType::VAR>()) {
    auto name = Consume(TokenType::IDENTIFIER, "Expect IDENTIFIER after key `var`.");
    Expr init_expr(nullptr);
    if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
      init_expr = ExpressionExpr();
    }
    Consume(TokenType::SEMICOLON, "Expect ; when var decl finish.");
    return MakeStmt<VarDeclStmt>(name, init_expr);
  }

  return Statement();
}
Expr Parser::Assignment() {
  Expr expr = Or();

  if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
    Token equals = Previous();
    Expr value = Assignment();  // use recurse to impl the right-associative

    if (auto state = expr->DownCast<VariableExpr>()) {
      Token name = state->name();
      return MakeExpr<AssignExpr>(name, value);
    } else if (auto state = expr->DownCast<GetAttrExpr>()) {
      return MakeExpr<SetAttrExpr>(state->src_object(), state->attr_name(), value);
    }

    Error(equals, "Invalid assignment target.");
  }

  return expr;
}
Stmt Parser::Block() { return MakeStmt<BlockStmt>(GetBlocks()); }
std::vector<Stmt> Parser::GetBlocks() {
  std::vector<Stmt> statements;

  while (!Check(TokenType::RIGHT_BRACE) && !IsAtEnd()) {
    statements.push_back(Declaration());
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after block.");
  return statements;
}
Stmt Parser::If() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'if'.");
  Expr condition = ExpressionExpr();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after if condition.");

  Stmt thenBranch = Statement();
  Stmt elseBranch(nullptr);
  if (AdvanceIfMatchAny<TokenType::ELSE>()) {
    elseBranch = Statement();
  }

  return MakeStmt<IfStmt>(condition, thenBranch, elseBranch);
}
Stmt Parser::While() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'while'.");
  Expr condition = ExpressionExpr();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");
  Stmt body = Statement();
  return MakeStmt<WhileStmt>(condition, body);
}
Stmt Parser::ForStmtSugar() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'for'.");

  Stmt initializer(nullptr);
  if (AdvanceIfMatchAny<TokenType::SEMICOLON>()) {
  } else if (Check(TokenType::VAR)) {
    initializer = Declaration();
  } else {
    initializer = ExpressionStmt();
  }

  Expr condition(nullptr);
  if (!Check(TokenType::SEMICOLON)) {
    condition = ExpressionExpr();
  }
  Consume(TokenType::SEMICOLON, "Expect ';' after loop condition.");

  Expr increment(nullptr);
  if (!Check(TokenType::RIGHT_PAREN)) {
    increment = ExpressionExpr();
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after for clauses.");
  Stmt body = Statement();

  if (IsValid(increment)) {
    std::vector<Stmt> body_with_increasement = {body};
    body_with_increasement.push_back(MakeStmt<ExprStmt>(increment));
    body = MakeStmt<BlockStmt>(body_with_increasement);
  }
  if (!IsValid(condition)) {
    auto tmp_true_token = MakeToken(TokenType::TRUE, "for_sugar_true", Peek()->line);
    condition = MakeExpr<LiteralExpr>(tmp_true_token);
  }
  body = MakeStmt<WhileStmt>(condition, body);

  if (IsValid(initializer)) {
    std::vector<Stmt> body_with_initializer = {initializer, body};
    body = MakeStmt<BlockStmt>(body_with_initializer);
  }

  return body;
}
Stmt Parser::Break() {
  auto src_token = Previous();
  if (src_token->type == TokenType::CONTINUE) {
    Error(Previous(), " 'continue' not supported yet.");
  }
  Consume(TokenType::SEMICOLON, std::string("Expect ';' after ") + src_token->lexeme);
  return MakeStmt<BreakStmt>(src_token);
}
Expr Parser::Call() {
  Expr expr = Primary();
  // We treat a.b as a sugar of GetAttr(a,"b"),but implement it as a new ast node
  // So, technically, only top level symbol will be treated as a variable at runtime, all .x.y follows is
  // just function call.
  // and more , if something like `a.b.c=d` a SetAttr will be called, so `SetAttr(GetAttr(a,"b"),"c",d)`will be called
  while (true) {
    if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
      expr = FinishCall(expr);
    } else if (AdvanceIfMatchAny<TokenType::DOT>()) {
      Token name = Consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
      expr = MakeExpr<GetAttrExpr>(expr, name);
    } else {
      break;
    }
  }

  return expr;
}
Expr Parser::FinishCall(const Expr& callee) {
  std::vector<Expr> arguments;
  if (!Check(TokenType::RIGHT_PAREN)) {
    do {
      if (arguments.size() >= 255) {
        Error(Peek(), "Can't have more than 255 arguments.");
      }
      arguments.push_back(ExpressionExpr());
    } while (AdvanceIfMatchAny<TokenType::COMMA>());
  }

  Token paren = Consume(TokenType::RIGHT_PAREN, "Expect ')' after arguments.");

  return MakeExpr<CallExpr>(callee, paren, arguments);
}
Stmt Parser::FunctionDef(const std::string& kind) {
  Token name = Consume(TokenType::IDENTIFIER, "Expect " + kind + " name.");
  Consume(TokenType::LEFT_PAREN, "Expect '(' after " + kind + " name.");
  std::vector<Token> parameters;
  if (!Check(TokenType::RIGHT_PAREN)) {
    do {
      if (parameters.size() >= 255) {
        Error(Peek(), "Can't have more than 255 parameters.");
      }

      parameters.push_back(Consume(TokenType::IDENTIFIER, "Expect parameter name."));
    } while (AdvanceIfMatchAny<TokenType::COMMA>());
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after parameters.");
  Consume(TokenType::LEFT_BRACE, "Expect '{' before " + kind + " body.");
  std::vector<Stmt> body = GetBlocks();
  return MakeStmt<FunctionStmt>(name, parameters, body);
}
Stmt Parser::Return() {
  Token keyword = Previous();
  Expr value(nullptr);
  if (!Check(TokenType::SEMICOLON)) {
    value = ExpressionExpr();
  }

  Consume(TokenType::SEMICOLON, "Expect ';' after return value.");
  return MakeStmt<ReturnStmt>(keyword, value);
}
Stmt Parser::ClassDef() {
  Token name = Consume(TokenType::IDENTIFIER, "Expect class name.");
  auto superclass = Expr();
  if (AdvanceIfMatchAny<TokenType::LESS>()) {
    Consume(TokenType::IDENTIFIER, "Expect superclass name after '<' .");
    superclass = MakeExpr<VariableExpr>(Previous());
  }
  Consume(TokenType::LEFT_BRACE, "Expect '{' before class body.");

  std::vector<Stmt> methods;
  while (!Check(TokenType::RIGHT_BRACE) && !IsAtEnd()) {
    methods.push_back(FunctionDef("method"));
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after class body.");

  return MakeStmt<ClassStmt>(name, superclass, methods);
}
LoxError Parser::Error(Token token, const std::string& msg) {
  err_found = true;
  ParserError err(token, msg);
  std::cout << err.what() << std::endl;
  return err;
}

}  // namespace lox
