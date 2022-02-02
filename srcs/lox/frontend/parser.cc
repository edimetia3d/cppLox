//
// License: MIT
//

#include "lox/frontend/parser.h"

#include <iostream>

#include "lox/common/global_setting.h"
namespace lox {

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

template <lox::ExprPtr (Parser::*HIGHER_PRECEDENCE_EXPRESSION)(), TokenType... SAME_PRECEDENCE_OPRATOR_TOKEN_TYPES>
ExprPtr Parser::BinaryExpr() {
  // This function is the " left_expr (op right_expr)* "

  // All token before this->current has been parsed into the left_expr
  auto left_expr = (this->*HIGHER_PRECEDENCE_EXPRESSION)();

  // if this->current is the operator token, we should parse an expression as right_expr.
  // if this->current is not matched, we could just return the left_expr
  // Because there could be multi same precedence operator, we use a loop to parse multiple times
  while (AdvanceIfMatchAny<SAME_PRECEDENCE_OPRATOR_TOKEN_TYPES...>()) {
    Token op = Previous();
    auto right_expr = (this->*HIGHER_PRECEDENCE_EXPRESSION)();
    left_expr = ASTNode::Make<lox::BinaryExpr>(BinaryExprAttr{.op = op}, std::move(left_expr), std::move(right_expr));
  }
  // ok now it's done
  return left_expr;
}

ExprPtr Parser::AnyExpression() { return AssignExpr(); }

ExprPtr Parser::AssignExpr() {
  ExprPtr expr = OrExpr();

  if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
    Token equals = Previous();
    ExprPtr value = AssignExpr();  // use recurse to impl the right-associative

    if (auto var_node = expr->DynAs<VariableExpr>()) {
      Token name = var_node->attr->name;
      return ASTNode::Make<lox::AssignExpr>(AssignExprAttr{.name = name}, std::move(value));
    } else if (auto get_attr_node = expr->DynAs<GetAttrExpr>()) {
      return ASTNode::Make<SetAttrExpr>(SetAttrExprAttr{.attr_name = get_attr_node->attr->attr_name},
                                        std::move(get_attr_node->src_object), std::move(value));
    }
    Error(equals, "Only identifier or attribute can be assigned");
  }

  return expr;
}

ExprPtr Parser::OrExpr() {
  auto left_expr = AndExpr();

  while (AdvanceIfMatchAny<TokenType::OR>()) {
    Token op = Previous();
    auto right_expr = AndExpr();
    left_expr = ASTNode::Make<LogicalExpr>(LogicalExprAttr{.op = op}, std::move(left_expr), std::move(right_expr));
  }
  return left_expr;
}
ExprPtr Parser::AndExpr() {
  auto left_expr = EqualityExpr();

  while (AdvanceIfMatchAny<TokenType::AND>()) {
    Token op = Previous();
    auto right_expr = EqualityExpr();
    left_expr = ASTNode::Make<LogicalExpr>(LogicalExprAttr{.op = op}, std::move(left_expr), std::move(right_expr));
  }
  return left_expr;
}

ExprPtr lox::Parser::EqualityExpr() {
  return BinaryExpr<&Parser::ComparisonExpr, TokenType::BANG_EQUAL, TokenType::EQUAL_EQUAL>();
}
ExprPtr lox::Parser::ComparisonExpr() {
  return BinaryExpr<&Parser::TermExpr, TokenType::GREATER, TokenType::GREATER_EQUAL, TokenType::LESS,
                    TokenType::LESS_EQUAL>();
}
ExprPtr lox::Parser::TermExpr() { return BinaryExpr<&Parser::FactorExpr, TokenType::MINUS, TokenType::PLUS>(); }
ExprPtr lox::Parser::FactorExpr() { return BinaryExpr<&Parser::UnaryExpr, TokenType::SLASH, TokenType::STAR>(); }
ExprPtr lox::Parser::UnaryExpr() {
  if (AdvanceIfMatchAny<TokenType::BANG, TokenType::MINUS>()) {
    Token op = Previous();
    auto right = UnaryExpr();  // just use recurse to parse multi unary operator with right-associative
    return ASTNode::Make<lox::UnaryExpr>(UnaryExprAttr{.op = op}, std::move(right));
  }
  return CallExpr();
}

ExprPtr Parser::CallExpr() {
  ExprPtr expr = Primary();
  // 1. We treat `a.b.c...` as a call to GetAttr, and `a.b.c=d` as a call to SetAttr
  // 2. operator dot and operator left_paren has same precedence
  while (true) {
    if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
      std::vector<ExprPtr> arguments;
      if (!Check(TokenType::RIGHT_PAREN)) {
        do {
          arguments.push_back(AnyExpression());
        } while (AdvanceIfMatchAny<TokenType::COMMA>());
      }
      Consume(TokenType::RIGHT_PAREN, "Expect ')' after arguments.");
      expr = ASTNode::Make<lox::CallExpr>(CallExprAttr{}, std::move(expr), std::move(arguments));
    } else if (AdvanceIfMatchAny<TokenType::DOT>()) {
      Token name = Consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
      expr = ASTNode::Make<GetAttrExpr>(GetAttrExprAttr{.attr_name = name}, std::move(expr));
    } else {
      break;
    }
  }

  return expr;
}

ExprPtr lox::Parser::Primary() {
  if (AdvanceIfMatchAny<TokenType::FALSE_TOKEN, TokenType::TRUE_TOKEN, TokenType::NIL, TokenType::NUMBER,
                        TokenType::STRING>())
    return ASTNode::Make<LiteralExpr>(LiteralExprAttr{.value = Previous()});

  if (AdvanceIfMatchAny<TokenType::IDENTIFIER, TokenType::THIS, TokenType::SUPER>()) {
    return ASTNode::Make<VariableExpr>(VariableExprAttr{.name = Previous()});
  }

  if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
    auto expr = AnyExpression();
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
    return ASTNode::Make<GroupingExpr>(GroupingExprAttr{}, std::move(expr));
  }
  Error(Peek(), "Primary get unknown token");
  return nullptr;
}
lox::Token lox::Parser::Consume(lox::TokenType type, const std::string& message) {
  if (Check(type)) return Advance();
  Error(Peek(), message);
  return Token{};
}
void lox::Parser::Synchronize() {
  while (!IsAtEnd()) {
    switch (Previous()->type) {
      case TokenType::SEMICOLON:
        return;
      default:
        break;  // do nothing
    };
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
        break;  // do nothing
    }

    Advance();
  }
}
std::unique_ptr<lox::FunctionStmt> lox::Parser::Parse() {
  std::vector<StmtPtr> statements;
  bool err_found = false;
  while (!IsAtEnd()) {
    auto stmt = AnyStatement();
    if (!stmt) {
      err_found = true;
      statements.clear();
    }
    if (!err_found) {
      statements.push_back(std::move(stmt));
    }
  }
  if (err_found) {
    return std::unique_ptr<lox::FunctionStmt>();
  }
  auto script = ASTNode::Make<lox::FunctionStmt>(FunctionStmtAttr{.name = Token(TokenType::IDENTIFIER, "<script>", -1)},
                                                 std::move(statements));
  return std::unique_ptr<lox::FunctionStmt>(script.release()->As<lox::FunctionStmt>());
}

StmtPtr lox::Parser::AnyStatement() {
  try {
    return DoAnyStatement();
  } catch (const ParserError& e) {
    Synchronize();
    return StmtPtr();
  }
}

StmtPtr lox::Parser::DoAnyStatement() {
  if (AdvanceIfMatchAny<TokenType::CLASS>()) {
    return ClassDefStmt();
  } else if (AdvanceIfMatchAny<TokenType::FUN>()) {
    return FunStmt("function");
  } else if (AdvanceIfMatchAny<TokenType::VAR>()) {
    return VarDefStmt();
  } else if (AdvanceIfMatchAny<TokenType::IF>()) {
    return IfStmt();
  } else if (AdvanceIfMatchAny<TokenType::WHILE>()) {
    return WhileStmt();
  } else if (AdvanceIfMatchAny<TokenType::BREAK, TokenType::CONTINUE>()) {
    return BreakStmt();
  } else if (AdvanceIfMatchAny<TokenType::FOR>()) {
    return ForStmt();
  } else if (AdvanceIfMatchAny<TokenType::PRINT>()) {
    return PrintStmt();
  } else if (AdvanceIfMatchAny<TokenType::RETURN>()) {
    return ReturnStmt();
  } else if (AdvanceIfMatchAny<TokenType::LEFT_BRACE>()) {
    return BlockStmt();
  } else {
    return ExpressionStmt();
  }
}
StmtPtr lox::Parser::VarDefStmt() {
  auto name = Consume(TokenType::IDENTIFIER, "Expect IDENTIFIER after key `var`.");
  ExprPtr init_expr;
  if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
    init_expr = AnyExpression();
  }
  Consume(TokenType::SEMICOLON, "Expect ; when var decl finish.");
  return ASTNode::Make<lox::VarDeclStmt>(VarDeclStmtAttr{.name = name}, std::move(init_expr));
}
StmtPtr Parser::PrintStmt() {
  ExprPtr value = AnyExpression();
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  return ASTNode::Make<lox::PrintStmt>(PrintStmtAttr{}, std::move(value));
}
StmtPtr Parser::ExpressionStmt() {
  ExprPtr expr = AnyExpression();
  if (GlobalSetting().interactive_mode && !Check(TokenType::SEMICOLON)) {
    return ASTNode::Make<lox::PrintStmt>(PrintStmtAttr{}, std::move(expr));
  }
  Consume(TokenType::SEMICOLON, "Expect ';' after value.");
  return ASTNode::Make<ExprStmt>(ExprStmtAttr{}, std::move(expr));
}

StmtPtr Parser::BlockStmt() { return ASTNode::Make<lox::BlockStmt>(BlockStmtAttr{}, GetBlocks()); }
std::vector<StmtPtr> Parser::GetBlocks() {
  std::vector<StmtPtr> statements;

  while (!Check(TokenType::RIGHT_BRACE) && !IsAtEnd()) {
    statements.push_back(AnyStatement());
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after BlockStmt.");
  return statements;
}
StmtPtr Parser::IfStmt() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'if'.");
  ExprPtr condition = AnyExpression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after if condition.");

  StmtPtr thenBranch = AnyStatement();
  StmtPtr elseBranch(nullptr);
  if (AdvanceIfMatchAny<TokenType::ELSE>()) {
    elseBranch = AnyStatement();
  }

  return ASTNode::Make<lox::IfStmt>(IfStmtAttr{}, std::move(condition), std::move(thenBranch), std::move(elseBranch));
}
StmtPtr Parser::WhileStmt() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'while'.");
  ExprPtr condition = AnyExpression();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after condition.");
  StmtPtr body = AnyStatement();
  return ASTNode::Make<lox::WhileStmt>(WhileStmtAttr{}, std::move(condition), std::move(body));
}
StmtPtr Parser::ForStmt() {
  Consume(TokenType::LEFT_PAREN, "Expect '(' after 'for'.");

  StmtPtr initializer(nullptr);
  if (AdvanceIfMatchAny<TokenType::SEMICOLON>()) {
  } else if (Check(TokenType::VAR)) {
    initializer = AnyStatement();
  } else {
    initializer = ExpressionStmt();
  }

  ExprPtr condition(nullptr);
  if (!Check(TokenType::SEMICOLON)) {
    condition = AnyExpression();
  }
  Consume(TokenType::SEMICOLON, "Expect ';' after loop condition.");

  ExprPtr increment(nullptr);
  if (!Check(TokenType::RIGHT_PAREN)) {
    increment = AnyExpression();
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after for clauses.");
  StmtPtr body = AnyStatement();

  return ASTNode::Make<lox::ForStmt>(ForStmtAttr{}, std::move(initializer), std::move(condition), std::move(increment),
                                     std::move(body));
}
StmtPtr Parser::BreakStmt() {
  auto src_token = Previous();
  Consume(TokenType::SEMICOLON, std::string("Expect ';' after ") + src_token->lexeme);
  return ASTNode::Make<lox::BreakStmt>(BreakStmtAttr{.src_token = src_token});
}

StmtPtr Parser::FunStmt(const std::string& kind) {
  Token name = Consume(TokenType::IDENTIFIER, "Expect " + kind + " name.");
  Consume(TokenType::LEFT_PAREN, "Expect '(' after " + kind + " name.");
  std::vector<Token> parameters;
  if (!Check(TokenType::RIGHT_PAREN)) {
    do {
      parameters.push_back(Consume(TokenType::IDENTIFIER, "Expect parameter name."));
    } while (AdvanceIfMatchAny<TokenType::COMMA>());
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after parameters.");
  Consume(TokenType::LEFT_BRACE, "Expect '{' before " + kind + " body.");
  std::vector<StmtPtr> body = GetBlocks();
  return ASTNode::Make<FunctionStmt>(FunctionStmtAttr{.name = name, .params = parameters}, std::move(body));
}
StmtPtr Parser::ReturnStmt() {
  Token keyword = Previous();
  ExprPtr value;
  if (!Check(TokenType::SEMICOLON)) {
    value = AnyExpression();
  }

  Consume(TokenType::SEMICOLON, "Expect ';' after return value.");
  return ASTNode::Make<lox::ReturnStmt>(ReturnStmtAttr{.keyword = keyword}, std::move(value));
}
StmtPtr Parser::ClassDefStmt() {
  Token name = Consume(TokenType::IDENTIFIER, "Expect class name.");
  auto superclass = ExprPtr();
  if (AdvanceIfMatchAny<TokenType::LESS>()) {
    Consume(TokenType::IDENTIFIER, "Expect superclass name after '<' .");
    superclass = ASTNode::Make<VariableExpr>(VariableExprAttr{.name = Previous()});
  }
  Consume(TokenType::LEFT_BRACE, "Expect '{' before class body.");

  std::vector<StmtPtr> methods;
  while (!Check(TokenType::RIGHT_BRACE) && !IsAtEnd()) {
    methods.push_back(FunStmt("method"));
  }

  Consume(TokenType::RIGHT_BRACE, "Expect '}' after class body.");

  return ASTNode::Make<ClassStmt>(ClassStmtAttr{.name = name}, std::move(superclass), std::move(methods));
}
void Parser::Error(Token token, const std::string& msg) {
  ParserError err(token->Dump() + ": " + msg);
  throw err;
}

}  // namespace lox
