//
// License: MIT
//

#include "lox/frontend/parser.h"

#include <cassert>
#include <iostream>
#include <map>

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
  err_found = false;
  while (!IsAtEnd()) {
    auto stmt = AnyStatement();
    if (!err_found) {
      statements.push_back(std::move(stmt));
    }
  }
  if (err_found) {
    return std::unique_ptr<lox::FunctionStmt>();
  }
  auto script = ASTNode::Make<lox::FunctionStmt>(FunctionStmtAttr{.name = Token(TokenType::IDENTIFIER, "<script>", -1)},
                                                 ExprPtr(), std::move(statements));
  return std::unique_ptr<lox::FunctionStmt>(script.release()->As<lox::FunctionStmt>());
}

StmtPtr lox::Parser::AnyStatement() {
  try {
    return DoAnyStatement();
  } catch (const ParserError& e) {
    Synchronize();
    std::cerr << e.what() << std::endl;
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
  auto comma_expr = ExprPtr();
  if (!Check(TokenType::RIGHT_PAREN)) {
    comma_expr = AnyExpression();
    // AnyExpression may return a single expression when there is no comma.
    if (comma_expr && !comma_expr->DynAs<CommaExpr>()) {
      std::vector<ExprPtr> args;
      args.push_back(std::move(comma_expr));
      comma_expr = ASTNode::Make<CommaExpr>(CommaExprAttr{}, std::move(args));
    }
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after parameters.");
  Consume(TokenType::LEFT_BRACE, "Expect '{' before " + kind + " body.");
  std::vector<StmtPtr> body = GetBlocks();
  return ASTNode::Make<FunctionStmt>(FunctionStmtAttr{.name = name}, std::move(comma_expr), std::move(body));
}
StmtPtr Parser::ReturnStmt() {
  Token keyword = Previous();
  ExprPtr value;
  if (!Check(TokenType::SEMICOLON)) {
    value = AnyExpression();
  }

  Consume(TokenType::SEMICOLON, "Expect ';' after return value.");
  return ASTNode::Make<lox::ReturnStmt>(ReturnStmtAttr{.src_token = keyword}, std::move(value));
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
  err_found = true;
  ParserError err(token->Dump() + ": " + msg);
  throw err;
}

std::shared_ptr<Parser> Parser::Make(ParserType type, Scanner* scanner) {
  switch (type) {
    case ParserType::RECURSIVE_DESCENT:
      return std::shared_ptr<Parser>(new RecursiveDescentParser(scanner));
    case ParserType::PRATT_PARSER:
      return std::shared_ptr<Parser>(new PrattParser(scanner));
    default:
      return nullptr;
  }
}
std::shared_ptr<Parser> Parser::Make(std::string type, Scanner* scanner) {
  static std::map<std::string, ParserType> _map = {{std::string("RecursiveDescent"), ParserType::RECURSIVE_DESCENT},
                                                   {std::string("PrattParser"), ParserType::PRATT_PARSER}};
  return Make(_map[type], scanner);
}

ExprPtr ParserWithExprUtils::ParseCallExpr(ExprPtr expr) {
  ExprPtr arguments;
  if (!Check(TokenType::RIGHT_PAREN)) {
    arguments = ForceCommaExpr();
  }
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after arguments.");
  expr = ASTNode::Make<CallExpr>(CallExprAttr{}, std::move(expr), std::move(arguments));
  return expr;
}

ExprPtr ParserWithExprUtils::ParseAssignOrSetAttr(ExprPtr left_expr, ExprPtr right_expr, Token equal_token) {
  if (auto var_node = left_expr->DynAs<VariableExpr>()) {
    Token name = var_node->attr->name;
    return ASTNode::Make<lox::AssignExpr>(AssignExprAttr{.name = name}, std::move(right_expr));
  } else if (auto get_attr_node = left_expr->DynAs<GetAttrExpr>()) {
    return ASTNode::Make<SetAttrExpr>(SetAttrExprAttr{.attr_name = get_attr_node->attr->attr_name},
                                      std::move(get_attr_node->src_object), std::move(right_expr));
  }
  Error(equal_token, "Only identifier or attribute can be assigned");
  return ExprPtr();
}
ExprPtr ParserWithExprUtils::ParseGetItemExpr(ExprPtr left_side) {
  ExprPtr item = AnyExpression();
  Consume(TokenType::RIGHT_SQUARE, "Expect ']' after get item.");
  return ASTNode::Make<GetItemExpr>(GetItemExprAttr{}, std::move(left_side), std::move(item));
}

ExprPtr ParserWithExprUtils::ForceCommaExpr() {
  // AnyExpression may return a single expression when there is no comma.
  auto comma_expr = AnyExpression();
  if (comma_expr && !comma_expr->DynAs<CommaExpr>()) {
    std::vector<ExprPtr> elements;
    elements.push_back(std::move(comma_expr));
    comma_expr = ASTNode::Make<CommaExpr>(CommaExprAttr{}, std::move(elements));
  }
  return comma_expr;
}
ExprPtr ParserWithExprUtils::ParseTensorExpr() {
  auto src_token = Previous();
  Consume(TokenType::LEFT_PAREN, "Expect '(' after Tensor");
  auto args = ForceCommaExpr();
  auto p_comma_expr = args->As<lox::CommaExpr>();
  Consume(TokenType::RIGHT_PAREN, "Expect ')' after Tensor");
  return ASTNode::Make<TensorExpr>(TensorExprAttr{.src_token = src_token}, std::move(p_comma_expr->elements[0]),
                                   std::move(p_comma_expr->elements[1]), std::move(p_comma_expr->elements[2]));
}

template <lox::ExprPtr (RecursiveDescentParser::*HIGHER_PRECEDENCE_EXPRESSION)(),
          TokenType... SAME_PRECEDENCE_OPRATOR_TOKEN_TYPES>
ExprPtr RecursiveDescentParser::BinaryExpr() {
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

ExprPtr RecursiveDescentParser::AnyExpression() { return CommaExpr(); }

ExprPtr RecursiveDescentParser::CommaExpr() {
  ExprPtr ret = AssignExpr();
  if (AdvanceIfMatchAny<TokenType::COMMA>()) {
    std::vector<ExprPtr> elements;
    Token src_token = Previous();
    elements.push_back(std::move(ret));
    elements.push_back(AssignExpr());
    while (AdvanceIfMatchAny<TokenType::COMMA>()) {
      elements.push_back(AssignExpr());
    }
    ret = ASTNode::Make<lox::CommaExpr>(CommaExprAttr{.src_token = src_token}, std::move(elements));
  }
  return ret;
}

ExprPtr RecursiveDescentParser::AssignExpr() {
  ExprPtr expr = OrExpr();

  if (AdvanceIfMatchAny<TokenType::EQUAL>()) {
    Token equal_token = Previous();
    ExprPtr right_expr = AssignExpr();  // use recurse to impl the right-associative
    return ParseAssignOrSetAttr(std::move(expr), std::move(right_expr), equal_token);
  }

  return expr;
}

ExprPtr RecursiveDescentParser::OrExpr() {
  auto left_expr = AndExpr();

  while (AdvanceIfMatchAny<TokenType::OR>()) {
    Token op = Previous();
    auto right_expr = AndExpr();
    left_expr = ASTNode::Make<LogicalExpr>(LogicalExprAttr{.op = op}, std::move(left_expr), std::move(right_expr));
  }
  return left_expr;
}
ExprPtr RecursiveDescentParser::AndExpr() {
  auto left_expr = EqualityExpr();

  while (AdvanceIfMatchAny<TokenType::AND>()) {
    Token op = Previous();
    auto right_expr = EqualityExpr();
    left_expr = ASTNode::Make<LogicalExpr>(LogicalExprAttr{.op = op}, std::move(left_expr), std::move(right_expr));
  }
  return left_expr;
}

ExprPtr RecursiveDescentParser::EqualityExpr() {
  return BinaryExpr<&RecursiveDescentParser::ComparisonExpr, TokenType::BANG_EQUAL, TokenType::EQUAL_EQUAL>();
}
ExprPtr RecursiveDescentParser::ComparisonExpr() {
  return BinaryExpr<&RecursiveDescentParser::TermExpr, TokenType::GREATER, TokenType::GREATER_EQUAL, TokenType::LESS,
                    TokenType::LESS_EQUAL>();
}
ExprPtr RecursiveDescentParser::TermExpr() {
  return BinaryExpr<&RecursiveDescentParser::FactorExpr, TokenType::MINUS, TokenType::PLUS>();
}
ExprPtr RecursiveDescentParser::FactorExpr() {
  return BinaryExpr<&RecursiveDescentParser::UnaryExpr, TokenType::SLASH, TokenType::STAR>();
}
ExprPtr RecursiveDescentParser::UnaryExpr() {
  if (AdvanceIfMatchAny<TokenType::BANG, TokenType::MINUS>()) {
    Token op = Previous();
    auto right = UnaryExpr();  // just use recurse to parse multi unary operator with right-associative
    return ASTNode::Make<lox::UnaryExpr>(UnaryExprAttr{.op = op}, std::move(right));
  }
  return CallExpr();
}

ExprPtr RecursiveDescentParser::CallExpr() {
  ExprPtr expr = Primary();
  // 1. We treat `a.b.c...` as a call to GetAttr, and `a.b.c=d` as a call to SetAttr
  // 2. operator dot and operator left_paren has same precedence
  while (true) {
    if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
      expr = ParseCallExpr(std::move(expr));
    } else if (AdvanceIfMatchAny<TokenType::DOT>()) {
      Token name = Consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
      expr = ASTNode::Make<GetAttrExpr>(GetAttrExprAttr{.attr_name = name}, std::move(expr));
    } else if (AdvanceIfMatchAny<TokenType::LEFT_SQUARE>()) {
      expr = ParseGetItemExpr(std::move(expr));
      break;
    } else {
      break;
    }
  }

  return expr;
}

ExprPtr RecursiveDescentParser::Primary() {
  if (AdvanceIfMatchAny<TokenType::FALSE_TOKEN, TokenType::TRUE_TOKEN, TokenType::NIL, TokenType::NUMBER,
                        TokenType::STRING>())
    return ASTNode::Make<LiteralExpr>(LiteralExprAttr{.value = Previous()});

  if (AdvanceIfMatchAny<TokenType::IDENTIFIER, TokenType::THIS, TokenType::SUPER>()) {
    return ASTNode::Make<VariableExpr>(VariableExprAttr{.name = Previous()});
  }

  if (AdvanceIfMatchAny<TokenType::TENSOR>()) {
    return ParseTensorExpr();
  }

  if (AdvanceIfMatchAny<TokenType::LEFT_PAREN>()) {
    auto expr = AnyExpression();
    Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
    return ASTNode::Make<GroupingExpr>(GroupingExprAttr{}, std::move(expr));
  }

  if (AdvanceIfMatchAny<TokenType::LEFT_SQUARE>()) {
    if (AdvanceIfMatchAny<TokenType::RIGHT_SQUARE>()) {
      return ASTNode::Make<ListExpr>(ListExprAttr{}, ExprPtr());
    }
    auto src_token = Previous();
    auto expr = ForceCommaExpr();
    Consume(TokenType::RIGHT_SQUARE, "Expect ']' after list expression.");
    return ASTNode::Make<lox::ListExpr>(ListExprAttr{.src_token = src_token}, std::move(expr));
  }

  Advance();  // Consume the error token and error
  Error(Peek(), "Primary get unknown token");
  return nullptr;
}

InfixOpInfoMap::InfixOpInfoMap() {
#define RULE_ITEM(TOKEN_T, PRECEDENCE_V, ASSOCIATIVITY_V)                                      \
  {                                                                                            \
    TokenType::TOKEN_T, { InfixPrecedence::PRECEDENCE_V, InfixAssociativity::ASSOCIATIVITY_V } \
  }
  // clang-format off
  auto map_tmp = std::map<TokenType,InfixOpInfo> {
  RULE_ITEM(COMMA         , COMMA                , LEFT_TO_RIGHT) ,
  RULE_ITEM(LEFT_PAREN    , CALL_OR_DOT_OR_INDEX , LEFT_TO_RIGHT) ,
  RULE_ITEM(LEFT_SQUARE   , CALL_OR_DOT_OR_INDEX , LEFT_TO_RIGHT) ,
  RULE_ITEM(DOT           , CALL_OR_DOT_OR_INDEX , LEFT_TO_RIGHT) ,
  RULE_ITEM(MINUS         , TERM                 , LEFT_TO_RIGHT) ,
  RULE_ITEM(PLUS          , TERM                 , LEFT_TO_RIGHT) ,
  RULE_ITEM(SLASH         , FACTOR               , LEFT_TO_RIGHT) ,
  RULE_ITEM(STAR          , FACTOR               , LEFT_TO_RIGHT) ,
  RULE_ITEM(BANG_EQUAL    , EQUALITY             , LEFT_TO_RIGHT) ,
  RULE_ITEM(EQUAL         , ASSIGNMENT           , RIGHT_TO_LEFT) ,
  RULE_ITEM(EQUAL_EQUAL   , EQUALITY             , LEFT_TO_RIGHT) ,
  RULE_ITEM(GREATER       , COMPARISON           , LEFT_TO_RIGHT) ,
  RULE_ITEM(GREATER_EQUAL , COMPARISON           , LEFT_TO_RIGHT) ,
  RULE_ITEM(LESS          , COMPARISON           , LEFT_TO_RIGHT) ,
  RULE_ITEM(LESS_EQUAL    , COMPARISON           , LEFT_TO_RIGHT) ,
  RULE_ITEM(AND           , AND                  , LEFT_TO_RIGHT) ,
  RULE_ITEM(OR            , OR                   , LEFT_TO_RIGHT) ,
  };

  // clang-format on
#undef RULE_ITEM

  data = std::move(map_tmp);
}
InfixOpInfoMap& InfixOpInfoMap::Instance() {
  static InfixOpInfoMap instance;
  return instance;
}
InfixOpInfoMap::InfixOpInfo* InfixOpInfoMap::Get(TokenType type) {
  if (InfixOpInfoMap::Instance().data.contains(type)) {
    return &InfixOpInfoMap::Instance().data[type];
  }
  return nullptr;
}
ExprPtr PrattParser::DoAnyExpression(InfixPrecedence lower_bound) {
  Advance();
  auto ret = PrefixExpr();
  while (auto op_info = InfixOpInfoMap::Get(current)) {
    if ((op_info->precedence > lower_bound ||
         (op_info->precedence == lower_bound && op_info->associativity == InfixAssociativity::RIGHT_TO_LEFT))) {
      Advance();
      ret = InfixExpr(std::move(ret));
    } else {
      break;
    }
  }
  return ret;
}
ExprPtr PrattParser::PrefixExpr() {
  auto bak_previous = previous;
  switch (previous->type) {
    case TokenType::LEFT_PAREN: {
      auto expr = AnyExpression();
      Consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
      return ASTNode::Make<GroupingExpr>(GroupingExprAttr{}, std::move(expr));
    }
    case TokenType::LEFT_SQUARE: {
      if (Peek()->type == TokenType::RIGHT_SQUARE) {
        Advance();
        return ASTNode::Make<lox::ListExpr>(ListExprAttr{.src_token = bak_previous}, nullptr);
      }
      auto expr = ForceCommaExpr();
      Consume(TokenType::RIGHT_SQUARE, "Expect ']' after expression.");
      return ASTNode::Make<ListExpr>(ListExprAttr{.src_token = bak_previous}, std::move(expr));
    }
    case TokenType::MINUS:
      [[fallthrough]];
    case TokenType::BANG: {
      auto right_expr = DoAnyExpression(InfixPrecedence::UNARY);
      return ASTNode::Make<UnaryExpr>(UnaryExprAttr{.op = bak_previous}, std::move(right_expr));
    }
    case TokenType::SUPER:
      [[fallthrough]];
    case TokenType::THIS:
      [[fallthrough]];
    case TokenType::IDENTIFIER: {
      return ASTNode::Make<VariableExpr>(VariableExprAttr{.name = bak_previous});
    }
    case TokenType::STRING:
      [[fallthrough]];
    case TokenType::NUMBER:
      [[fallthrough]];
    case TokenType::TRUE_TOKEN:
      [[fallthrough]];
    case TokenType::FALSE_TOKEN:
      [[fallthrough]];
    case TokenType::NIL:
      return ASTNode::Make<LiteralExpr>(LiteralExprAttr{.value = bak_previous});
    case TokenType::TENSOR: {
      return ParseTensorExpr();
    }
    default:
      Error(previous, "Expect expression.");
  }
  return ExprPtr();
}

ExprPtr PrattParser::InfixExpr(ExprPtr left_side_expr) {
  Token bak_previous = previous;
  switch (previous->type) {
    case TokenType::LEFT_PAREN: {
      return ParseCallExpr(std::move(left_side_expr));
    }
    case TokenType::LEFT_SQUARE: {
      return ParseGetItemExpr(std::move(left_side_expr));
    }
    case TokenType::COMMA: {
      auto right_expr = DoAnyExpression(InfixPrecedence::COMMA);
      if (left_side_expr->DynAs<CommaExpr>()) {
        // we need to flatten the comma expression
        left_side_expr->As<CommaExpr>()->elements.push_back(std::move(right_expr));
        return ASTNode::Make<CommaExpr>(CommaExprAttr{.src_token = bak_previous},
                                        std::move(left_side_expr->As<CommaExpr>()->elements));
      } else {
        std::vector<ExprPtr> exprs;
        exprs.push_back(std::move(left_side_expr));
        exprs.push_back(std::move(right_expr));
        return ASTNode::Make<CommaExpr>(CommaExprAttr{.src_token = bak_previous}, std::move(exprs));
      }
    }
    case TokenType::DOT: {
      Token name = Consume(TokenType::IDENTIFIER, "Expect property name after '.'.");
      return ASTNode::Make<GetAttrExpr>(GetAttrExprAttr{.attr_name = name}, std::move(left_side_expr));
    }
    case TokenType::EQUAL: {
      auto right_expr = DoAnyExpression(InfixPrecedence::ASSIGNMENT);
      return ParseAssignOrSetAttr(std::move(left_side_expr), std::move(right_expr), bak_previous);
    }
    case TokenType::MINUS:
      [[fallthrough]];
    case TokenType::PLUS:
      [[fallthrough]];
    case TokenType::SLASH:
      [[fallthrough]];
    case TokenType::STAR:
      [[fallthrough]];
    case TokenType::BANG_EQUAL:
      [[fallthrough]];
    case TokenType::EQUAL_EQUAL:
      [[fallthrough]];
    case TokenType::GREATER:
      [[fallthrough]];
    case TokenType::GREATER_EQUAL:
      [[fallthrough]];
    case TokenType::LESS:
      [[fallthrough]];
    case TokenType::LESS_EQUAL: {
      auto right_expr = DoAnyExpression(InfixOpInfoMap::Get(bak_previous)->precedence);
      return ASTNode::Make<BinaryExpr>(BinaryExprAttr{.op = bak_previous}, std::move(left_side_expr),
                                       std::move(right_expr));
    }
    case TokenType::AND:
      [[fallthrough]];
    case TokenType::OR: {
      auto right_expr = DoAnyExpression(InfixOpInfoMap::Get(bak_previous)->precedence);
      return ASTNode::Make<LogicalExpr>(LogicalExprAttr{.op = bak_previous}, std::move(left_side_expr),
                                        std::move(right_expr));
    }
    default:
      Error(bak_previous, "Expect expression.");
  }
  return ExprPtr();
}
}  // namespace lox
