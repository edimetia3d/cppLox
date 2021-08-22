//
// License: MIT
//

#ifndef CPPLOX_INCLUDES_LOX_TOKEN_H_
#define CPPLOX_INCLUDES_LOX_TOKEN_H_

#include <memory>
#include <string>
#include <vector>

#include "lox/token_type.h"
namespace lox {

class TokenBase;
class AstNode;
using Token = std::shared_ptr<TokenBase>;

class TokenBase {
 public:
  static Token Make(const TokenType& type, const std::string& lexeme, int line) {
    return std::shared_ptr<TokenBase>(new TokenBase(type, lexeme, line));
  }

  static TokenType GetIdentifierType(const std::string& identifier);
  std::string Str() const;

  std::string lexeme_;
  TokenType type_;
  int line_;

 protected:
  explicit TokenBase(const TokenType& type, const std::string& lexeme, int line)
      : type_(type), line_(line), lexeme_(std::move(lexeme)) {}
};

static inline Token MakeToken(const TokenType& type, const std::string& lexeme, int line) {
  return TokenBase::Make(type, lexeme, line);
}

static inline void BindParent(Token token, AstNode* parent) {
  // token do not need parent;
}
static inline void BindParent(const std::vector<Token>& tokens, AstNode* parent) {
  // token do not need parent;
}
static inline bool IsModified(Token token) { return false; }

static inline bool IsModified(const std::vector<Token>& tokens) { return false; }

static inline void ResetModify(Token token) {
  // token do not need parent;
}
static inline void ResetModify(const std::vector<Token>& tokens) {
  // token do not need parent;
}
}  // namespace lox
#endif  // CPPLOX_INCLUDES_LOX_TOKEN_H_
