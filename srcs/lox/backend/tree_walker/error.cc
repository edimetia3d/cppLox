
#include "lox/backend/tree_walker/error.h"
lox::TreeWalkerError::TreeWalkerError(const Token& token, const std::string& message)
    : LoxError(token->Str() + " what(): " + message) {}
