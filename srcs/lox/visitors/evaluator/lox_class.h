//
// Created by edi on 8/18/21.
//

#ifndef CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
#define CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_

#include <string>

#include "lox/ast/stmt.h"
#include "lox/lox_object/lox_object.h"
namespace lox {
class LoxClassState : public object::LoxObjectState {
 public:
  explicit LoxClassState(std::string name) : name_(name) {}

 private:
  std::string name_;
};
}  // namespace lox

#endif  // CPPLOX_SRCS_LOX_VISITORS_EVALUATOR_LOX_CLASS_H_
