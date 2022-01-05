//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_VIRTUAL_MACHINE_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_VIRTUAL_MACHINE_H_
#include "lox/backend/backend.h"
#include "lox/frontend/scanner.h"

namespace lox::vm {
class VirtualMachine : public BackEnd {
 public:
  LoxError Run(Scanner& scanner) override;
};
}  // namespace lox
#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_VIRTUAL_MACHINE_H_
