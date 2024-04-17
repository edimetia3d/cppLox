//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_VIRTUAL_MACHINE_H_
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_VIRTUAL_MACHINE_H_
#include "lox/backend/backend.h"
#include "lox/backend/virtual_machine/core/vm.h"
#include "lox/frontend/scanner.h"

namespace lox::vm {
class VirtualMachine : public BackEnd {
public:
  void Run(Scanner &scanner) override;

private:
  VM vm_;
};
} // namespace lox::vm
#endif // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_VIRTUAL_MACHINE_H_
