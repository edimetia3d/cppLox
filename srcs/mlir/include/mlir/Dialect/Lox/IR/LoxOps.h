//
// License: MIT
//

/**
 * @file LoxInterface.h
 * @brief Contains the Lox operations.
 */

#ifndef LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXOPS_H
#define LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXOPS_H

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Ops usually will depend on types/interfaces of the dialect.
#include "mlir/Dialect/Lox/IR/LoxInterface.h"
#include "mlir/Dialect/Lox/IR/LoxTypes.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Lox/IR/Lox.h.inc"

#endif // LOX_SRCS_MLIR_INCLUDE_MLIR_DIALECT_LOX_IR_LOXOPS_H
