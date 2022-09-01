//
// License: MIT
//
#include <mlir/IR/DialectImplementation.h>

#ifndef LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_PRINT_DIALECT_TYPE_H
#define LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_PRINT_DIALECT_TYPE_H
namespace {
/***
 * Mimic a for loop by calling LoxT::parse(parser) in order, and return the first one that succeeds.
 */
template <class LoxT, class... RemainedT> static mlir::Type TypedParse(mlir::DialectAsmParser &parser) {
  llvm::Expected<LoxT> expect = LoxT::parse(parser);
  if (expect) {
    return expect.get();
  } else {
    if constexpr (sizeof...(RemainedT) > 0) {
      if (errorToErrorCode(expect.takeError()).value() == static_cast<int>(std::errc::wrong_protocol_type)) {
        return TypedParse<RemainedT...>(parser);
      } else {
        return mlir::Type();
      }
    } else {
      return mlir::Type();
    }
  }
  llvm_unreachable("should never reach");
}

/**
 * Mimic a for loop by calling LoxT::isa(parser) in order, and print the first one that succeeds.
 */
template <class LoxT, class... RemainedT> void TypedPrint(mlir::Type type, mlir::DialectAsmPrinter &printer) {
  if (type.isa<LoxT>()) {
    return LoxT::print(type.cast<LoxT>(), printer);
  } else {
    if constexpr (sizeof...(RemainedT) > 0) {
      return TypedPrint<RemainedT...>(type, printer);
    }
  }
  llvm_unreachable("should never reach");
}
} // namespace

#endif // LOX_SRCS_MLIR_LIB_DIALECT_LOX_IR_PRINT_DIALECT_TYPE_H
