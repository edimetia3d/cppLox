//
// License: MIT
//

#ifndef CPPLOX_SRCS_LOX_BINARY_DISPATCHER_H_
#define CPPLOX_SRCS_LOX_BINARY_DISPATCHER_H_

#define DISPATCHTER(OPERATION_NAME) D##OPERATION_NAME

#define DEF_DISPATCHER(OPERATION_NAME)                                                       \
  template <class LhsT, class... LhsTs>                                                      \
  struct DISPATCHTER(OPERATION_NAME) {                                                       \
    template <class RhsT, class... RhsTs>                                                    \
    struct In {                                                                              \
      static void Run(LoxObjectState *lhs, LoxObjectState *rhs, LoxObject *output) {         \
        DISPATCHTER(OPERATION_NAME)<LhsT>::template In<RhsT>::Run(lhs, rhs, output);         \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
        DISPATCHTER(OPERATION_NAME)<LhsT>::template In<RhsTs...>::Run(lhs, rhs, output);     \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
        DISPATCHTER(OPERATION_NAME)<LhsTs...>::template In<RhsT>::Run(lhs, rhs, output);     \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
        DISPATCHTER(OPERATION_NAME)<LhsTs...>::template In<RhsTs...>::Run(lhs, rhs, output); \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
      }                                                                                      \
    };                                                                                       \
                                                                                             \
    template <class RhsT>                                                                    \
    struct In<RhsT> {                                                                        \
      static void Run(LoxObjectState *lhs, LoxObjectState *rhs, LoxObject *output) {         \
        DISPATCHTER(OPERATION_NAME)<LhsT>::template In<RhsT>::Run(lhs, rhs, output);         \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
        DISPATCHTER(OPERATION_NAME)<LhsTs...>::template In<RhsT>::Run(lhs, rhs, output);     \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
      }                                                                                      \
    };                                                                                       \
  };                                                                                         \
                                                                                             \
  template <class LhsT>                                                                      \
  struct DISPATCHTER(OPERATION_NAME)<LhsT> {                                                 \
    template <class RhsT, class... RhsTs>                                                    \
    struct In {                                                                              \
      static void Run(LoxObjectState *lhs, LoxObjectState *rhs, LoxObject *output) {         \
        DISPATCHTER(OPERATION_NAME)<LhsT>::template In<RhsT>::Run(lhs, rhs, output);         \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
        DISPATCHTER(OPERATION_NAME)<LhsT>::template In<RhsTs...>::Run(lhs, rhs, output);     \
        if (output->State()) {                                                               \
          return;                                                                            \
        }                                                                                    \
      }                                                                                      \
    };                                                                                       \
                                                                                             \
    template <class RhsT>                                                                    \
    struct In<RhsT> {                                                                        \
      static void Run(LoxObjectState *lhs, LoxObjectState *rhs, LoxObject *output) {         \
        if constexpr (requires(LhsT * lhs, RhsT * rhs) { OPERATION_NAME(lhs, rhs); }) {      \
          if (auto dlhs = dynamic_cast<LhsT *>(lhs)) {                                       \
            if (auto drhs = dynamic_cast<RhsT *>(rhs)) {                                     \
              *output = LoxObject(OPERATION_NAME(dlhs, drhs));                               \
            }                                                                                \
          }                                                                                  \
        }                                                                                    \
      }                                                                                      \
    };                                                                                       \
  };

#endif  // CPPLOX_SRCS_LOX_BINARY_DISPATCHER_H_