//
// LICENSE: MIT
//

#ifndef LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_GC_H
#define LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_GC_H
#include <set>
#include <unordered_map>

#include "lox/object/object.h"

namespace lox {
struct GC {
  // singleton
  using MarkerFn = void (*)(void*);
  struct Marker {
    MarkerFn marker_fn;
    void* marker_fn_arg;
    bool operator<(const Marker& rhs) const;
  };
  GC(const GC&) = delete;
  GC(GC&&) = delete;
  GC& operator=(const GC&) = delete;
  GC& operator=(GC&&) = delete;

  static GC& Instance();
  void collectGarbage();
  void RegisterMarker(MarkerFn fn, void* arg);
  void UnRegisterMarker(MarkerFn fn, void* arg);
  void RecursiveMark(Object* object);

  struct RegisterMarkerGuard {
    RegisterMarkerGuard(MarkerFn fn, void* arg);
    ~RegisterMarkerGuard();
    Marker marker;
  };

  int gc_threashold = 1024;

 private:
  GC() = default;
  void markRoots();
  void Sweep();
  std::__1::set<Marker> markers;
  std::unordered_map<void*, bool> marked;
};
}  // namespace lox

#endif  // LOX_SRCS_LOX_BACKEND_VIRTUAL_MACHINE_COMMON_GC_H
