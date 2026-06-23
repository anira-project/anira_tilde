// Loader shim for anira~.
//
// anira~ depends on libtorch. macOS dyld identifies a dylib by its install
// name, so if another libtorch-based external (e.g. nn~) is loaded into the
// Max process first, our libtorch references bind to that foreign copy — a
// different build whose symbols don't match. The real external then either
// fails to load or aborts during libtorch's static initialisers, before any
// of our code can run, so we can't explain what happened.
//
// To give the user a clear message instead of a silent failure, the
// libtorch-linked implementation lives in a separate dylib
// (anira_tilde_impl.dylib). THIS bundle links no libtorch, so its ext_main
// always runs first. It checks whether a libtorch is already resident and
// only then dlopens the implementation; otherwise it posts a warning and
// installs an inert stub object so the patch loads without crashing.

#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ext.h"
#include "ext_obex.h"

static const char* k_conflict_msg =
    "anira~: cannot load \xE2\x80\x94 another external has already loaded "
    "the libtorch dependency, which would collide. Workaround: quit Max "
    "and do not load the conflicting external.";

// ---------------------------------------------------------------------------
// Detection: is any libtorch-family image already loaded? At the point this
// shim's ext_main runs we have not yet loaded our own implementation, so any
// libtorch present must belong to another external.
// ---------------------------------------------------------------------------

static const char* foreign_libtorch_loaded() {
    static const char* leaves[] = { "libtorch.dylib", "libtorch_cpu.dylib", "libc10.dylib" };
    const uint32_t n = _dyld_image_count();
    for (uint32_t i = 0; i < n; ++i) {
        const char* path = _dyld_get_image_name(i);
        if (!path) continue;
        const char* base = strrchr(path, '/');
        base = base ? base + 1 : path;
        for (size_t k = 0; k < sizeof(leaves) / sizeof(leaves[0]); ++k)
            if (strcmp(base, leaves[k]) == 0) return path;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Inert stub object, registered when we refuse to load. It exists only so the
// object box resolves and the explanation is repeated on instantiation.
// ---------------------------------------------------------------------------

static t_class* s_stub_class = nullptr;

typedef struct _anira_stub { t_object ob; } t_anira_stub;

static void* stub_new(t_symbol* s, long argc, t_atom* argv) {
    (void)s; (void)argc; (void)argv;
    t_anira_stub* x = (t_anira_stub*)object_alloc(s_stub_class);
    error("%s", k_conflict_msg);
    return x;
}

static void stub_free(t_anira_stub* x) { (void)x; }

static void install_stub() {
    s_stub_class = class_new("anira~", (method)stub_new, (method)stub_free,
                             (long)sizeof(t_anira_stub), 0L, A_GIMME, 0);
    class_register(CLASS_BOX, s_stub_class);
}

// ---------------------------------------------------------------------------
// Locate anira_tilde_impl.dylib, which sits next to this bundle in externals/.
// dli_fname is .../externals/anira~.mxo/Contents/MacOS/anira~; strip four
// trailing path components to reach .../externals.
// ---------------------------------------------------------------------------

static bool impl_path(char* out, size_t cap) {
    Dl_info info;
    if (!dladdr((void*)&impl_path, &info) || !info.dli_fname) return false;
    char buf[PATH_MAX];
    strncpy(buf, info.dli_fname, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    for (int i = 0; i < 4; ++i) {
        char* slash = strrchr(buf, '/');
        if (!slash) return false;
        *slash = '\0';
    }
    const int len = snprintf(out, cap, "%s/anira_tilde_impl.dylib", buf);
    return len > 0 && (size_t)len < cap;
}

// ---------------------------------------------------------------------------
// Max entry point. Same build path as a normal Min external, but links no
// libtorch, so it always gets to run.
// ---------------------------------------------------------------------------

extern "C" void ext_main(void* r) {
    if (const char* foreign = foreign_libtorch_loaded()) {
        error("%s", k_conflict_msg);
        error("anira~: (conflicting libtorch already loaded from %s)", foreign);
        install_stub();
        return;
    }

    char path[PATH_MAX];
    if (!impl_path(path, sizeof(path))) {
        error("anira~: internal error \xE2\x80\x94 could not locate implementation dylib.");
        install_stub();
        return;
    }

    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        error("anira~: failed to load implementation: %s", dlerror());
        install_stub();
        return;
    }

    typedef void (*impl_main_fn)(void*);
    auto impl_main = (impl_main_fn)dlsym(handle, "anira_tilde_impl_main");
    if (!impl_main) {
        error("anira~: failed to find implementation entry: %s", dlerror());
        install_stub();
        return;
    }

    impl_main(r);  // registers the real anira~ class
}
