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
#include <sys/syslimits.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

#include "ext.h"
#include "ext_mess.h"
#include "ext_obex.h"
#include "ext_post.h"

namespace {

const char* k_conflict_msg =
    "anira~: cannot load \xE2\x80\x94 another external has already loaded "
    "the libtorch dependency, which would collide. Workaround: quit Max "
    "and do not load the conflicting external.";

// ---------------------------------------------------------------------------
// Detection: is any libtorch-family image already loaded? At the point this
// shim's ext_main runs we have not yet loaded our own implementation, so any
// libtorch present must belong to another external.
// ---------------------------------------------------------------------------

const char* foreign_libtorch_loaded() {
    static constexpr std::array<const char*, 3> k_leaves = {"libtorch.dylib",
                                                            "libtorch_cpu.dylib",
                                                            "libc10.dylib"};
    const uint32_t n = _dyld_image_count();
    for (uint32_t i = 0; i < n; ++i) {
        const char* path = _dyld_get_image_name(i);
        if (!path) { continue; }
        const char* base = strrchr(path, '/');
        base = base ? base + 1 : path;
        for (const char* leaf : k_leaves) {
            if (strcmp(base, leaf) == 0) { return path; }
        }
    }
    return nullptr;
}

// Is our own implementation dylib already resident? anira~ and mc.anira~ are two
// bundles sharing one anira_tilde_impl.dylib, so once either object loads it the
// other would see libtorch resident. That libtorch is ours, not a foreign
// collision: if the impl is already loaded, the sibling object may proceed (its
// dlopen just refcounts the same image).
bool own_impl_loaded() {
    const uint32_t n = _dyld_image_count();
    for (uint32_t i = 0; i < n; ++i) {
        const char* path = _dyld_get_image_name(i);
        if (!path) { continue; }
        const char* base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (strcmp(base, "anira_tilde_impl.dylib") == 0 ||
            strcmp(base, "mc_anira_tilde_impl.dylib") == 0) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Inert stub object, registered when we refuse to load. It exists only so the
// object box resolves and the explanation is repeated on instantiation.
// ---------------------------------------------------------------------------

// One stub class per object name we would otherwise have registered. stub_new
// looks up its own class from the box's class name so a single new/free pair
// serves both.
struct AniraStub {
    t_object m_ob;
};

void* stub_new(t_symbol* s, long argc, t_atom* argv) {
    (void)argc;
    (void)argv;
    auto* c = (t_class*)class_findbyname(CLASS_BOX, s);
    auto* x = (AniraStub*)object_alloc(c);
    error("%s", k_conflict_msg);
    return x;
}

void stub_free(AniraStub* x) {
    (void)x;
}

void install_stub_named(const char* name) {
    t_class* c = class_new(name,
                           (method)stub_new,
                           (method)stub_free,
                           (long)sizeof(AniraStub),
                           nullptr,
                           A_GIMME,
                           0);
    class_register(CLASS_BOX, c);
}

// ---------------------------------------------------------------------------
// This one shim binary is shipped as two bundles, anira~.mxo and mc.anira~.mxo
// (the latter a copy made at build time). Each must register only its own object
// class, so we recover which one we are from our bundle directory name at load
// time. dli_fname is .../externals/<NAME>.mxo/Contents/MacOS/<exec>; strip three
// trailing components to reach <NAME>.mxo, then drop the ".mxo" extension.
// ---------------------------------------------------------------------------

void bundle_object_name(char* out, size_t cap) {
    // Default identity if anything below fails.
    snprintf(out, cap, "anira~");

    Dl_info info;
    if (!dladdr((void*)&bundle_object_name, &info) || !info.dli_fname) { return; }
    std::array<char, PATH_MAX> buf{};
    strncpy(buf.data(), info.dli_fname, buf.size() - 1);
    for (int i = 0; i < 3; ++i) {
        char* slash = strrchr(buf.data(), '/');
        if (!slash) { return; }
        *slash = '\0';
    }
    const char* base = strrchr(buf.data(), '/');
    base = base ? base + 1 : buf.data();  // "<NAME>.mxo"
    if (base[0] == '\0') { return; }
    snprintf(out, cap, "%s", base);
    char* dot = strrchr(out, '.');  // drop ".mxo"
    if (dot) { *dot = '\0'; }
}

// ---------------------------------------------------------------------------
// Locate anira_tilde_impl.dylib, which sits next to this bundle in externals/.
// dli_fname is .../externals/<NAME>.mxo/Contents/MacOS/<exec>; strip four
// trailing path components to reach .../externals.
// ---------------------------------------------------------------------------

bool impl_path(char* out, size_t cap, const char* dylib_leaf) {
    Dl_info info;
    if (!dladdr((void*)&impl_path, &info) || !info.dli_fname) { return false; }
    std::array<char, PATH_MAX> buf{};
    strncpy(buf.data(), info.dli_fname, buf.size() - 1);
    for (int i = 0; i < 4; ++i) {
        char* slash = strrchr(buf.data(), '/');
        if (!slash) { return false; }
        *slash = '\0';
    }
    const int len = snprintf(out, cap, "%s/%s", buf.data(), dylib_leaf);
    return len > 0 && std::cmp_less(len, cap);
}

}  // namespace

// ---------------------------------------------------------------------------
// Max entry point. Same build path as a normal Min external, but links no
// libtorch, so it always gets to run.
// ---------------------------------------------------------------------------

extern "C" void ext_main(void* r) {
    // Which object are we? anira~.mxo and mc.anira~.mxo share this binary.
    std::array<char, 256> name{};
    bundle_object_name(name.data(), name.size());
    const bool is_mc = strcmp(name.data(), "mc.anira~") == 0;

    // A resident libtorch is only a genuine conflict if it isn't the one our own
    // impl loaded (i.e. our impl isn't already present). Otherwise the sibling
    // object (anira~/mc.anira~) already brought in our libtorch and we proceed.
    if (const char* foreign = foreign_libtorch_loaded()) {
        if (!own_impl_loaded()) {
            error("%s", k_conflict_msg);
            error("%s: (conflicting libtorch already loaded from %s)", name.data(), foreign);
            install_stub_named(name.data());
            return;
        }
    }

    // Each object has its own impl dylib (separate translation units, so each
    // has its own min-api class registry).
    const char* dylib = is_mc ? "mc_anira_tilde_impl.dylib" : "anira_tilde_impl.dylib";
    std::array<char, PATH_MAX> path{};
    if (!impl_path(path.data(), path.size(), dylib)) {
        error("%s: internal error \xE2\x80\x94 could not locate implementation dylib.",
              name.data());
        install_stub_named(name.data());
        return;
    }

    void* handle = dlopen(path.data(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        error("%s: failed to load implementation: %s", name.data(), dlerror());
        install_stub_named(name.data());
        return;
    }

    // Each bundle registers exactly one class via its dedicated entry point.
    const char* entry = is_mc ? "mc_anira_tilde_impl_main" : "anira_tilde_impl_main";
    using impl_main_fn = void (*)(void*);
    auto impl_main = (impl_main_fn)dlsym(handle, entry);
    if (!impl_main) {
        error("%s: failed to find implementation entry: %s", name.data(), dlerror());
        install_stub_named(name.data());
        return;
    }

    impl_main(r);  // registers the real object class
}
