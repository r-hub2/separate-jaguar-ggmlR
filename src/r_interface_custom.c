// Custom operations: named-registry bindings for GGML_OP_CUSTOM.
//
// Downstream packages (llamaR, sd2R) that statically link libggml.a can call
// ggml_custom_4d()/ggml_custom_inplace() directly from their own C code. This
// file exists for the other direction: a graph assembled from R that needs one
// node backed by a C kernel compiled elsewhere.
//
// The kernel is addressed by NAME, never by a raw function pointer handed
// across the package boundary. A bad name is a clean R error; a raw pointer of
// the wrong type would be a segfault. Downstream registers its kernels once, at
// package load, via the R_RegisterCCallable symbol declared in
// inst/include/ggmlR.h:
//
//     void (*reg)(const char *, ggml_custom_op_t) =
//         (void (*)(const char *, ggml_custom_op_t))
//             R_GetCCallable("ggmlR", "ggmlR_register_custom_op");
//     reg("my_kernel", my_kernel_fn);
//
// and R then builds the node with ggml_custom(ctx, "my_kernel", ...).

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "r_ptr_check.h"

// ============================================================================
// Kernel registry
// ============================================================================

// Entries are never removed: a registered kernel may already be referenced by a
// live graph node, so unregistering could leave a dangling fun pointer to be
// called at compute time. Registration is a load-time, once-per-kernel action.
#define R_CUSTOM_MAX_OPS 64

struct r_custom_entry {
    char             * name;
    ggml_custom_op_t   fun;
};

static struct r_custom_entry r_custom_ops[R_CUSTOM_MAX_OPS];
static int                   r_custom_n_ops = 0;

static ggml_custom_op_t r_custom_lookup(const char * name) {
    for (int i = 0; i < r_custom_n_ops; i++) {
        if (strcmp(r_custom_ops[i].name, name) == 0) {
            return r_custom_ops[i].fun;
        }
    }
    return NULL;
}

// Exported to downstream packages via R_RegisterCCallable (see R_init_ggmlR).
// Re-registering an existing name replaces its kernel, which keeps package
// reloads during development idempotent.
void ggmlR_register_custom_op(const char * name, ggml_custom_op_t fun) {
    if (name == NULL || fun == NULL) {
        return;
    }

    for (int i = 0; i < r_custom_n_ops; i++) {
        if (strcmp(r_custom_ops[i].name, name) == 0) {
            r_custom_ops[i].fun = fun;
            return;
        }
    }

    if (r_custom_n_ops >= R_CUSTOM_MAX_OPS) {
        return;  // silently ignored: called from downstream C, not from R
    }

    size_t len = strlen(name);
    char * copy = (char *) malloc(len + 1);
    if (copy == NULL) {
        return;
    }
    memcpy(copy, name, len + 1);

    r_custom_ops[r_custom_n_ops].name = copy;
    r_custom_ops[r_custom_n_ops].fun  = fun;
    r_custom_n_ops++;
}

// Resolve a name coming from R, erroring with the list of what is available.
static ggml_custom_op_t r_custom_require(SEXP name_sexp) {
    if (TYPEOF(name_sexp) != STRSXP || LENGTH(name_sexp) != 1) {
        error("'fun' must be a single character string naming a registered custom op");
    }

    const char * name = CHAR(STRING_ELT(name_sexp, 0));
    ggml_custom_op_t fun = r_custom_lookup(name);

    if (fun == NULL) {
        if (r_custom_n_ops == 0) {
            error("No custom op named '%s': no custom ops are registered. "
                  "A C kernel must be registered by a package via "
                  "R_GetCCallable(\"ggmlR\", \"ggmlR_register_custom_op\").", name);
        }
        // Build a comma-separated list of registered names for the message.
        size_t total = 0;
        for (int i = 0; i < r_custom_n_ops; i++) {
            total += strlen(r_custom_ops[i].name) + 2;
        }
        char * known = (char *) R_alloc(total + 1, 1);
        known[0] = '\0';
        for (int i = 0; i < r_custom_n_ops; i++) {
            if (i > 0) {
                strcat(known, ", ");
            }
            strcat(known, r_custom_ops[i].name);
        }
        error("No custom op named '%s'. Registered: %s", name, known);
    }

    return fun;
}

// List registered kernel names (diagnostics from R).
SEXP R_ggml_custom_ops(void) {
    SEXP out = PROTECT(allocVector(STRSXP, r_custom_n_ops));
    for (int i = 0; i < r_custom_n_ops; i++) {
        SET_STRING_ELT(out, i, mkChar(r_custom_ops[i].name));
    }
    UNPROTECT(1);
    return out;
}

// ============================================================================
// Argument marshalling
// ============================================================================

// Collect the variadic tensor arguments from an R list of external pointers.
// `max_args` differs between the two builders: ggml_custom_inplace puts `a` in
// src[0] and so has one slot fewer than ggml_custom_4d.
static int r_custom_collect_args(SEXP args_list,
                                 struct ggml_tensor ** args,
                                 int max_args) {
    if (args_list == R_NilValue) {
        return 0;
    }
    if (TYPEOF(args_list) != VECSXP) {
        error("'args' must be a list of tensors or NULL");
    }

    int n_args = LENGTH(args_list);
    if (n_args > max_args) {
        error("Too many custom op arguments: %d given, at most %d supported",
              n_args, max_args);
    }

    for (int i = 0; i < n_args; i++) {
        args[i] = (struct ggml_tensor *)
            r_ptr_required(VECTOR_ELT(args_list, i), "custom op argument tensor");
    }

    return n_args;
}

// ============================================================================
// Builders
// ============================================================================

// ggml_custom_4d: a new tensor of the given type/shape, computed by `fun`.
SEXP R_ggml_custom_4d(SEXP ctx_ptr, SEXP type, SEXP ne0, SEXP ne1, SEXP ne2, SEXP ne3,
                      SEXP args_list, SEXP fun_name, SEXP n_tasks_sexp) {
    struct ggml_context * ctx =
        (struct ggml_context *) r_ptr_required(ctx_ptr, "context");

    ggml_custom_op_t fun = r_custom_require(fun_name);

    struct ggml_tensor * args[GGML_MAX_SRC];
    int n_args = r_custom_collect_args(args_list, args, GGML_MAX_SRC - 1);

    enum ggml_type dtype = (enum ggml_type) asInteger(type);
    int64_t n0 = (int64_t) asReal(ne0);
    int64_t n1 = (int64_t) asReal(ne1);
    int64_t n2 = (int64_t) asReal(ne2);
    int64_t n3 = (int64_t) asReal(ne3);

    int n_tasks = asInteger(n_tasks_sexp);
    if (n_tasks != GGML_N_TASKS_MAX && n_tasks < 1) {
        error("'n_tasks' must be >= 1 or GGML_N_TASKS_MAX (%d)", GGML_N_TASKS_MAX);
    }

    struct ggml_tensor * result =
        ggml_custom_4d(ctx, dtype, n0, n1, n2, n3, args, n_args, fun, n_tasks, NULL);

    if (result == NULL) {
        error("Failed to create custom operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ggml_custom_inplace: writes into a view of `a`; src[0] is `a` itself.
SEXP R_ggml_custom_inplace(SEXP ctx_ptr, SEXP a_ptr, SEXP args_list,
                           SEXP fun_name, SEXP n_tasks_sexp) {
    struct ggml_context * ctx =
        (struct ggml_context *) r_ptr_required(ctx_ptr, "context");
    struct ggml_tensor * a =
        (struct ggml_tensor *) r_ptr_required(a_ptr, "tensor");

    ggml_custom_op_t fun = r_custom_require(fun_name);

    struct ggml_tensor * args[GGML_MAX_SRC];
    int n_args = r_custom_collect_args(args_list, args, GGML_MAX_SRC - 2);

    int n_tasks = asInteger(n_tasks_sexp);
    if (n_tasks != GGML_N_TASKS_MAX && n_tasks < 1) {
        error("'n_tasks' must be >= 1 or GGML_N_TASKS_MAX (%d)", GGML_N_TASKS_MAX);
    }

    struct ggml_tensor * result =
        ggml_custom_inplace(ctx, a, args, n_args, fun, n_tasks, NULL);

    if (result == NULL) {
        error("Failed to create custom operation");
    }

    return R_MakeExternalPtr(result, R_NilValue, R_NilValue);
}

// ============================================================================
// Backend guard
// ============================================================================

// GGML_OP_CUSTOM is implemented by the CPU backend only -- the kernel is a host
// function pointer, so a Vulkan device cannot run it. The node->backend binding
// is decided by ggml_backend_sched at compute time, not when the node is built,
// so this check cannot live in the builders above; it is called from the
// compute entry points instead.
//
// Returns the name of the first offending node, or NULL if the graph is safe.
const char * ggmlR_custom_check_backend(struct ggml_cgraph * graph,
                                        ggml_backend_t backend) {
    if (graph == NULL || backend == NULL) {
        return NULL;
    }
    if (ggml_backend_is_cpu(backend)) {
        return NULL;
    }

    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor * node = ggml_graph_node(graph, i);
        if (node != NULL && node->op == GGML_OP_CUSTOM) {
            return node->name[0] != '\0' ? node->name : "<unnamed>";
        }
    }

    return NULL;
}

// Scheduler path. Per-node backend assignment is internal to
// ggml_backend_sched, so the check here is the one that can be made soundly:
// a custom node is fine as long as the scheduler owns a CPU backend to fall
// back to. With a GPU-only scheduler the node has nowhere to run.
//
// Returns the name of the first custom node when no CPU backend is present,
// NULL otherwise.
const char * ggmlR_custom_check_sched(struct ggml_cgraph * graph,
                                      ggml_backend_sched_t sched) {
    if (graph == NULL || sched == NULL) {
        return NULL;
    }

    const char * custom_node = NULL;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        struct ggml_tensor * node = ggml_graph_node(graph, i);
        if (node != NULL && node->op == GGML_OP_CUSTOM) {
            custom_node = node->name[0] != '\0' ? node->name : "<unnamed>";
            break;
        }
    }
    if (custom_node == NULL) {
        return NULL;
    }

    int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; i++) {
        if (ggml_backend_is_cpu(ggml_backend_sched_get_backend(sched, i))) {
            return NULL;
        }
    }

    return custom_node;
}
