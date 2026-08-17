/* A minimal C consumer of gridoxide.
 *
 * Doubles as a smoke test: CI compiles and runs it, so the header parsing, the
 * struct layouts and the symbol linkage are all checked by a real C compiler
 * rather than assumed.
 *
 *   usage: powerflow_c <network.json>
 */

#include "gridoxide.h"

#include <stdio.h>
#include <stdlib.h>

static int fail(const char *what, gridoxide_status status) {
    const char *message = gridoxide_last_error_message();
    fprintf(stderr, "%s failed (status %d): %s\n", what, (int)status,
            message ? message : "no message");
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <network.json>\n", argv[0]);
        return 2;
    }

    printf("gridoxide %s, ABI %u (header says %u)\n", gridoxide_version(),
           gridoxide_abi_version(), (unsigned)GRIDOXIDE_ABI_VERSION);

    /* Always start from the defaults: a later-added option then gets its
     * intended value rather than a zero. */
    gridoxide_options options;
    gridoxide_options_default(&options);
    options.s_base_va = 1e8; /* pglib documents are per-unit on 100 MVA */

    gridoxide_powerflow *pf = NULL;
    gridoxide_status status = gridoxide_powerflow_from_pgm_file(argv[1], &options, &pf);
    if (status != GRIDOXIDE_STATUS_OK) {
        return fail("loading the network", status);
    }

    status = gridoxide_powerflow_solve(pf);
    if (status != GRIDOXIDE_STATUS_OK) {
        gridoxide_powerflow_free(pf);
        return fail("solving", status);
    }

    size_t n = gridoxide_powerflow_bus_count(pf);
    double *vm = malloc(n * sizeof(double));
    if (vm == NULL) {
        gridoxide_powerflow_free(pf);
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    status = gridoxide_powerflow_voltage_magnitude(pf, vm, n);
    if (status != GRIDOXIDE_STATUS_OK) {
        free(vm);
        gridoxide_powerflow_free(pf);
        return fail("reading voltages", status);
    }

    double lo = vm[0], hi = vm[0];
    for (size_t i = 1; i < n; i++) {
        if (vm[i] < lo) lo = vm[i];
        if (vm[i] > hi) hi = vm[i];
    }
    printf("%zu buses, %zu branches, converged in %zu iterations\n", n,
           gridoxide_powerflow_branch_count(pf), gridoxide_powerflow_iterations(pf));
    printf("largest mismatch %.3e pu\n", gridoxide_powerflow_max_mismatch(pf));
    printf("voltage range %.6f .. %.6f pu\n", lo, hi);

    /* Islands are worth reading even on success: a component with no source
     * reports NO_REFERENCE_BUS without failing the solve. */
    size_t islands = gridoxide_powerflow_island_count(pf);
    for (size_t i = 0; i < islands; i++) {
        gridoxide_island_status island;
        if (gridoxide_powerflow_island_status(pf, i, &island) == GRIDOXIDE_STATUS_OK) {
            printf("island %zu: %zu buses, status %d\n", i,
                   gridoxide_powerflow_island_bus_count(pf, i), (int)island);
        }
    }

    free(vm);
    gridoxide_powerflow_free(pf);
    return 0;
}
