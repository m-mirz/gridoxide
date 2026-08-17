// The same thing through the C++ wrapper — RAII, std::vector, exceptions.
//
// Compare with powerflow.c beside it: no manual free, no status checks in the
// happy path, no malloc. Both link the same library.
//
//   usage: powerflow_cpp <network.json>

#include "gridoxide.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <network.json>\n";
        return 2;
    }

    try {
        // Only meaningful against a prebuilt library, where the header and the
        // binary can drift apart. Harmless otherwise.
        gridoxide::check_abi_compatibility();
        std::cout << "gridoxide " << gridoxide::version() << ", ABI "
                  << gridoxide_abi_version() << '\n';

        auto options = gridoxide::PowerFlow::default_options();
        options.s_base_va = 1e8; // pglib documents are per-unit on 100 MVA

        auto pf = gridoxide::PowerFlow::from_pgm_file(argv[1], options);
        pf.solve();

        const auto vm = pf.voltage_magnitude();
        const auto [lo, hi] = std::minmax_element(vm.begin(), vm.end());
        std::printf("%zu buses, %zu branches, converged in %zu iterations\n", pf.bus_count(),
                    pf.branch_count(), pf.iterations());
        std::printf("largest mismatch %.3e pu\n", pf.max_mismatch());
        std::printf("voltage range %.6f .. %.6f pu\n", *lo, *hi);

        // AC branch flows — the capability the Python binding does not expose.
        const auto [p, q] = pf.branch_flow();
        double loading = 0.0;
        for (std::size_t b = 0; b < p.size(); ++b) {
            loading = std::max(loading, std::hypot(p[b], q[b]));
        }
        std::printf("busiest branch carries %.4f pu\n", loading);

        // And the same network with the imbalance shared across generators
        // instead of dumped on one bus.
        pf.solve_distributing_slack();
        const auto shift = pf.slack_shift();
        double moved = 0.0;
        for (double s : shift) {
            moved += s;
        }
        std::printf("distributed slack moved %.4f pu off the reference bus\n", moved);

        return 0;
    } catch (const gridoxide::Error &e) {
        std::cerr << "gridoxide error (status " << static_cast<int>(e.status())
                  << "): " << e.what() << '\n';
        return 1;
    }
}
