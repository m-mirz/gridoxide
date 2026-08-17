/* gridoxide — a C++ convenience layer over gridoxide.h.
 *
 * Header-only, C++17, no dependencies beyond the standard library. Hand
 * written, unlike gridoxide.h next to it, which cbindgen generates.
 *
 * What it adds over the C API, and nothing more:
 *
 *   - RAII: the handle is released when the object goes out of scope, on every
 *     path including an exception.
 *   - std::vector results: the count-then-fill dance happens once, here.
 *   - Exceptions: a status code becomes a gridoxide::Error carrying the
 *     message the C API would have made you fetch separately.
 *
 * It deliberately does *not* wrap the network model. A caller who already has
 * buses and lines in memory passes gridoxide_bus / gridoxide_line arrays
 * straight through — inventing a C++ mirror of a struct that is already POD
 * would be a copy for nothing.
 *
 * THREADING: a PowerFlow is single-thread-affine, inheriting the C handle's
 * rule. It is movable but not copyable, so it cannot be duplicated onto
 * another thread by accident.
 */

#ifndef GRIDOXIDE_HPP
#define GRIDOXIDE_HPP

#include "gridoxide.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gridoxide {

/** A failed call, carrying the status code and the library's own message. */
class Error : public std::runtime_error {
public:
    Error(gridoxide_status status, const std::string &what)
        : std::runtime_error(what), status_(status) {}

    /** The code the C API returned. */
    gridoxide_status status() const noexcept { return status_; }

private:
    gridoxide_status status_;
};

namespace detail {

/** The library's last message on this thread, or a fallback naming the code. */
inline std::string last_error(gridoxide_status status) {
    const char *message = gridoxide_last_error_message();
    if (message != nullptr) {
        return std::string(message);
    }
    return "gridoxide call failed with status " + std::to_string(static_cast<int>(status));
}

/** Throws unless the call succeeded. */
inline void check(gridoxide_status status) {
    if (status != GRIDOXIDE_STATUS_OK) {
        throw Error(status, last_error(status));
    }
}

} // namespace detail

/** Which terminal of a branch a flow is measured at. */
enum class Terminal : int { From = 0, To = 1 };

/**
 * A power-flow model.
 *
 * Construct once per topology and solve as often as you like: the symbolic
 * factorization is computed on the first solve and reused by every one after.
 *
 * Note that a network read from a power-grid-model document has **one bus per
 * node plus one per source** — the conversion adds a virtual slack bus behind
 * each source, so `bus_count()` exceeds the node count of the original file.
 * Index results by what this object reports, not by your own numbering.
 */
class PowerFlow {
public:
    /** Options with the library's defaults filled in. */
    static gridoxide_options default_options() noexcept {
        gridoxide_options options{};
        gridoxide_options_default(&options);
        return options;
    }

    /** Reads a power-grid-model JSON file. */
    static PowerFlow from_pgm_file(const std::string &path,
                                   const gridoxide_options &options = default_options()) {
        gridoxide_powerflow *handle = nullptr;
        detail::check(gridoxide_powerflow_from_pgm_file(path.c_str(), &options, &handle));
        return PowerFlow(handle);
    }

    /** Reads a power-grid-model JSON document already in memory. */
    static PowerFlow from_pgm_string(const std::string &json,
                                     const gridoxide_options &options = default_options()) {
        gridoxide_powerflow *handle = nullptr;
        detail::check(gridoxide_powerflow_from_pgm_string(json.data(), json.size(), &options,
                                                          &handle));
        return PowerFlow(handle);
    }

    /**
     * Builds from arrays the caller already holds, in per-unit.
     *
     * Bus indices in the branch and shunt arrays refer to positions in `buses`.
     * ZIP (voltage-dependent) load terms cannot be expressed this way — use a
     * document if the network has them.
     */
    static PowerFlow from_arrays(const std::vector<gridoxide_bus> &buses,
                                 const std::vector<gridoxide_line> &lines = {},
                                 const std::vector<gridoxide_transformer> &transformers = {},
                                 const std::vector<gridoxide_shunt> &shunts = {},
                                 const gridoxide_options &options = default_options()) {
        gridoxide_powerflow *handle = nullptr;
        detail::check(gridoxide_powerflow_from_arrays(
            buses.data(), buses.size(), lines.data(), lines.size(), transformers.data(),
            transformers.size(), shunts.data(), shunts.size(), &options, &handle));
        return PowerFlow(handle);
    }

    ~PowerFlow() { gridoxide_powerflow_free(handle_); }

    PowerFlow(const PowerFlow &) = delete;
    PowerFlow &operator=(const PowerFlow &) = delete;

    PowerFlow(PowerFlow &&other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    PowerFlow &operator=(PowerFlow &&other) noexcept {
        if (this != &other) {
            gridoxide_powerflow_free(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /** An ordinary AC power flow. */
    void solve() { detail::check(gridoxide_powerflow_solve(handle_)); }

    /** With the PV→PQ outer loop, so generators respect their reactive limits. */
    void solve_enforcing_q_limits(std::size_t max_outer_iterations = 10) {
        detail::check(gridoxide_powerflow_solve_enforcing_q_limits(handle_, max_outer_iterations));
    }

    /**
     * With the system imbalance shared across generators rather than left on
     * one slack bus.
     *
     * An empty `factors` gives every `Slack` and `PV` bus an equal share.
     * Otherwise it is one weight per bus, normalized per island; zero means the
     * bus does not participate.
     */
    void solve_distributing_slack(const std::vector<double> &factors = {}) {
        const double *data = factors.empty() ? nullptr : factors.data();
        detail::check(
            gridoxide_powerflow_solve_distributing_slack(handle_, data, factors.size()));
    }

    std::size_t bus_count() const noexcept { return gridoxide_powerflow_bus_count(handle_); }

    /** Branch count — lines first, then transformers, which is the order every
     *  per-branch result uses. */
    std::size_t branch_count() const noexcept {
        return gridoxide_powerflow_branch_count(handle_);
    }

    std::vector<double> voltage_magnitude() const {
        std::vector<double> out(bus_count());
        detail::check(gridoxide_powerflow_voltage_magnitude(handle_, out.data(), out.size()));
        return out;
    }

    std::vector<double> voltage_angle() const {
        std::vector<double> out(bus_count());
        detail::check(gridoxide_powerflow_voltage_angle(handle_, out.data(), out.size()));
        return out;
    }

    /** Active and reactive power entering each branch at `terminal`, per-unit. */
    std::pair<std::vector<double>, std::vector<double>>
    branch_flow(Terminal terminal = Terminal::From) const {
        std::vector<double> p(branch_count());
        std::vector<double> q(branch_count());
        detail::check(gridoxide_powerflow_branch_flow(handle_, static_cast<int>(terminal),
                                                      p.data(), q.data(), p.size()));
        return {std::move(p), std::move(q)};
    }

    /**
     * How far each bus's active schedule moved in the last distributed-slack
     * solve.
     *
     * Empty if the last solve did not distribute slack — the C API reports that
     * as `GRIDOXIDE_STATUS_NO_ANSWER`, which is a modelled answer rather than a
     * failure, so it does not throw.
     */
    std::vector<double> slack_shift() const {
        std::vector<double> out(bus_count());
        gridoxide_status status =
            gridoxide_powerflow_slack_shift(handle_, out.data(), out.size());
        if (status == GRIDOXIDE_STATUS_NO_ANSWER) {
            return {};
        }
        detail::check(status);
        return out;
    }

    std::size_t iterations() const noexcept {
        return gridoxide_powerflow_iterations(handle_);
    }

    double max_mismatch() const noexcept {
        return gridoxide_powerflow_max_mismatch(handle_);
    }

    std::size_t island_count() const noexcept {
        return gridoxide_powerflow_island_count(handle_);
    }

    /**
     * One island's outcome.
     *
     * Worth reading even after a successful solve: an island with no source
     * reports `NO_REFERENCE_BUS` and does *not* make the solve fail, so this is
     * the only place a de-energized pocket of the network becomes visible.
     */
    gridoxide_island_status island_status(std::size_t island) const {
        gridoxide_island_status out{};
        detail::check(gridoxide_powerflow_island_status(handle_, island, &out));
        return out;
    }

    std::vector<std::size_t> island_buses(std::size_t island) const {
        std::vector<std::size_t> out(gridoxide_powerflow_island_bus_count(handle_, island));
        detail::check(gridoxide_powerflow_island_buses(handle_, island, out.data(), out.size()));
        return out;
    }

    /** The underlying C handle, for anything this wrapper does not cover. */
    gridoxide_powerflow *handle() const noexcept { return handle_; }

private:
    explicit PowerFlow(gridoxide_powerflow *handle) noexcept : handle_(handle) {}
    gridoxide_powerflow *handle_;
};

/** The linked library's version. */
inline std::string version() { return std::string(gridoxide_version()); }

/**
 * Throws unless the linked library speaks the ABI this header was compiled
 * against.
 *
 * Worth calling once at startup when linking a prebuilt library, where the
 * header and the binary can be updated independently. Pointless when building
 * from source, where they cannot.
 */
inline void check_abi_compatibility() {
    const uint32_t linked = gridoxide_abi_version();
    if (linked != GRIDOXIDE_ABI_VERSION) {
        throw Error(GRIDOXIDE_STATUS_INTERNAL,
                    "gridoxide ABI mismatch: this header describes version " +
                        std::to_string(GRIDOXIDE_ABI_VERSION) +
                        ", but the linked library speaks version " + std::to_string(linked));
    }
}

} // namespace gridoxide

#endif /* GRIDOXIDE_HPP */
