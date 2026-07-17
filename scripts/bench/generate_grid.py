#!/usr/bin/env python3
"""Ports power-grid-model's tests/benchmark_cpp/fictional_grid_generator.hpp
(FictionalGridGenerator::generate_mv_grid / generate_lv_grid) to Python,
using the same option values as PGM's own release-mode C++ benchmark
(tests/benchmark_cpp/benchmark.cpp, the #else branch), radial / symmetric /
no tap changer / no measurements / no fault case.

Not a bit-for-bit RNG replica of the C++ mt19937_64 sequence (different RNG
engine) -- produces a structurally equivalent radial MV/LV distribution grid
of comparable scale, as a valid PGM JSON `input` document. Since gridoxide
and power-grid-model are both benchmarked against this exact same generated
JSON file, the comparison is apples-to-apples regardless of how the topology
was generated.

Usage:
    python3 generate_grid.py <output.json> [--target-nodes N] [--seed N]

`--target-nodes` is a *rough* specification, not a guarantee: LV-grid
attachment is a stochastic Bernoulli process, so the realized node count
varies by seed. The three scales used in this project's README/benchmarks
were generated with:

    python3 generate_grid.py grid_small.json  --target-nodes 200   # -> 192 nodes
    python3 generate_grid.py grid_medium.json --target-nodes 1500  # -> 1,003 nodes
    python3 generate_grid.py grid_large.json  --target-nodes 2200  # -> 2,605 nodes

all with the default seed (42).
"""
import argparse
import json
import random
import sys


def generate(target_nodes: int, seed: int, out_path: str) -> None:
    rng = random.Random(seed)

    # Option, matching benchmark.cpp's #else (release) branch, except
    # n_node_total_specified is caller-supplied (see module docstring: the
    # LV-grid attachment is stochastic, so hitting a specific node count
    # exactly requires overshooting the nominal target somewhat).
    n_node_total_specified = target_nodes
    n_mv_feeder = 20
    n_node_per_mv_feeder = 10
    n_lv_feeder = 10
    n_connection_per_lv_feeder = 40
    has_mv_ring = False
    has_lv_ring = False

    # --- derive n_lv_grid / ratio_lv_grid / n_parallel_hv_mv_transformer, per generate_grid() ---
    total_mv_connection = n_mv_feeder * n_node_per_mv_feeder + 2
    node_per_lv_grid = n_lv_feeder * n_connection_per_lv_feeder * 2 + 1
    if total_mv_connection > n_node_total_specified:
        n_lv_grid = 0
        n_mv_feeder = (n_node_total_specified - 2) // n_node_per_mv_feeder
        total_mv_connection = n_mv_feeder * n_node_per_mv_feeder
    else:
        n_lv_grid = (n_node_total_specified - total_mv_connection) // node_per_lv_grid
    if n_lv_grid > total_mv_connection:
        n_mv_feeder = n_lv_grid // n_node_per_mv_feeder + 1
    total_mv_connection = n_mv_feeder * n_node_per_mv_feeder
    ratio_lv_grid = n_lv_grid / total_mv_connection if total_mv_connection > 0 else 1.0
    n_parallel_hv_mv_transformer = int(n_mv_feeder * 10.0 * 1.1 / 60.0) + 1

    print(f"n_mv_feeder={n_mv_feeder} n_lv_grid(expected)={n_lv_grid} "
          f"ratio_lv_grid={ratio_lv_grid} n_parallel_hv_mv_transformer={n_parallel_hv_mv_transformer}",
          file=sys.stderr)

    node, line, source, sym_load, asym_load, transformer, shunt = [], [], [], [], [], [], []
    _id = [0]

    def nid():
        _id[0] += 1
        return _id[0]

    def scale_cable(ln, ratio):
        for k in ("r1", "x1", "c1", "r0", "x0", "c0"):
            ln[k] *= ratio

    mv_ring = []
    lv_ring = []

    def generate_lv_grid(mv_node, mv_base_load):
        id_lv_busbar = nid()
        node.append({"id": id_lv_busbar, "u_rated": 400.0})
        transformer.append({
            "id": nid(), "from_node": mv_node, "to_node": id_lv_busbar,
            "from_status": 1, "to_status": 1,
            "u1": 10.5e3, "u2": 420.0, "sn": max(1500e3, mv_base_load * 1.2),
            "uk": 0.06, "pk": 8.8e3, "i0": 0.01, "p0": 1e3,
            "winding_from": 2, "winding_to": 1, "clock": 11,
            "tap_side": 0, "tap_pos": 3, "tap_min": -10, "tap_max": 10, "tap_nom": 3, "tap_size": 250.0,
        })

        lv_main_line_t = dict(r1=0.206, x1=0.079, c1=0.72e-6, tan1=0.0004, r0=0.94, x0=0.387, c0=0.36e-6, tan0=0.0)
        lv_conn_line_t = dict(r1=1.15, x1=0.096, c1=0.43e-6, tan1=0.0004, r0=4.6, x0=0.408, c0=0.258e-6, tan0=0.0)

        base_load = mv_base_load / (n_lv_feeder * n_connection_per_lv_feeder) / 1.2

        for _ in range(n_lv_feeder):
            prev_main = id_lv_busbar
            for j in range(n_connection_per_lv_feeder):
                cur_main = nid()
                node.append({"id": cur_main, "u_rated": 400.0})
                conn_node = nid()
                node.append({"id": conn_node, "u_rated": 400.0})

                main_line = dict(lv_main_line_t)
                main_line.update(id=nid(), from_node=prev_main, to_node=cur_main,
                                  from_status=1, to_status=1)
                scale_cable(main_line, rng.uniform(0.8 * 0.2 / n_connection_per_lv_feeder,
                                                    1.2 * 0.2 / n_connection_per_lv_feeder))
                line.append(main_line)

                conn_line = dict(lv_conn_line_t)
                conn_line.update(id=nid(), from_node=cur_main, to_node=conn_node,
                                  from_status=1, to_status=1)
                scale_cable(conn_line, rng.uniform(5e-3, 20e-3))
                line.append(conn_line)

                apparent_power = rng.uniform(0.8 * base_load, 1.2 * base_load)
                phase = rng.randint(0, 2)
                p3 = [0.0, 0.0, 0.0]
                q3 = [0.0, 0.0, 0.0]
                p3[phase] = apparent_power * 0.8
                q3[phase] = apparent_power * 0.6
                asym_load.append({
                    "id": nid(), "node": conn_node, "status": 1, "type": rng.randint(0, 2),
                    "p_specified": p3, "q_specified": q3,
                })

                if j == n_connection_per_lv_feeder - 1:
                    lv_ring.append(cur_main)
                prev_main = cur_main

        if len(lv_ring) > 1 and has_lv_ring:
            lv_ring.append(lv_ring[0])
            for a, b in zip(lv_ring, lv_ring[1:]):
                ln = dict(lv_main_line_t)
                ln.update(id=nid(), from_node=a, to_node=b, from_status=1, to_status=1)
                scale_cable(ln, rng.uniform(0.8 * 0.2 / n_connection_per_lv_feeder,
                                             1.2 * 0.2 / n_connection_per_lv_feeder))
                line.append(ln)

    def generate_mv_grid():
        id_source_node = nid()
        node.append({"id": id_source_node, "u_rated": 150.0e3})
        source.append({
            "id": nid(), "node": id_source_node, "status": 1,
            "u_ref": 1.05, "sk": 2000e6, "rx_ratio": 0.1, "z01_ratio": 1.0,
        })

        id_mv_busbar = nid()
        node.append({"id": id_mv_busbar, "u_rated": 10.5e3})
        for _ in range(n_parallel_hv_mv_transformer):
            transformer.append({
                "id": nid(), "from_node": id_source_node, "to_node": id_mv_busbar,
                "from_status": 1, "to_status": 1,
                "u1": 150.0e3, "u2": 10.5e3, "sn": 60.0e6, "uk": 0.203, "pk": 200e3,
                "i0": 0.01, "p0": 40e3,
                "winding_from": 1, "winding_to": 2, "clock": 5,
                "tap_side": 0, "tap_pos": 0, "tap_min": -10, "tap_max": 10, "tap_nom": 0, "tap_size": 2.5e3,
            })
            shunt.append({
                "id": nid(), "node": id_mv_busbar, "status": 1,
                "g1": 0.0, "b1": 0.0, "g0": 0.0, "b0": -1.0 / 7.0,
            })

        mv_line_t = dict(r1=0.063, x1=0.103, c1=0.4e-6, tan1=0.0004, r0=0.275, x0=0.101, c0=0.66e-6, tan0=0.0)

        for _i in range(n_mv_feeder):
            prev_node = id_mv_busbar
            for j in range(n_node_per_mv_feeder):
                cur = nid()
                node.append({"id": cur, "u_rated": 10.5e3})

                ln = dict(mv_line_t)
                ln.update(id=nid(), from_node=prev_node, to_node=cur, from_status=1, to_status=1)
                scale_cable(ln, rng.uniform(0.8 * 10.0 / n_node_per_mv_feeder,
                                             1.2 * 10.0 / n_node_per_mv_feeder))
                line.append(ln)

                if rng.random() < ratio_lv_grid:
                    generate_lv_grid(cur, 10.0 / n_node_per_mv_feeder)
                else:
                    scale = rng.uniform(0.8 * 10.0 / n_node_per_mv_feeder,
                                         1.2 * 10.0 / n_node_per_mv_feeder)
                    sym_load.append({
                        "id": nid(), "node": cur, "status": 1, "type": rng.randint(0, 2),
                        "p_specified": 0.8e6 * scale, "q_specified": 0.6e6 * scale,
                    })

                if j == n_node_per_mv_feeder - 1:
                    mv_ring.append(cur)
                prev_node = cur

        if len(mv_ring) > 1 and has_mv_ring:
            mv_ring.append(mv_ring[0])
            for a, b in zip(mv_ring, mv_ring[1:]):
                ln = dict(mv_line_t)
                ln.update(id=nid(), from_node=a, to_node=b, from_status=1, to_status=1)
                scale_cable(ln, rng.uniform(0.8 * 10.0 / n_node_per_mv_feeder,
                                             1.2 * 10.0 / n_node_per_mv_feeder))
                line.append(ln)

    generate_mv_grid()

    doc = {
        "version": "1.0",
        "type": "input",
        "is_batch": False,
        "attributes": {},
        "data": {
            "node": node,
            "line": line,
            "source": source,
            "sym_load": sym_load,
            "asym_load": asym_load,
            "transformer": transformer,
            "shunt": shunt,
        },
    }

    with open(out_path, "w") as f:
        json.dump(doc, f)

    print(f"nodes={len(node)} lines={len(line)} transformers={len(transformer)} "
          f"sym_load={len(sym_load)} asym_load={len(asym_load)} shunt={len(shunt)} "
          f"source={len(source)} lv_grids~{len(transformer) - n_parallel_hv_mv_transformer}",
          file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="path to write the generated PGM-format input.json to")
    parser.add_argument("--target-nodes", type=int, default=2200,
                         help="rough target node count (default: 2200, ~2,600 realized nodes with the default seed)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()
    generate(args.target_nodes, args.seed, args.output)
