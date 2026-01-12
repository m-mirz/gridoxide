# gridoxide

`gridoxide` is a power flow analysis tool written in Rust. It uses the Newton-Raphson method to solve the power flow equations for an electrical grid defined in a JSON file.

## Building

To build the project, you need to have the Rust toolchain installed. You can find instructions on how to install it at [rustup.rs](https://rustup.rs/).

Once you have Rust installed, you can build the project by running:

```bash
cargo build
```

For an optimized release build, use:

```bash
cargo build --release
```

## Running

You can run the program using `cargo run`:

```bash
cargo run
```

If you have built the project, you can also run the executable directly. From the project root:

For a debug build:
```bash
./target/debug/gridoxide
```

For a release build:
```bash
./target/release/gridoxide
```

## Testing

To run the tests for the project, use:

```bash
cargo test
```
