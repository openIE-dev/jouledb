//! Write an HDC similarity `.mlpackage` with a random +-1 codebook.
//!
//! `cargo run -p joule-ane-rt --example write_hdc_mlpackage -- <out.mlpackage> <matmul|conv> <n> <d> <k>`
//!
//! Works on any host (the generator is pure Rust); used to validate the
//! artifact with `scripts/verify_mlpackage.py` and `xcrun coremlcompiler`.

use joule_ane_rt::{Layout, ModelShape, write_mlpackage};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!("usage: {} <out.mlpackage> <matmul|conv> <n> <d> <k>", args[0]);
        std::process::exit(2);
    }
    let layout = if args[2] == "conv" { Layout::ChannelsFirstConv } else { Layout::MatMul };
    let parse = |s: &str| s.parse::<usize>().unwrap_or(1).max(1);
    let shape = ModelShape { layout, n: parse(&args[3]), d: parse(&args[4]), k: parse(&args[5]) };
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let codebook: Vec<i8> = (0..shape.k * shape.d)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 { 1 } else { -1 }
        })
        .collect();
    let out = std::path::Path::new(&args[1]);
    let _ = std::fs::remove_dir_all(out);
    match write_mlpackage(out, &shape, &codebook) {
        Ok(()) => println!("wrote {} ({:?})", out.display(), shape),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
