use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let include_dir = crate_dir.join("include");
    let output_path = crate_dir.join("include/pnp.h");

    if let Err(e) = std::fs::create_dir_all(&include_dir) {
        eprintln!("cargo:warning=failed to create include dir: {}", e);
    }

    let status = std::process::Command::new("cbindgen")
        .arg("--crate")
        .arg("pnp-ffi")
        .arg("-c")
        .arg(crate_dir.join("cbindgen.toml"))
        .arg("-o")
        .arg(&output_path)
        .current_dir(&crate_dir)
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("cargo:warning=cbindgen exited with {}", s),
        Err(e) => eprintln!("cargo:warning=cbindgen not found: {}", e),
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
