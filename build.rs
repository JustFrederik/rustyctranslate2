fn main() {
    cxx_build::bridge("src/lib.rs")
        .flag_if_supported("-std=c++17")
        .compile("ctranslate2rs");
    println!("cargo:rustc-link-lib=dylib=ctranslate2");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/translator.h");
}
