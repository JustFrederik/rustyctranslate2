use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use build_target::{Arch, Family, Os};
use cmake::Config;

fn main() {
    use std::process::Command;

    fn main() {
        let repo_url = "https://github.com/OpenNMT/CTranslate2.git";
        let target_dir = "target/CTranslate2";

        let repo_exists = Path::new(target_dir).exists();

        if repo_exists {
            let status = Command::new("git")
                .arg("-C")
                .arg(target_dir)
                .arg("pull")
                .status()
                .expect("Failed to execute Git pull");

            if !status.success() {
                panic!("Failed to update external project");
            }
        } else {
            let status = Command::new("git")
                .arg("clone")
                .arg("--recursive")
                .arg(repo_url)
                .arg(target_dir)
                .status()
                .expect("Failed to execute Git clone");

            if !status.success() {
                panic!("Failed to clone external project");
            }
        }
    }

    let arch = build_target::target_arch().unwrap();   // eg. "x86_64", "aarch64", ...
    let family = build_target::target_family().unwrap(); // eg. "windows", "unix", ...
    let os = build_target::target_os().unwrap();     // eg. "android", "linux", ...
    let mut config = Config::new("target/CTranslate2");
    config.define("CMAKE_BUILD_TYPE", "Release").define("BUILD_CLI", "OFF");

    match family {
        Family::Unix => {
            match arch {
                Arch::AARCH64 | Arch::ARM => {
                    config.define("WITH_MKL", "OFF").define("WITH_RUY", "ON");
                }
                _ => {
                    config.define("WITH_DNNL", "ON");
                }
            }

            match os {
                Os::MacOs => {
                    config.define("CMAKE_INSTALL_RPATH_USE_LINK_PATH", "ON");
                    match arch {
                        Arch::AARCH64 | Arch::ARM => {
                            config.define("CMAKE_OSX_ARCHITECTURES", "arm64")
                                .define("WITH_ACCELERATE", "ON")
                                .define("OPENMP_RUNTIME", "NONE")
                        }
                        _ => {
                            //TODO:
                            // ONEAPI_INSTALLER_URL=https://registrationcenter-download.intel.com/akdlm/irc_nas/19080/m_BaseKit_p_2023.0.0.25441_offline.dmg
                            //     wget -q $ONEAPI_INSTALLER_URL
                            //     hdiutil attach -noverify -noautofsck $(basename $ONEAPI_INSTALLER_URL)
                            //     sudo /Volumes/$(basename $ONEAPI_INSTALLER_URL .dmg)/bootstrapper.app/Contents/MacOS/bootstrapper --silent --eula accept --components intel.oneapi.mac.mkl.devel
                            //     ONEDNN_VERSION=3.1.1
                            //     wget -q https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
                            //     tar xf *.tar.gz && rm *.tar.gz
                            //     cd oneDNN-*
                            //     cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
                            //     make -j$(sysctl -n hw.physicalcpu_max) install
                            //     cd ..
                            //     rm -r oneDNN-*
                        }
                    }
                }
                _ => {
                    config.define("OPENMP_RUNTIME", "COMP");
                    match arch {
                        Arch::AARCH64 | Arch::ARM => {
                            config
                                .define("CMAKE_PREFIX_PATH", "/opt/OpenBLAS")
                                .define("WITH_OPENBLAS", "ON")
                            //TODO:
                            // OPENBLAS_VERSION=0.3.21
                            //     curl -L -O https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz
                            //     tar xf *.tar.gz && rm *.tar.gz
                            //     cd OpenBLAS-*
                            //     # NUM_THREADS: maximum value for intra_threads
                            //     # NUM_PARALLEL: maximum value for inter_threads
                            //     make TARGET=ARMV8 NO_SHARED=1 BUILD_SINGLE=1 NO_LAPACK=1 ONLY_CBLAS=1 USE_OPENMP=1 NUM_THREADS=32 NUM_PARALLEL=8
                            //     make install NO_SHARED=1
                            //     cd ..
                            //     rm -r OpenBLAS-*
                        }
                        _ => {
                            config.define("CMAKE_CXX_FLAGS", "-msse4.1")
                                .define("WITH_CUDA", "ON")
                                .define("WITH_CUDNN", "ON")
                                .define("CUDA_DYNAMIC_LOADING", "ON")
                                .define("CUDA_NVCC_FLAGS", "-Xfatbin=-compress-all")
                                .define("CUDA_ARCH_LIST", "Common")
                            //TODO:
                            // # Install CUDA 11.2, see:
                            //     # * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/base/Dockerfile
                            //     # * https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.2.2/centos7-x86_64/devel/Dockerfile
                            //     yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
                            //     yum install --setopt=obsoletes=0 -y \
                            //         cuda-nvcc-11-2-11.2.152-1 \
                            //         cuda-cudart-devel-11-2-11.2.152-1 \
                            //         libcurand-devel-11-2-10.2.3.152-1 \
                            //         libcudnn8-devel-8.1.1.33-1.cuda11.2 \
                            //         libcublas-devel-11-2-11.4.1.1043-1
                            //     ln -s cuda-11.2 /usr/local/cuda
                            //     ONEAPI_VERSION=2023.0.0
                            //     yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
                            //     rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
                            //     yum install -y intel-oneapi-mkl-devel-$ONEAPI_VERSION
                            //     ONEDNN_VERSION=3.1.1
                            //     curl -L -O https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
                            //     tar xf *.tar.gz && rm *.tar.gz
                            //     cd oneDNN-*
                            //     cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
                            //     make -j$(nproc) install
                            //     cd ..
                            //     rm -r oneDNN-*
                        }
                    }
                }
            }
        }
        Family::Windows => {
            //TODO:
            // CUDA_ROOT=""
            // curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_461.33_win10.exe
            // ./cuda.exe -s nvcc_11.2 cudart_11.2 cublas_dev_11.2 curand_dev_11.2
            // rm cuda.exe
            // curl -L -nv -o cudnn.zip https://developer.download.nvidia.com/compute/redist/cudnn/v8.1.1/cudnn-11.2-windows-x64-v8.1.1.33.zip
            // unzip cudnn.zip && rm cudnn.zip
            // cp -r cuda/* "$CUDA_ROOT"
            // rm -r cuda/
            // # See https://github.com/oneapi-src/oneapi-ci for installer URLs
            // curl -L -nv -o webimage.exe https://registrationcenter-download.intel.com/akdlm/irc_nas/19078/w_BaseKit_p_2023.0.0.25940_offline.exe
            // ./webimage.exe -s -x -f webimage_extracted --log extract.log
            // rm webimage.exe
            // ./webimage_extracted/bootstrapper.exe -s --action install --components="intel.oneapi.win.mkl.devel" --eula=accept -p=NEED_VS2017_INTEGRATION=0 -p=NEED_VS2019_INTEGRATION=0 --log-dir=.
            // ONEDNN_VERSION=3.1.1
            // curl -L -O https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${ONEDNN_VERSION}.tar.gz
            // tar xf *.tar.gz && rm *.tar.gz
            // cd oneDNN-*
            // cmake -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;REORDER" -DONEDNN_BUILD_GRAPH=OFF .
            // cmake --build . --config Release --target install --parallel 2
            // cd ..
            // rm -r oneDNN-*
            config.define("CMAKE_INSTALL_PREFIX", "$CTRANSLATE2_ROOT")
                .define("CUDA_TOOLKIT_ROOT_DIR", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2")
                .define("CMAKE_PREFIX_PATH", "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/windows/compiler/lib/intel64_win;C:/Program Files (x86)/oneDNN")
                .define("WITH_DNNL", "ON")
                .define("WITH_CUDA", "ON")
                .define("WITH_CUDNN", "ON")
                .define("CUDA_DYNAMIC_LOADING", "ON")
                .define("CUDA_NVCC_FLAGS", "-Xfatbin=-compress-all")
                .define("CUDA_ARCH_LIST", "Common");
        }
        _ => unimplemented!()
    }

    let dst = config.build();

    cxx_build::bridge("src/lib.rs")
        .flag_if_supported("-std=c++17")
        .include(Path::new(&format!("{}/include", dst.display())))
        .compile("ctranslate2rs");

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=ctranslate2");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=include/translator.h");
}
