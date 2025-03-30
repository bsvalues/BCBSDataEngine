{pkgs}: {
  deps = [
    pkgs.opencl-headers
    pkgs.ocl-icd
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.geckodriver
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.openssl
  ];
}
