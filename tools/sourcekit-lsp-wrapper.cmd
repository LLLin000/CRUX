@echo off
rem PATH shim: sourcekit-lsp needs Swift runtime/toolchain dirs that may not
rem be in the host (omp) process PATH. Prepend them, then exec.
set "SWIFT_RUNTIME=C:\Users\Lin\AppData\Local\Programs\Swift\Runtimes\6.3.3\usr\bin"
set "SWIFT_TOOLCHAIN=C:\Users\Lin\AppData\Local\Programs\Swift\Toolchains\6.3.3+Asserts\usr\bin"
set "PATH=%SWIFT_RUNTIME%;%SWIFT_TOOLCHAIN%;%PATH%"
set "SDKROOT=C:\Users\Lin\AppData\Local\Programs\Swift\Platforms\6.3.3\Windows.platform\Developer\SDKs\Windows.sdk"
"%SWIFT_TOOLCHAIN%\sourcekit-lsp.exe" %*
