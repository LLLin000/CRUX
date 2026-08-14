@echo off
rem Local Swift build helper: init VS BuildTools env (VC + Windows SDK) then swift build
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Users\Lin\AppData\Local\Programs\Swift\Runtimes\6.3.3\usr\bin;C:\Users\Lin\AppData\Local\Programs\Swift\Toolchains\6.3.3+Asserts\usr\bin;%PATH%"
set "SDKROOT=C:\Users\Lin\AppData\Local\Programs\Swift\Platforms\6.3.3\Windows.platform\Developer\SDKs\Windows.sdk"
cd /d "%~dp0.."
swift %*
