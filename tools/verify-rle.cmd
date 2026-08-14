@echo off
rem compile+run the RLE/median verification (same-compilation-unit access to private)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set "PATH=C:\Users\Lin\AppData\Local\Programs\Swift\Runtimes\6.3.3\usr\bin;C:\Users\Lin\AppData\Local\Programs\Swift\Toolchains\6.3.3+Asserts\usr\bin;%PATH%"
set "SDKROOT=C:\Users\Lin\AppData\Local\Programs\Swift\Platforms\6.3.3\Windows.platform\Developer\SDKs\Windows.sdk"
cd /d "%~dp0.."
type Sources\CRUXCore\ColorMath.swift > "%TEMP%\crux_verify_all.swift"
type Sources\CRUXCore\RouteSelector.swift >> "%TEMP%\crux_verify_all.swift"
type "%TEMP%\crux_verify_main.swift" >> "%TEMP%\crux_verify_all.swift"
cd /d "%TEMP%"
swiftc crux_verify_all.swift -o crux_verify_all.exe
if errorlevel 1 exit /b 1
crux_verify_all.exe
