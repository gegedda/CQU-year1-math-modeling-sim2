@echo off
REM ============================================================
REM B题 全脚本一键验证
REM 运行方式: 双击此文件，或在终端中运行 run_all.bat
REM ============================================================
REM 使用系统 PATH 中的 python，或手动指定路径
set PYTHON=python
REM 自动检测脚本所在目录
set ROOT=%~dp0

echo ============================================================
echo   B题 全脚本验证开始
echo   %date% %time%
echo ============================================================

echo.
echo [1/9] 问题1 基础求解 (NN + 2-opt + SA) ...
%PYTHON% "%ROOT%\src\problem1\solve_all.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题1 基础求解 & exit /b 1 ) else ( echo [OK] )

echo.
echo [2/9] 问题1 SA 参数敏感性分析 ...
%PYTHON% "%ROOT%\src\problem1\sensitivity_analysis.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题1 敏感性分析 & exit /b 1 ) else ( echo [OK] )

echo.
echo [3/9] 问题1 四算法对比 ...
%PYTHON% "%ROOT%\src\problem1\algorithm_comparison.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题1 算法对比 & exit /b 1 ) else ( echo [OK] )

echo.
echo [4/9] 问题2 分层 TSP ...
%PYTHON% "%ROOT%\src\problem2\hierarchical_solver.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题2 分层TSP & exit /b 1 ) else ( echo [OK] )

echo.
echo [5/9] 问题2 换刀时间敏感性 ...
%PYTHON% "%ROOT%\src\problem2\sensitivity_analysis.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题2 敏感性 & exit /b 1 ) else ( echo [OK] )

echo.
echo [6/9] 问题3 槽位分配 ...
%PYTHON% "%ROOT%\src\problem3\slot_assignment.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题3 槽位分配 & exit /b 1 ) else ( echo [OK] )

echo.
echo [7/9] 问题3 单件取放 ...
%PYTHON% "%ROOT%\src\problem3\pick_place_single.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题3 单件取放 & exit /b 1 ) else ( echo [OK] )

echo.
echo [8/9] 问题3 预拾取 ...
%PYTHON% "%ROOT%\src\problem3\pick_place_double.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题3 预拾取 & exit /b 1 ) else ( echo [OK] )

echo.
echo [9/9] 问题3 配对策略对比 ...
%PYTHON% "%ROOT%\src\problem3\pairing_comparison.py"
if %errorlevel% neq 0 ( echo [FAIL] 问题3 配对对比 & exit /b 1 ) else ( echo [OK] )

echo.
echo ============================================================
echo   全部 9 个脚本验证通过
echo   %date% %time%
echo ============================================================
pause
