@echo off
setlocal enabledelayedexpansion

echo Enter dataset number (1-3):
set /p dataset=

echo Enter agent number (1-6):
set /p agent=

echo Running tests for dataset %dataset% and agent %agent%

rem Loop through test IDs 1 to 3
for %%i in (1 2 3) do (
    echo Running test %%i...
    python main.py --dataset %dataset% --agent %agent% --test %%i
)

echo All tests completed!