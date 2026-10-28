@echo off
rem Lance pyCBA avec Anaconda (double-clic). Aucun parametre: on peut glisser un fichier image sur ce .bat.
setlocal
set "HERE=%~dp0"
set "ACT="
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "C:\ProgramData\anaconda3" "C:\ProgramData\miniconda3" "C:\anaconda3") do (
    if not defined ACT if exist "%%~D\Scripts\activate.bat" set "ACT=%%~D\Scripts\activate.bat"
)
if not defined ACT (
    where conda >nul 2>nul && (for /f "delims=" %%C in ('where conda') do set "ACT=%%~dpC..\Scripts\activate.bat")
)
if not defined ACT (
    echo Anaconda introuvable. Editez ce fichier et mettez le chemin de "Scripts\activate.bat" dans ACT.
    pause
    exit /b 1
)
call "%ACT%" || (pause & exit /b 1)
python "%HERE%run_cba.py" %*
if errorlevel 1 pause
endlocal
