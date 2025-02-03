@echo off
rem Installation script for ADS on native Windows platforms
rem
rem Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
rem License: see the file LICENSE
rem
rem Usage: install_ads.bat <version>
rem e.g.
rem        install_ads.bat 5.5

echo:
echo *******************************
echo * Welcome to ADS installation *
echo *******************************

rem This option is needed for expanding !git_ref!, which is set (*and expanded*!) inside the 'if' statement below.
rem See also https://stackoverflow.com/q/9102422 for a further description of this behavior.
setLocal EnableDelayedExpansion

set TMP_DIR=%temp%\tmp-%RANDOM%%RANDOM%
mkdir %TMP_DIR%

rem Try to ensure that Git is available on the PATH prior to invoking `git clone` to avoid 'command not found' errors
rem   - See also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3912
rem NB: This *should* be handled by the git installer, and we even have reports of people running git --version
rem successfully and still getting an error. However, there are perhaps situations where someone has installed git but
rem hasn't refreshed their terminal. Manually modifying the PATH is a bit of a hacky workaround, especially if Git has
rem been installed somewhere else, but if this mitigates a user post on the forum, this will save us some dev time.
PATH=%PATH%;C:\Program Files\Git
git --version >nul 2>&1 || (
    echo ### git not found. Make sure that git is installed ^(and a fresh Command Prompt window has been opened^) before running the ADS installer.
    goto error
)

rem Default value: 'master', however this value is updated on stable release branches.
set git_ref=master

rem Check to see if the PWD contains the project source files (using `__init__.py` as a proxy for the entire source dir)
rem If it exists, then we can reliably access source files (e.g. `requirements-freeze.txt`) from the PWD.
if exist AxonDeepSeg\__init__.py (
  set ADS_SOURCE=%cd%
rem If __init__.py isn't present, then the installation script is being run by itself (i.e. without source files).
rem So, we need to clone ADS to a TMPDIR to access the source files, and update ADS_SOURCE accordingly.
) else (
  set ADS_SOURCE=%TMP_DIR%\axondeepseg
  echo:
  echo ### Source files not present. Downloading source files ^(@ !git_ref!^) to !ADS_SOURCE!...
  git clone -b !git_ref! --single-branch --depth 1 https://github.com/axondeepseg/ads_base.git !ADS_SOURCE!
  rem Since we're git cloning into a TMPDIR, this can never be an "in-place" installation, so we force "package" instead.
  set ADS_INSTALL_TYPE=package
)

rem Get installation type if not already specified
if [%ADS_INSTALL_TYPE%]==[] (
  rem The file 'requirements-freeze.txt` only exists for stable releases
  if exist %ADS_SOURCE%\requirements-freeze.txt (
    set ADS_INSTALL_TYPE=package
  rem If it doesn't exist, then we can assume that a dev is performing an in-place installation from master
  ) else (
    set ADS_INSTALL_TYPE=in-place
  )
)

rem Fetch the version of ADS from the source file
set /p ADS_VERSION=<%ADS_SOURCE%\AxonDeepSeg\version.txt

echo:
echo ### ADS version ......... %ADS_VERSION%
echo ### Installation type ... %ADS_INSTALL_TYPE%

rem if installing from git folder, then becomes default installation folder
if %ADS_INSTALL_TYPE%==in-place (
  set ADS_DIR=%ADS_SOURCE%
) else (
  set ADS_DIR=%USERPROFILE%\ads_%ADS_VERSION%
)

rem Allow user to set a custom installation directory
:while_loop_ads_dir
  echo:
  echo ### ADS will be installed here: [%ADS_DIR%]
  set keep_default_path=yes
  :while_loop_path_agreement
    set /p keep_default_path="### Do you agree? [y]es/[n]o: "
    echo %keep_default_path% | findstr /b [YyNn] >nul 2>&1 || goto :while_loop_path_agreement
  :done_while_loop_path_agreement

  echo %keep_default_path% | findstr /b [Yy] >nul 2>&1
  if %errorlevel% EQU 0 (
    rem user accepts default path, so exit loop
    goto :done_while_loop_ads_dir
  )

  rem user enters new path
  echo:
  echo ### Choose install directory.
  set /p new_install="### Warning^! Give full path ^(e.g. C:\Users\username\ads_v3.0^): "

  rem Check user-selected path for spaces
  if not "%new_install%"=="%new_install: =%" (
       echo ### WARNING: Install directory %new_install% contains spaces.
       echo ### ADS uses conda, which does not permit spaces in installation paths.
       echo ### More details can be found here: https://github.com/ContinuumIO/anaconda-issues/issues/716
       echo:
       goto :while_loop_ads_dir
  )

  rem Validate the user's choice of path
  if exist %new_install% (
    rem directory exists, so update ADS_DIR and exit loop
    echo ### WARNING: '%new_install%' already exists. Files will be overwritten.
    set ADS_DIR=%new_install%
    goto :done_while_loop_ads_dir
  ) else (
    if [%new_install%]==[]  (
      rem If no input, asking again, and again, and again
      goto :while_loop_ads_dir
    ) else (
      set ADS_DIR=%new_install%
      goto :done_while_loop_ads_dir
    )
  )
:done_while_loop_ads_dir

rem Create directory
if not exist %ADS_DIR% (
  mkdir %ADS_DIR% || goto error
)

rem Copy files to destination directory
echo:
if not %ADS_DIR%==%ADS_SOURCE% (
  echo ### Copying source files from %ADS_SOURCE% to %ADS_DIR%
  xcopy /s /e /q /y %ADS_SOURCE% %ADS_DIR% || goto error
) else (
  echo ### Skipping copy of source files ^(source and destination folders are the same^)
)

rem Clean old install setup in bin/ if existing
if exist %ADS_DIR%\bin\ (
  echo ### Removing axondeepseg softlink inside the ADS directory...
  del %ADS_DIR%\bin\axondeepseg_* || goto error
  del %ADS_DIR%\bin\download_* || goto error
)
rem Remove old python folder
if exist %ADS_DIR%\ads_conda\ (
  echo ### Removing existing 'ads_conda' folder inside the ADS directory...
  rmdir /s /q %ADS_DIR%\ads_conda\ || goto error
)
rem Remove old '.egg-info` folder created by editable installs
if exist %ADS_DIR%\ads_base.egg-info\ (
  echo ### Removing existing '.egg-info' folder inside the ADS directory...
  rmdir /s /q %ADS_DIR%\ads_base.egg-info\ || goto error
)

rem Move into the ADS installation directory
pushd %ADS_DIR% || goto error

rem Install portable miniconda instance. (Command source: https://github.com/conda/conda/issues/1977)
echo:
echo ### Downloading Miniconda installer...
curl -o %TMP_DIR%\miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
echo:
echo ### Installing portable copy of Miniconda...
start /wait "" %TMP_DIR%\miniconda.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%cd%\ads_conda

rem Create and activate miniconda environment to install ADS into
echo:
echo ### Using Conda to create virtual environment...
ads_conda\Scripts\conda create -y -p ads_conda\envs\venv_ads python=3.11 || goto error
CALL ads_conda\Scripts\activate.bat ads_conda\envs\venv_ads || goto error
echo Virtual environment created and activated successfully!

rem Install ADS and its requirements
if exist requirements-freeze.txt (
  set requirements_file=requirements-freeze.txt || goto error
) else (
  set requirements_file=requirements.txt || goto error
)
echo:
echo ### Installing ADS and its dependencies from %requirements_file%...
rem Skip pip==21.2 to avoid dependency resolver issue (https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3593)
ads_conda\envs\venv_ads\python -m pip install -U "pip^!=21.2.*" || goto error
ads_conda\envs\venv_ads\Scripts\pip install -r %requirements_file% || goto error
ads_conda\envs\venv_ads\Scripts\pip install -e . plugins\ --use-pep517 || goto error

rem Install external dependencies
echo:
echo ### Downloading model files and test data...
ads_conda\envs\venv_ads\Scripts\download_model -d AxonDeepSeg/models/ || goto error
ads_conda\envs\venv_ads\Scripts\download_tests

rem Copying ADS scripts to an isolated folder (so we can add scripts to the PATH without adding the entire venv_ads)
echo:
echo ### Copying ADS's CLI scripts to %CD%\bin\
xcopy %CD%\ads_conda\envs\venv_ads\Scripts\axondeepseg_*.* %CD%\bin\ /v /y /q /i || goto error
xcopy %CD%\ads_conda\envs\venv_ads\Scripts\download_*.* %CD%\bin\ /v /y /q /i || goto error
echo cmd /k %CD%\ads_conda\Scripts\activate.bat venv_ads> %CD%\bin\ads_activate.bat
echo cmd /k %CD%\ads_conda\envs\venv_ads\Scripts\napari.exe> %CD%\bin\ads_napari.bat

echo ### Checking installation...
ads_conda\envs\venv_ads\Scripts\axondeepseg_test

rem Give further instructions that the user add the Scripts directory to their PATH
echo:
echo ### Installation finished!
echo:
echo To use ADS's command-line scripts in Command Prompt, please follow these instructions:
echo:
echo 1. Open the Start Menu -^> Type 'edit environment' -^> Open 'Edit environment variables for your account'
echo 2. Click 'New', then enter 'ADS_DIR' for the variable name. For the value, copy and paste this directory:
echo:
echo    %CD%
echo:
echo 3. Click 'OK', then click on the 'Path' variable, then click the 'Edit...' button.
echo 4. Click 'New', then copy and paste this directory:
echo:
echo    %CD%\bin\
echo:
echo 5. Click 'OK' three times. You can now access ADS's scripts in the Command Prompt.
echo:
echo If you have any questions or concerns, feel free to create a new topic on ADS's forum:
echo   --^> https://github.com/axondeepseg/axondeepseg/discussions

rem Return to initial directory and deactivate the virtual environment
goto exit

:error
set cached_errorlevel=%errorlevel%
echo:
echo Installation failed with error code %cached_errorlevel%.
echo Please copy and paste the installation log in a new topic on ADS's forum:
echo   --^> https://github.com/axondeepseg/axondeepseg/discussions

:exit
if "%cached_errorlevel%"=="" set cached_errorlevel=0
popd
where deactivate >nul 2>&1
if %errorlevel% EQU 0 call conda deactivate
PAUSE
exit /b %cached_errorlevel%
