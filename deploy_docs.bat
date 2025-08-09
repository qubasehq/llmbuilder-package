@echo off
echo 🚀 LLMBuilder Documentation Deployment Script
echo.

echo Installing documentation dependencies...
python -m pip install -r docs/requirements.txt

echo.
echo Building documentation...
python -m mkdocs build --clean --strict

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Documentation build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Documentation built successfully!
echo.

echo Choose deployment option:
echo 1. Deploy to GitHub Pages
echo 2. Serve locally for testing
echo 3. Build only (already done)

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo 📤 Deploying to GitHub Pages...
    python -m mkdocs gh-deploy --clean
    echo ✅ Deployed to GitHub Pages!
    echo 🌐 Your docs will be available at: https://qubasehq.github.io/llmbuilder-package/
) else if "%choice%"=="2" (
    echo.
    echo 🌐 Starting local server...
    echo 📖 Documentation will be available at: http://127.0.0.1:8000/
    echo Press Ctrl+C to stop the server
    python -m mkdocs serve
) else (
    echo.
    echo ✅ Documentation built in 'site' directory
    echo To serve locally: python -m mkdocs serve
    echo To deploy: python -m mkdocs gh-deploy
)

echo.
pause