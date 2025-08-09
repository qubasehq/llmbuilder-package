@echo off
echo 🚀 LLMBuilder Documentation - Vercel Deployment
echo.

echo Installing documentation dependencies...
python -m pip install -r docs/requirements.txt

echo.
echo Building documentation for Vercel...
python -m mkdocs build --clean --strict

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Documentation build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Documentation built successfully!
echo.

echo 📁 Site contents ready for Vercel deployment:
dir site /b

echo.
echo 🌐 Your documentation is ready for deployment to: https://llm-package.gainandshine.com/
echo.
echo 📋 Next steps:
echo 1. Install Vercel CLI: npm i -g vercel
echo 2. Login to Vercel: vercel login
echo 3. Deploy: vercel --prod
echo.
echo Or simply drag and drop the 'site' folder to Vercel dashboard.
echo.

set /p deploy="Would you like to deploy now with Vercel CLI? (y/n): "

if /i "%deploy%"=="y" (
    echo.
    echo 🚀 Deploying to Vercel...
    
    where vercel >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Vercel CLI not found. Please install it first:
        echo npm i -g vercel
        pause
        exit /b 1
    )
    
    cd site
    vercel --prod
    cd ..
    
    echo.
    echo ✅ Deployment completed!
    echo 🌐 Your docs should be live at: https://llm-package.gainandshine.com/
) else (
    echo.
    echo 📦 Site folder is ready for manual deployment
    echo 💡 You can drag and drop the 'site' folder to Vercel dashboard
)

echo.
pause