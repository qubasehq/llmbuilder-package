@echo off
echo Installing documentation dependencies...
python -m pip install -r docs/requirements.txt

echo.
echo Starting documentation server...
python -m mkdocs serve

echo.
echo If successful, open: http://127.0.0.1:8000/
pause