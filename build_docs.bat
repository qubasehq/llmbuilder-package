@echo off
echo Installing documentation dependencies...
python -m pip install -r docs/requirements.txt

echo.
echo Building documentation...
python -m mkdocs build

echo.
echo Documentation built in 'site' directory
echo To serve locally, run: python -m mkdocs serve
pause