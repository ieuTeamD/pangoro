cd "C:\Users\khalh\Desktop\APY\GroupProject\pangoro"
python -m pip install --upgrade build

python setup.py sdist
python -m twine upload --repository pypi dist/*

pip install --upgrade pangoro

REM python -m twine upload --repository testpypi dist/*
