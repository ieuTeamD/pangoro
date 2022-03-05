cd "C:\Users\khalh\Desktop\APY\GroupProject\pangoro"
py -m pip install --upgrade build

python setup.py sdist
py -m twine upload --repository pypi dist/*

pip install --upgrade pangoro

#py -m twine upload --repository testpypi dist/*
