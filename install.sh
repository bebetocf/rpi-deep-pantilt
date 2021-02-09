pip cache purge
pip uninstall rpi_deep_pantilt -y
python setup.py bdist_wheel
pip install dist/rpi_deep_pantilt-1.2.1-py2.py3-none-any.whl