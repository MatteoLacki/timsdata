reinstall:
	pip uninstall timsdata -y
	pip install -e .
test:
	python ~/Projects/bruker/test_timsdata.py
