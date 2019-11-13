install:
	pip install .
install_e:
	pip install -e .
reinstall:
	pip uninstall timsdata -y
	pip install .
reinstall_e:
	pip uninstall timsdata -y
	pip install -e .
test:
	python ~/Projects/bruker/test_timsdata.py
