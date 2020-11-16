
.PHONY: clean
clean:
	rm -rf ./build
	rm -rf ./dist
	rm -rf ./mypy_cache
	rm -rf ./pytest_cache

.PHONY: format
format:
	black --line-length 100 .

.PHONY: lint
lint:
	flake8 --count --max-line-length 100 .
	black --check --diff --line-length 100 .
	mypy --ignore-missing-imports .
	# pylint disables:
	#   * C0301: line too long
	#   * C0103: snake-case naming
	#   * C0330: wrong hanging indent before block
	#   * E0401: unable to import
	#   * R0903: too few public methods
	#   * W0212: access to protected member
	pylint --disable=C0103,C0301,C0330,E0401,R0903,W0212 dask_pytorch/

.PHONY: unit-tests
unit-tests:
	pip uninstall -y dask-pytorch
	python setup.py develop
	pytest --cov=dask_pytorch tests/

.PHONY: test
test: clean lint unit-tests

.PHONY: format
	@echo -e '\n\nCheck formatting with Black...'
	black --line-length 100 --exclude '/(\.vscode|node_modules)/' .
