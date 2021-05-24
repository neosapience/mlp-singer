.PHONY: style

style:
	black . --exclude=hifi=gan
	isort . --profile=black --skip=hifi-gan