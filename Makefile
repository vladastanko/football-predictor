.PHONY: install all demo test
install: ; pip install -e ".[dev]"
all:     ; fpred all
demo:    ; fpred demo
test:    ; pytest -q
