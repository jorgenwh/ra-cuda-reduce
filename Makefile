.PHONY: install uninstall clean

install: clean
	pip install -e .

uninstall: clean
	pip uninstall cured

clean:
	$(RM) -rf build/ cured.egg-info *.so
