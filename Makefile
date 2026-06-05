.PHONY: build clean publish

build:
	uv build

clean:
	rm -rf build dist tinytasktree.egg-info ui/dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

publish: build
	uv publish
