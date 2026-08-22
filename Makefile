.PHONY: build engine test dev clean

# Full installation: builds llama.cpp for the detected backend and installs
# the capybara CLI/GUI into ~/.capybara (and ~/.local/bin on PATH).
build:
	./install.sh

engine:
	CAPYBARA_ENGINE_ONLY=1 ./install.sh

test:
	python3 capybara.test.py -v

# Run the local development copy without installing, e.g.:
#   make dev ARGS="pull smollm"
dev:
	python3 capybara.py $(ARGS)

clean:
	rm -rf __pycache__ *.pyc .DS_Store
