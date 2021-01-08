.PHONY: build clean

build:
	mkdir -p build
	mpic++ -o ./build/sorter main.cxx

clean:
	rm -rf ./build
