.PHONY: release test clean

release:
	cmake --preset desktop-release
	cmake --build --preset desktop-release

test:
	cmake --preset desktop-debug
	cmake --build --preset desktop-debug --target anira_tilde_tests
	ctest --preset desktop-debug

clean:
	rm -rf build/ externals/
