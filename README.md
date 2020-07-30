## Compile the extension on Windows 10

It is easier to use CMake with "NMake Makefiles" as the generator.

1. Active the Python virtualenv in `x64 Native Tools Command Prompt for VS 2019`
2. `mkdir build; cd build`
3. `cmake -S.. -B. -G"NMake Makefiles"`

In this way, CMake can successfully locate Python development artifacts (include directories and libraries), and correctly locate cuda toolset.
