# Holey Tools

Holey Tools is a Blender 5.2+ extension for editing mesh holes to precise physical
dimensions. It contains the interactive **Resize Hole** and **Extract Axis** tools.

## Documentation

- [User guide](src/README.md)
- [Architecture](src/docs/architecture.md)
- [Resize Hole workflow](src/docs/resize-hole-workflow.md)
- [Extract Axis workflow](src/docs/extract-axis-workflow.md)

The uploadable extension is built from `src/`; tests and development scripts remain
at the repository root.

## Build

Build with the newest reachable Git tag matching `#.#.#`:

```sh
./build
```

Pass a version to override the tag, for example `./build 1.0.0`. On Windows, use
`./build.ps1` or `./build.ps1 1.0.0`. The helpers package a temporary source copy,
so version overrides do not modify the working tree. Set `BLENDER` to the Blender
executable when it is not available on `PATH` or in the standard installation
location.
