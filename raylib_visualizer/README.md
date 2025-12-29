# Raylib Scope Visualizer

This directory contains a small Raylib-based visualizer for Perceptrome scope data.

## Build

Raylib must be installed on your system. Then:

```bash
make
```

## Run

```bash
./perceptrome_scope generated/scope_snapshot.csv
```

If the CSV file does not exist, the visualizer will display synthetic sample data instead.

## CSV format

Each non-empty line should contain two floating point values:

```
error_value,metric_value
```

Lines beginning with `#` are ignored.
