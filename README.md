# Logit Lens

## Installation

### UV
- Option1: Writing to pyproject.toml
    ```toml
    dependencies = [
            "logit-lens"
    ]

    [tool.uv.sources]
    logit-lens = { git = "https://github.com/gokamoda/logit-lens" }
    ```

- Option2: uv add
```bash
uv add git+https://github.com/gokamoda/logit-lens
```

## Usage
See [example.py](src/example.py) or [example.ipynb](examples/example.ipynb) for usage examples.