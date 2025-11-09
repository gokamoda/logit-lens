# Logit Lens

## Installation

### UV
- Option1: Writing to pyproject.toml
    ```toml
    dependencies = [
            "logit-lens"
    ]

    [tool.uv.sources]
    logit-lens = { git = "https://github.com/gokamoda/logit-lens", rev = "3ebcb1f" }
    ```

- Option2: uv add
```bash
uv add logit-lens --git https://github.com/gokamoda/logit-lens --rev 3ebcb1f
uv add git+https://github.com/gokamoda/logit-lens --rev 3ebcb1f
```

## Usage
See [example.py](src/example.py) or [example.ipynb](examples/example.ipynb) for usage examples.