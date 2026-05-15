# Documentation Deployment

This project publishes documentation with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and Read the
Docs. The documentation build installs only docs dependencies and never installs
the `nunchaku_lite` package, because package installation builds the CUDA
extension.

The expected hosted URL is:

```text
https://nunchaku-lite.readthedocs.io/en/latest/
```

## Update Flow

1. Edit Markdown files under `docs/`.
2. Update `mkdocs.yml` when adding, removing, or renaming pages.
3. Validate the site locally:

   ```bash
   python -m pip install -r docs/requirements.txt
   mkdocs build --strict
   ```

4. Preview the site when needed:

   ```bash
   mkdocs serve
   ```

5. Open a pull request. The docs workflow runs `mkdocs build --strict`.
6. Merge to `main` after the docs and CPU test workflows pass.

## Read the Docs Setup

Create or connect a Read the Docs project with the slug `nunchaku-lite`.
Configure it to build from this GitHub repository. The repository includes
`.readthedocs.yml`, so Read the Docs will:

- use Python 3.12 on Ubuntu;
- install `docs/requirements.txt`;
- build the MkDocs site from `mkdocs.yml`;
- fail the build on MkDocs warnings.

After the Read the Docs project is connected, merges to `main` should trigger a
new `latest` build automatically.

## Troubleshooting

If Read the Docs fails, reproduce the build locally:

```bash
python -m pip install -r docs/requirements.txt
mkdocs build --strict
```

Fix warnings as build failures. Common issues are missing pages in `mkdocs.yml`,
broken relative links, or Markdown syntax that renders differently under strict
mode.
