<!-- Many thanks for contributing to this project! -->

# PR Checklist for all PRs

<!-- Please fill in the appropriate checklist below (delete whatever is not relevant). These are the most common things requested on pull requests (PRs). -->

- [ ] This comment contains a description of changes (with reason)
- [ ] This comment contains a human-written summary of the changes (not AI-generated)
- [ ] Referenced issue is linked
- [ ] If you've fixed a bug or added code that should be tested, add tests!
- [ ] Documentation in `docs` is updated. Public modules under documented packages are picked up automatically by the recursive API autosummary.

<!-- Only applies to PRs for a new version release, delete the lines that don't apply -->

### Changes

### Bug fixes

### New features

### Maintenance

## Version release checklist

- [ ] Update the version in pyproject.toml
- [ ] Update the fallback `release` in docs/conf.py
- [ ] Run `uv lock --upgrade` to get the latest package versions. This updates uv.lock; commit it
- [ ] Run `uv sync --extra cpu` and `uv run pytest -m "not network"` to check the updated lock file
- [ ] (If the docs dependencies were updated, run `uv run --group docs sphinx-build -W docs docs/_build`)
- [ ] If you updated the python version:
  - [ ] Update the Dockerfile so that it always runs on the latest python version. Watch out: the 'builder' is the full python, the 'runtime' is a slim python build.
  - [ ] Update the python version in .github/workflows/: run_tests.yml, build_package.yml
  - [ ] Update the python version in .readthedocs.yml
  - [ ] Update `requires-python` in pyproject.toml
  - [ ] Update the documentation: docs/how-to/install.rst

Then,

1. Open a PR from development to main with these changes.
2. Wait for a review and merge.
3. Create a new release on GitHub with the version number. Update the release notes with the changes made in this version.
4. If the Docker github action fails (e.g., no space left on device), you can do it manually:
   - Enable Docker buildx locally. If you don't have a builder that supports multi-arch, create one:
   ```{bash}
   docker buildx ls
   docker buildx create --use --name multiarch-builder
   docker buildx inspect --bootstrap
   ```
   - Login to ghcr `docker login --username <GitHub-Username> --password <personal-access-token> ghcr.io`
   - Build the multi-platform image, insert the correct version `docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/daisybio/drevalpy:v<INSERT-VERSION> -t ghcr.io/daisybio/drevalpy:latest --push .`
