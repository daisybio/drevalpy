<!-- Many thanks for contributing to this project! -->

# PR Checklist for all PRs

<!-- Please fill in the appropriate checklist below (delete whatever is not relevant). These are the most common things requested on pull requests (PRs). -->

- [ ] This comment contains a description of changes (with reason)
- [ ] Referenced issue is linked
- [ ] If you've fixed a bug or added code that should be tested, add tests!
- [ ] Documentation in `docs` is updated. If you've created a new file, add it to the API documentation pages.

<!-- Only applies to PRs for a new version release, delete the lines that don't apply -->

### Changes

### Bug fixes

### New features

### Maintenance

## Version release checklist

- [ ] Update the version in pyproject.toml
- [ ] Update version/release in docs/conf.py
- [ ] Run ‚poetry update‘ to get the latest package versions. This will update the poetry.lock file
- [ ] Run ‚poetry export --without-hashes --without development -f requirements.txt -o requirements.txt‘ to update the requirements.txt file
- [ ] (If one of the sphinx packages has been updated, you also need to update docs/requirements.txt)
- [ ] (If poetry itself was updated, update that in the Dockerfile)
- [ ] If you updated the python version:
  - [ ] Update the Dockerfile so that it always runs on the latest python version. Watch out: the ‚builder‘ is the full python, the ‚runtime‘ is a slim python build.
  - [ ] Update the python version in .github/workflows/: run_tests.yml, build_package.yml, publish_docs.yml, python-package.yml
  - [ ] Update the python version in noxfile.py
  - [ ] Update the documentation: contributing.rst, installation.rst

Then,

1. Open a PR from development to main with these changes.
2. Wait for a review and merge.
3. Create a new release on GitHub with the version number. Update the release notes with the changes made in this version.
