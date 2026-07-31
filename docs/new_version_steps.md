## New Version Release steps [developers only]

Releasing pyemu is a single button press: the
[`Release` workflow](https://github.com/pypest/pyemu/actions/workflows/release.yml)
does the version bump, the build, the tag and the PyPI upload for you. Nothing
needs to be done locally, and because the version is computed and tagged in the
same job, the git tag and the published artifact can never disagree.

1) Make sure the branch you want to release from is up to date and its CI is
   green. Any branch will do - the workflow releases from whichever branch you
   dispatch it on and pushes the bump commit and tag back to that same branch.

2) On GitHub go to **Actions → Release → Run workflow** and choose:
   - **Use workflow from**: the branch to release from.
   - **bump**: what to bump. This is passed straight to `uv version --bump`,
     so the options are `major`, `minor`, `patch`, `stable`, `alpha`, `beta`,
     `rc`, `post` and `dev`.
   - **repository**: `pypi` for a real release, or `testpypi` to rehearse
     (see below).
   - **dry_run**: tick this to rehearse without uploading anything at all.

3) **DEPLOYMENT MAY NEED TO BE APPROVED MANUALLY ON GITHUB!** The job runs in
   the `pypi` environment; if that environment has required reviewers the run
   pauses until it is approved.

4) The workflow then, in order:
   - runs `uv version --bump <bump>` and `uv lock`,
   - rewrites the `version` and `date-released` fields of `CITATION.cff`,
   - runs `uv build` and smoke tests both the wheel and the sdist,
   - commits the bump, tags it `v<newversion>` and pushes both to the branch
     you released from,
   - publishes with `uv publish`,
   - creates a GitHub release with auto-generated notes (marked as a
     pre-release for anything that is not a plain `X.Y.Z`).

   The tag is pushed *before* the upload, so if publishing fails you can delete
   the tag and the bump commit and try again.

5) (optional) bring other branches up to date, e.g. if you released from a
   release branch:
   ```
   git checkout develop
   git merge <release branch>
   git push origin develop
   ```

### Rehearsing a release

Two modes, covering different halves of the process:

- **dry_run** bumps, builds and smoke tests, then stops. Nothing leaves the
  runner. It does *not* exercise the push or the upload, so it proves neither
  the `contents: write` permission nor the trusted-publishing setup.
- **repository = testpypi** does a real upload to TestPyPI, but deliberately
  leaves no trace in git: no bump commit, no tag, no GitHub release. The
  version number stays free for the real release afterwards. Repeat rehearsals
  need a different version each time (TestPyPI, like PyPI, will not accept the
  same version twice), so bump `dev` if you are going round more than once.

For a full-fidelity rehearsal including the git writes, run the workflow on
your own fork with `repository = pypi` and dry_run off. Everything up to and
including the tag push will run; the publish step then fails because PyPI only
trusts `pypest/pyemu`, which is the expected outcome.

### Notes

- The workflow file must exist on the repository's **default branch** for the
  "Run workflow" button to appear at all. Once it is there you can dispatch it
  on any branch, and it runs that branch's copy of the file.
- Nothing in the workflow hardcodes a branch name - `main`, `develop` and
  release branches all work the same way.
- The workflow needs push rights to the branch you release from. If that branch
  is protected, either allow the `github-actions[bot]` actor to push to it or
  release from an unprotected branch and merge afterwards.
- Pushing a `v*` tag by hand no longer publishes anything - use the workflow.
- Auth uses [trusted publishing](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
  (OIDC), so there is no API token to manage. The publisher is matched on
  owner, repository, workflow filename (`release.yml`) and environment name.
  The environment name follows the `repository` input, so PyPI must trust
  environment `pypi` and TestPyPI must trust environment `testpypi`.
  See also https://docs.astral.sh/uv/guides/integration/github/#publishing-to-pypi
- `pyemu.__version__` is read from the installed package metadata, so
  `pyproject.toml` is the only place a version number is authored by hand -
  and even that is done by `uv version` inside the workflow.
