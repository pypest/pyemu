## New Version Release steps [developers only]

When you are ready to make a new release of pyemu, follow these steps using 
`uv` (https://docs.astral.sh/uv/) to manage versioning.
1) checkout `main` or release branch (or just `develop`).


2) update version number with uv:
   ```
   uv version --bump <BUMP>
   ```
    ('BUMP' is major, minor, patch, stable, alpha, beta, rc, post, dev)


3) update `CITATION.cff` with version number and date


4) commit version number change: 
   ```
   git add pyproject.toml uv.lock CITATION.cff
   git commit -m"Bump version to <newversion>"
   ```
   
5) tag new version:
   ```
   git tag -a v<newversion> -m <version message>
   ```
   **Note: the 'v' is v important!**


6) push branch (e.g. `main`) and tags to remote:
   ```
   git push origin main
   git push origin v<newversion>
   ```
   If everything is setup on the CI (GitHub Actions) and PyPI, 
   .github/workflows/release.yml should take care of the rest.
   
   **DEPLOYMENT MAY NEED TO BE APPROVED MANUALLY ON GITHUB!**


7) (optional) bring `develop` or other branch up to date:
   ```
   git checkout develop
   git merge main
   git push origin develop
   ```

Check out example here:
\
https://docs.astral.sh/uv/guides/integration/github/#publishing-to-pypi
\
and here:
\
https://docs.pypi.org/trusted-publishers/adding-a-publisher/