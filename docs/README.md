# Generating tflyrics documentation

In order to generate HTML documentation for `tflyrics`, you need to install
some additional packages:

```sh
pip install Sphinx
pip install sphinx-material
pip install m2r
```

Then, you can generate the documentation like this:

```sh
sphinx-build -b html docs/docs_source/ docs/
```

You can check the docs offline at `docs/index.html`, and online on the
[Github Pages site](https://ggiuffre.github.io/tf-lyrics/) of this repo.


