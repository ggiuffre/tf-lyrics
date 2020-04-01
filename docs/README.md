# Generating tflyrics documentation

In order to generate HTML documentation for `tflyrics`, you need to install
some additional packages:

```shell
pip install Sphinx
pip install sphinx-material
pip install m2r
```

Then, you can generate the documentation like this:

```
sphinx-build -b html docs/docs_source/ docs/
```


