# Documents

This repository includes documents are described by TeX format.

- Why using TeX?
  - Various format documents are are able to make by same TeX source, because the source and the template are separated.
  - Easy to write the formulas.
  - Easy to compare diffs of document changes.

## Usage

### Docker build

```
$ docker-compose build
```

### Create Container

```
$ docker-compose up -d
```

### Build TeX

```
$ docker-compose exec --user $UID tex bash
$ cd <project directory>/
$ uplatex <tex file>
$ dvipdfmx <dvi file>
```

## Reference

- [Tex Wiki](https://texwiki.texjp.org/Linux)
- [TeX Live](https://tug.org/texlive/)
- [LaTeXで原稿を作成する：避けるべき5つの悪習慣](https://thinkscience.co.jp/ja/articles/LaTeX-habits-to-avoid)
