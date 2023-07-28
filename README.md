# Documents

This repository includes documents are described by TeX format.

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

* [Tex Wiki](https://texwiki.texjp.org/Linux)
* [TeX Live](https://tug.org/texlive/)

