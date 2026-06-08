# Install

`jouledb` is published to crates.io and to GitHub Releases.

## Fastest: `cargo binstall`

```bash
cargo binstall jouledb
```

`cargo-binstall` fetches the prebuilt binary for your platform from [GitHub Releases](https://github.com/openIE-dev/jouledb/releases) without compiling anything.

If you don't have `cargo-binstall` yet:

```bash
cargo install cargo-binstall
```

## From source: `cargo install`

```bash
cargo install jouledb
```

This compiles the source. Slower; useful if your platform isn't in our prebuilt list.

## Direct download

```bash
curl -fsSL -o jouledb.tar.gz \
  https://github.com/openIE-dev/jouledb/releases/latest/download/jouledb-$(uname -s)-$(uname -m).tar.gz
tar xzf jouledb.tar.gz
mv jouledb-*/jouledb /usr/local/bin/
```

## Verify

```bash
jouledb --version
```
