#!/usr/bin/env bash
# Fill the formula's sha256 values from a published release: update.sh v0.1.0
set -euo pipefail
TAG=$1; V=${TAG#v}; F=$(dirname "$0")/jouledb.rb
sed -i.bak "s/^  version \".*\"/  version \"$V\"/" "$F"
for t in aarch64-apple-darwin x86_64-apple-darwin aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu; do
  sum=$(curl -fsSL "https://github.com/openIE-dev/jouledb/releases/download/$TAG/jouledb-$V-$t.tar.gz.sha256" | cut -d' ' -f1)
  key=REPLACE_$(echo "$t" | tr 'a-z-' 'A-Z_')
  sed -i.bak "s/\"$key\"\|\"[0-9a-f]\{64\}\" # $t/\"$sum\"/" "$F"
done
rm -f "$F.bak"; echo "updated $F for $TAG"
