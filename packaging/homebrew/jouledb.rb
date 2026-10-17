# Homebrew formula for a tap (e.g. openIE-dev/homebrew-tap). The sha256 values
# come from the .sha256 files the Release workflow attaches to each tag;
# packaging/homebrew/update.sh fills them in. Until a release is cut, the
# head build compiles from main: `brew install --HEAD ./packaging/homebrew/jouledb.rb`.
class Jouledb < Formula
  desc "Energy-aware database: measured joules per query, pgwire, MCP, HDC"
  homepage "https://github.com/openIE-dev/jouledb"
  version "0.1.0"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/openIE-dev/jouledb.git", branch: "main"

  on_macos do
    on_arm do
      url "https://github.com/openIE-dev/jouledb/releases/download/v#{version}/jouledb-#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "REPLACE_AARCH64_APPLE_DARWIN"
    end
    on_intel do
      url "https://github.com/openIE-dev/jouledb/releases/download/v#{version}/jouledb-#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "REPLACE_X86_64_APPLE_DARWIN"
    end
  end
  on_linux do
    on_arm do
      url "https://github.com/openIE-dev/jouledb/releases/download/v#{version}/jouledb-#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_AARCH64_UNKNOWN_LINUX_GNU"
    end
    on_intel do
      url "https://github.com/openIE-dev/jouledb/releases/download/v#{version}/jouledb-#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_X86_64_UNKNOWN_LINUX_GNU"
    end
  end

  def install
    if build.head?
      system "cargo", "install", "--locked", *std_cargo_args(path: "crates/joule-db-server")
    else
      bin.install "jouledb"
    end
  end

  service do
    run [opt_bin/"jouledb", "-p", "8080", "-d", var/"jouledb"]
    keep_alive true
  end

  test do
    assert_match "jouledb", shell_output("#{bin}/jouledb --version")
  end
end
