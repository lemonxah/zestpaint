# Get current version from Cargo.toml
version := `grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/'`

# Show current version
current:
    @echo "{{version}}"

# Bump version: just bump 0.3.0
bump new_version:
    # Cargo.toml + Cargo.lock (matfetch keeps its own version)
    sed -i '0,/^version = ".*"/s//version = "{{new_version}}"/' Cargo.toml
    cargo update --workspace
    # Arch PKGBUILD
    sed -i 's/^pkgver=.*/pkgver={{new_version}}/' pkg/zestsurface/PKGBUILD
    # RPM spec
    sed -i 's/^Version:.*/Version:        {{new_version}}/' pkg/rpm/zestsurface.spec
    @echo "Bumped to {{new_version}}"

# Run all tests
test:
    cargo test --workspace

# Build release binary
build:
    cargo build --workspace --release

# Build and install locally
install: build
    sudo install -Dm755 target/release/zestsurface /usr/bin/zestsurface
    sudo install -Dm644 zestsurface.desktop /usr/share/applications/zestsurface.desktop
    sudo install -Dm644 assets/icon/zestsurface.svg /usr/share/icons/hicolor/scalable/apps/zestsurface.svg
    sudo install -Dm644 assets/icon/zestsurface-16.png /usr/share/icons/hicolor/16x16/apps/zestsurface.png
    sudo install -Dm644 assets/icon/zestsurface-32.png /usr/share/icons/hicolor/32x32/apps/zestsurface.png
    sudo install -Dm644 assets/icon/zestsurface-48.png /usr/share/icons/hicolor/48x48/apps/zestsurface.png
    sudo install -Dm644 assets/icon/zestsurface-64.png /usr/share/icons/hicolor/64x64/apps/zestsurface.png
    sudo install -Dm644 assets/icon/zestsurface-128.png /usr/share/icons/hicolor/128x128/apps/zestsurface.png
    sudo install -Dm644 assets/icon/zestsurface-256.png /usr/share/icons/hicolor/256x256/apps/zestsurface.png
    sudo install -Dm644 assets/icon/zestsurface-512.png /usr/share/icons/hicolor/512x512/apps/zestsurface.png
    -sudo gtk-update-icon-cache -q /usr/share/icons/hicolor
    -kbuildsycoca6 --noincremental >/dev/null 2>&1
    @echo "Installed"
    @echo "KDE keeps loaded icons in memory: run 'systemctl --user restart plasma-plasmashell' (or log out and in) to see a new icon."

# Bump, commit, tag, and push. Optional message: just release 0.5.0 "fix remapping"
release new_version msg="":
    just bump {{new_version}}
    git add Cargo.toml Cargo.lock pkg/zestsurface/PKGBUILD pkg/rpm/zestsurface.spec
    git commit -m "v{{new_version}}{{ if msg != "" { ": " + msg } else { "" } }}"
    git tag -a "v{{new_version}}" -m "v{{new_version}}{{ if msg != "" { ": " + msg } else { "" } }}"
    git push && git push --tags
    gh release create "v{{new_version}}" --generate-notes

# ---- Packages ---------------------------------------------------------------
# Nix: nix build ./pkg/nix
# Arch release tarball: makepkg -si in pkg/zestsurface
# Arch git: makepkg -si in pkg/zestsurface-git

# Works on that Ubuntu release and newer; build one per target, e.g. just deb ubuntu:25.04
# Build a .deb inside an Ubuntu container (podman/docker): pkg/deb/zestsurface_<version>-<image>_amd64.deb
deb image="ubuntu:24.04":
    pkg/deb/build-deb.sh --container {{image}}

# Works on that Fedora release only; build one per target, e.g. just rpm fedora:43
# Build an .rpm inside a Fedora container (podman/docker)
rpm image="fedora:latest":
    pkg/rpm/build-rpm.sh --container {{image}}
