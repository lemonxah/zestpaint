#!/bin/bash
set -euo pipefail

# Build a .deb package for ZestSurface
# Usage:
#   ./build-deb.sh                          build on this Debian/Ubuntu machine; on
#                                           other distros, same as --container
#   ./build-deb.sh --container [IMAGE]      build inside IMAGE (default ubuntu:24.04)
#                                           with podman or docker, from any distro
#
# The .deb only works on the release it was built on (or newer), so build one
# per target, e.g. --container ubuntu:24.04, --container ubuntu:25.04.
#
# Prerequisites for a native build (Debian/Ubuntu):
#   sudo apt install build-essential cmake python3 git pkg-config dpkg-dev
#   plus Rust >= 1.85 from rustup (Ubuntu's packaged rustc is too old)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

# A .deb built against a non-Debian host's libraries won't run on Ubuntu (and
# dpkg-shlibdeps can't map them to packages), so build in a container there.
if [ "${1:-}" != "--container" ] && [ ! -f /etc/debian_version ]; then
    echo "Not a Debian/Ubuntu system: building in an ubuntu:24.04 container instead"
    set -- --container
fi

if [ "${1:-}" = "--container" ]; then
    IMAGE="${2:-ubuntu:24.04}"
    ENGINE=$(command -v podman || command -v docker || true)
    [ -n "$ENGINE" ] || { echo "podman or docker is required for --container" >&2; exit 1; }
    TAG="zestsurface-deb-build:${IMAGE//[:\/]/-}"
    "$ENGINE" build --build-arg "BASE=$IMAGE" -t "$TAG" -f "$SCRIPT_DIR/Containerfile" "$SCRIPT_DIR"
    # Separate target dir so the host's build cache isn't mixed with the container's.
    # Rootless podman maps root to the calling user; docker needs --user so the
    # output isn't owned by root.
    USER_ARGS=()
    [[ "$ENGINE" == *docker ]] && USER_ARGS=(--user "$(id -u):$(id -g)")
    "$ENGINE" run --rm "${USER_ARGS[@]}" \
        -v "$PROJECT_DIR:/src:Z" -w /src \
        -e CARGO_TARGET_DIR="/src/target/deb-${IMAGE//[:\/]/-}" \
        -e DEB_SUFFIX="${IMAGE//[:\/]/}" \
        "$TAG" pkg/deb/build-deb.sh
    exit 0
fi

VERSION=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
ARCH=$(dpkg --print-architecture)
PKG_NAME="zestsurface"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
# e.g. zestsurface_0.1.0_amd64 or zestsurface_0.1.0-ubuntu24.04_amd64
DEB_VERSION="${VERSION}${DEB_SUFFIX:+-$DEB_SUFFIX}"
PKG_DIR="$SCRIPT_DIR/${PKG_NAME}_${DEB_VERSION}_${ARCH}"

echo "Building ZestSurface v${DEB_VERSION} .deb for ${ARCH}..."

cargo build --release --locked -p "$PKG_NAME"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/DEBIAN"

install -Dm755 "$TARGET_DIR/release/$PKG_NAME" "$PKG_DIR/usr/bin/$PKG_NAME"
strip "$PKG_DIR/usr/bin/$PKG_NAME"
install -Dm644 "$PKG_NAME.desktop" "$PKG_DIR/usr/share/applications/$PKG_NAME.desktop"
install -Dm644 README.md "$PKG_DIR/usr/share/doc/$PKG_NAME/README.md"
install -Dm644 LICENSE "$PKG_DIR/usr/share/doc/$PKG_NAME/copyright"
install -Dm644 assets/fonts/PLEX-LICENSE.txt "$PKG_DIR/usr/share/doc/$PKG_NAME/PLEX-LICENSE.txt"
install -Dm644 assets/fonts/LUCIDE-LICENSE.txt "$PKG_DIR/usr/share/doc/$PKG_NAME/LUCIDE-LICENSE.txt"
install -Dm644 assets/icon/zestsurface.svg "$PKG_DIR/usr/share/icons/hicolor/scalable/apps/zestsurface.svg"
for size in 16 32 48 64 128 256 512; do
    install -Dm644 "assets/icon/zestsurface-${size}.png" \
        "$PKG_DIR/usr/share/icons/hicolor/${size}x${size}/apps/zestsurface.png"
done

# Shared-library dependencies, worked out from the binary's actual links.
# Vulkan, X11 and Wayland are dlopen'd, so dpkg-shlibdeps cannot see them.
SHLIBS_DIR=$(mktemp -d)
trap 'rm -rf "$SHLIBS_DIR"' EXIT
mkdir -p "$SHLIBS_DIR/debian"
printf 'Source: %s\n\nPackage: %s\nArchitecture: any\n' "$PKG_NAME" "$PKG_NAME" > "$SHLIBS_DIR/debian/control"
SHLIBS_DEPENDS=$(cd "$SHLIBS_DIR" && dpkg-shlibdeps -O \
    "$PKG_DIR/usr/bin/$PKG_NAME" \
    | sed -n 's/^shlibs:Depends=//p')

DLOPEN_DEPENDS="libvulkan1, libx11-6, libx11-xcb1, libxcursor1, libxi6, libxkbcommon0, libxkbcommon-x11-0, libwayland-client0, hicolor-icon-theme"

INSTALLED_SIZE=$(du -sk "$PKG_DIR" | cut -f1)

cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: zestsurface
Version: ${DEB_VERSION}
Section: graphics
Priority: optional
Architecture: ${ARCH}
Depends: ${SHLIBS_DEPENDS}, ${DLOPEN_DEPENDS}
Recommends: xdg-desktop-portal
Conflicts: zestpaint
Replaces: zestpaint
Installed-Size: ${INSTALLED_SIZE}
Maintainer: Ryno Kotze <lemon.xah@gmail.com>
Homepage: https://github.com/lemonxah/zestsurface
Description: 3D texture painting
 ZestSurface paints PBR textures on 3D models. It loads FBX and glTF, paints
 a layer stack with masks, bakes mesh maps, and exports textures.
EOF

dpkg-deb --build --root-owner-group "$PKG_DIR"
rm -rf "$PKG_DIR"

echo ""
echo "Package built: ${PKG_DIR}.deb"
