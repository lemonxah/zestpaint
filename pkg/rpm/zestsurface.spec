Name:           zestsurface
Version:        0.1.0
Release:        1%{?dist}
Summary:        3D texture painting for PBR materials

License:        GPL-3.0-or-later AND OFL-1.1-RFN AND ISC
URL:            https://github.com/lemonxah/zestsurface
Source0:        %{name}-%{version}.tar.gz

# Edition 2024 needs Rust >= 1.85. shaderc-sys builds a vendored shaderc.
BuildRequires:  rust >= 1.85
BuildRequires:  cargo
BuildRequires:  gcc-c++
BuildRequires:  make
BuildRequires:  cmake
BuildRequires:  python3
BuildRequires:  git

# Vulkan, X11 and Wayland are dlopen'd, so the automatic requires miss them.
Requires:       vulkan-loader
Requires:       libX11
Requires:       libX11-xcb
Requires:       libXcursor
Requires:       libXi
Requires:       libxkbcommon
Requires:       libxkbcommon-x11
Requires:       wayland
Requires:       hicolor-icon-theme
Recommends:     xdg-desktop-portal
Obsoletes:      zestpaint

%description
ZestSurface paints PBR textures on 3D models. It loads FBX and glTF, paints
a layer stack with masks, bakes mesh maps, and exports textures.

%prep
%autosetup

%build
export RUSTUP_TOOLCHAIN=stable
export CARGO_TARGET_DIR=target
cargo build --release --locked -p %{name}

%install
install -Dm755 target/release/%{name} %{buildroot}%{_bindir}/%{name}
install -Dm644 %{name}.desktop %{buildroot}%{_datadir}/applications/%{name}.desktop
install -Dm644 assets/icon/%{name}.svg %{buildroot}%{_datadir}/icons/hicolor/scalable/apps/%{name}.svg
for size in 16 32 48 64 128 256 512; do
    install -Dm644 "assets/icon/%{name}-${size}.png" \
        "%{buildroot}%{_datadir}/icons/hicolor/${size}x${size}/apps/%{name}.png"
done

%files
%license LICENSE
%license assets/fonts/PLEX-LICENSE.txt
%license assets/fonts/LUCIDE-LICENSE.txt
%doc README.md
%{_bindir}/%{name}
%{_datadir}/applications/%{name}.desktop
%{_datadir}/icons/hicolor/scalable/apps/%{name}.svg
%{_datadir}/icons/hicolor/16x16/apps/%{name}.png
%{_datadir}/icons/hicolor/32x32/apps/%{name}.png
%{_datadir}/icons/hicolor/48x48/apps/%{name}.png
%{_datadir}/icons/hicolor/64x64/apps/%{name}.png
%{_datadir}/icons/hicolor/128x128/apps/%{name}.png
%{_datadir}/icons/hicolor/256x256/apps/%{name}.png
%{_datadir}/icons/hicolor/512x512/apps/%{name}.png

%changelog
* Sun Nov 01 2026 Ryno Kotze <lemon.xah@gmail.com> - 0.1.0-1
- Initial package
