{
  lib,
  rustPlatform,
  cmake,
  python3,
  git,
  makeWrapper,
  autoAddDriverRunpath,
  vulkan-loader,
  libxkbcommon,
  wayland,
  xorg,
}:

let
  root = ../..;
  # Loaded with dlopen, so ldd does not see them. The wrapper puts them on
  # LD_LIBRARY_PATH; autoAddDriverRunpath lets libvulkan find the NixOS GPU driver.
  runtimeLibs = [
    vulkan-loader
    libxkbcommon
    wayland
    xorg.libX11
    xorg.libXcursor
    xorg.libXi
  ];
in
rustPlatform.buildRustPackage {
  pname = "zestsurface";
  # Version and crates come from the repo itself, so nothing here needs
  # updating per release
  version = (lib.importTOML (root + "/Cargo.toml")).package.version;

  src = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.unions [
      (root + "/Cargo.toml")
      (root + "/Cargo.lock")
      (root + "/src")
      (root + "/assets")
      (root + "/tools/matfetch")
      (root + "/zestsurface.desktop")
      (root + "/README.md")
      (root + "/LICENSE")
    ];
  };

  cargoLock.lockFile = root + "/Cargo.lock";

  cargoBuildFlags = [ "-p" "zestsurface" ];

  # shaderc-sys builds its vendored shaderc with cmake during the Rust build.
  dontUseCmakeConfigure = true;

  nativeBuildInputs = [
    cmake
    python3
    git
    makeWrapper
    autoAddDriverRunpath
  ];

  buildInputs = runtimeLibs;

  postInstall = ''
    install -Dm644 zestsurface.desktop $out/share/applications/zestsurface.desktop
    install -Dm644 README.md $out/share/doc/zestsurface/README.md
    install -Dm644 LICENSE $out/share/licenses/zestsurface/LICENSE
    install -Dm644 assets/fonts/PLEX-LICENSE.txt $out/share/licenses/zestsurface/PLEX-LICENSE.txt
    install -Dm644 assets/fonts/LUCIDE-LICENSE.txt $out/share/licenses/zestsurface/LUCIDE-LICENSE.txt
    install -Dm644 assets/icon/zestsurface.svg $out/share/icons/hicolor/scalable/apps/zestsurface.svg
    for size in 16 32 48 64 128 256 512; do
      install -Dm644 "assets/icon/zestsurface-$size.png" \
        "$out/share/icons/hicolor/''${size}x''${size}/apps/zestsurface.png"
    done
    wrapProgram $out/bin/zestsurface \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibs}
  '';

  passthru = {
    inherit runtimeLibs;
  };

  meta = {
    description = "3D texture painting for PBR materials";
    homepage = "https://github.com/lemonxah/zestsurface";
    license = with lib.licenses; [
      gpl3Plus
      ofl
      isc
    ];
    maintainers = [ ];
    platforms = lib.platforms.linux;
    mainProgram = "zestsurface";
  };
}
