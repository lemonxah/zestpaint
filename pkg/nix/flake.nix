{
  description = "ZestSurface - 3D texture painting for PBR materials";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        zestsurface = pkgs.callPackage ./default.nix { };
      in
      {
        packages.default = zestsurface;

        devShells.default = pkgs.mkShell {
          inputsFrom = [ zestsurface ];
          packages = with pkgs; [
            rust-analyzer
            clippy
            rustfmt
          ];
          env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath zestsurface.runtimeLibs;
        };
      }
    );
}
