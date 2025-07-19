{
  description = "Otter Backend Flake";
  inputs = {
    nixpkgs = {url = "github:NixOS/nixpkgs/nixos-25.05";};
    nixpkgs-unstable = {url = "github:NixOS/nixpkgs/nixos-unstable";};
    flake-utils = {url = "github:numtide/flake-utils";};
  };

  outputs = {
    self,
    nixpkgs,
    nixpkgs-unstable,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      inherit (pkgs.lib) optional optionals;
      pkgs = import nixpkgs {inherit system;};
      unstable = import nixpkgs-unstable {inherit system;};
    in
      with pkgs; {
        formatter = nixpkgs.legacyPackages.${system}.alejandra;

        devShells.default = pkgs.mkShell {
          buildInputs = [
            zig
            glfw
            vulkan-headers
            vulkan-loader
            vulkan-tools
            vulkan-tools-lunarg
            vulkan-validation-layers
            shader-slang
            shaderc
          ];

          LD_LIBRARY_PATH="${glfw}/lib:${freetype}/lib:${vulkan-loader}/lib:${vulkan-validation-layers}/lib";
          VULKAN_SDK = "${vulkan-headers}";
          VK_LAYER_PATH = "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
      });
}

