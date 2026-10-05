{
  description = "This is a template for python dev";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells = builtins.mapAttrs (system: pkgs: {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python3
          uv
        ];
      };
    }) nixpkgs.legacyPackages;
  };
}
