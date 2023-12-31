{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    systems,
    ...
  } @ inputs: let
    forEachSystem = nixpkgs.lib.genAttrs (import systems);
  in {
    devShells =
      forEachSystem
      (system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        default = devenv.lib.mkShell {
          inherit inputs pkgs;
          modules = [
            {
              # https://devenv.sh/reference/options/
              packages = with pkgs; [
                gcc
                zlib
                stdenv.cc.cc.lib
                nodePackages.pyright
                python310Packages.ruff-lsp
              ];

              dotenv.enable = true;

              enterShell = ''
                # python main.py -sc install=fish | source
              '';

              languages.python = {
                enable = true;
                version = "3.11.3";
                venv = {
                  enable = true;
                  requirements = ''
                    tqdm
                    langchain
                    openai
                    pydantic
                    genanki
                    spacy
                    python_dotenv
                  '';
                };
              };
            }
          ];
        };
      });
  };
}
