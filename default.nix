let
  # Import a fixed Nixpkgs version for reproducibility
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/976fa3369d722e76f37c77493d99829540d43845.tar.gz") {};

  # Define Python 3.9 environment with required packages (excluding pyogrio)
  pythonEnv = pkgs.python39.withPackages (ps: with ps; [
    certifi
    contourpy
    cycler
    fonttools
    geopandas
    importlib-resources
    kiwisolver
    laspy
    matplotlib
    numpy
    packaging
    pandas
    pillow
    pyparsing
    pyproj
    python-dateutil
    pytz
    scipy
    seaborn
    shapely
    six
    tzdata
    zipp
  ]);

  # System dependencies for GIS and Python
  systemPackages = with pkgs; [
    git
    gdal
    proj
    geos
    pythonEnv
  ];
in
pkgs.mkShell {
  buildInputs = systemPackages;

  shellHook = ''
    echo "🔹 Nix Environment Ready!"
    echo "Installing pyogrio via pip..."
    pip install --no-cache-dir pyogrio
  '';
}
