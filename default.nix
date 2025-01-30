let
  # Pin Nixpkgs for stability
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/976fa3369d722e76f37c77493d99829540d43845.tar.gz") {};

  # Define Python 3.9 environment with necessary packages
  pythonEnv = pkgs.python39.withPackages (ps: with ps; [
    certifi
    contourpy
    cycler
    fonttools
    geopandas
    ps.importlib-resources  # Corrected package name for Nix
    kiwisolver
    laspy
    matplotlib
    numpy
    packaging
    pandas
    pillow
    pyogrio
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

  # System dependencies (required for GIS and data processing)
  systemPackages = with pkgs; [
    git      # Version control
    gdal     # Geospatial data library
    proj     # Coordinate system transformations
    geos     # Geometric operations support
    pythonEnv
  ];
in
pkgs.mkShell {
  buildInputs = systemPackages;

  shellHook = ''
    echo "🔹 Reproducible Nix Environment Ready! 🔹"
    echo "💡 Run your pipeline with: python main.py"
  '';
}
