#!/bin/bash

# Script to build the macOS application bundle using PyInstaller

APP_NAME="XiaoXiaoBaccarat"
ENTRY_POINT="gui/run_gui.py" # Relative to the xiaoxiao1_mac directory
ICON_FILE="assets/app_icon.icns" # Relative to the xiaoxiao1_mac directory

# Ensure we are in the script's directory's parent (xiaoxiao1_mac)
# This makes relative paths for PyInstaller work correctly.
cd "$(dirname "$0")" || exit

echo "Starting PyInstaller build for $APP_NAME..."

# Clean previous builds
echo "Cleaning previous builds (dist/ and build/ directories)..."
rm -rf ./dist
rm -rf ./build
rm -f ./${APP_NAME}.spec # Remove spec file if it exists to ensure fresh generation

# PyInstaller command
# --onedir: Create a directory bundle (recommended for GUI apps with data files)
# --windowed: No console window when the app runs
# --name: Name of the application
# --icon: Application icon (must be .icns for macOS)
# --add-data: Include assets, strategy configs, and an empty data directory structure.
#             Format is 'source_path:destination_in_bundle'.
#             The destination is relative to the app's root inside the bundle.
# --hidden-import: For modules PyInstaller might miss. Add more as needed.
# --noconfirm: Overwrite output directory without asking

PYINSTALLER_CMD="pyinstaller \\
    --name \"$APP_NAME\" \\
    --onedir \\
    --windowed \\
    --icon \"$ICON_FILE\" \\
    --add-data \"assets:assets\" \\
    --add-data \"output/strategy_configs:output/strategy_configs\" \\
    --add-data \"data:data\" \\
    --hidden-import=\"PySide6.QtSvg\" \\
    --hidden-import=\"PySide6.QtOpenGLWidgets\" \\
    --hidden-import=\"pyqtgraph.colors\" \\
    --hidden-import=\"pyqtgraph.colormap\" \\
    --noconfirm \\
    $ENTRY_POINT"

echo "Running PyInstaller command:"
echo "$PYINSTALLER_CMD"
eval "$PYINSTALLER_CMD"

# Check if build was successful
if [ -d "dist/$APP_NAME.app" ]; then
    echo ""
    echo "Build successful!"
    echo "The application bundle can be found in: dist/$APP_NAME.app"
    echo ""
    echo "To create a DMG (optional):"
    echo "  hdiutil create -volname \"$APP_NAME\" -srcfolder \"dist/$APP_NAME.app\" -ov -format UDZO \"dist/$APP_NAME.dmg\""
    echo ""
    echo "Important Notes:"
    echo "- Test the .app bundle thoroughly on a clean macOS environment if possible."
    echo "- The $ICON_FILE should be a valid .icns file for the icon to display correctly."
    echo "  You can create one from a PNG using 'iconutil' on macOS:"
    echo "    mkdir YourIcon.iconset"
    echo "    sips -z 16 16     YourIcon.png --out YourIcon.iconset/icon_16x16.png"
    echo "    sips -z 32 32     YourIcon.png --out YourIcon.iconset/icon_32x32.png"
    echo "    # ... (add other sizes: 64, 128, 256, 512, 1024 for retina if needed) ..."
    echo "    sips -z 128 128   YourIcon.png --out YourIcon.iconset/icon_128x128.png"
    echo "    sips -z 256 256   YourIcon.png --out YourIcon.iconset/icon_256x256.png"
    echo "    sips -z 512 512   YourIcon.png --out YourIcon.iconset/icon_512x512.png"
    echo "    cp YourIcon.png YourIcon.iconset/icon_256x256@2x.png # Example for retina"
    echo "    iconutil -c icns YourIcon.iconset -o YourIcon.icns"
    echo "- If the 'data' directory in the bundle needs an initial baccarat.db, ensure it's copied there"
    echo "  or modify the --add-data for 'data/baccarat.db:data/baccarat.db' if it exists at build time."
    echo "  Currently, it just creates an empty 'data' directory in the bundle."
    echo "- You might need to add more --hidden-import options if the app fails to run due to missing modules."

else
    echo ""
    echo "Build failed. Check the output and warnings above."
fi
