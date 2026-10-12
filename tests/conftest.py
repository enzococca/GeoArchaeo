"""
Configurazione pytest per GeoArchaeo.

I test del motore (test_engine.py) girano con qualunque Python che abbia
numpy/scipy/scikit-learn. I test headless della GUI richiedono il Python di
QGIS, ad esempio su macOS:

    /Applications/QGIS.app/Contents/MacOS/bin/python3 -m pytest tests/
"""
import os
import sys

# Rende importabile il pacchetto plugin (cartella padre) come "GeoArchaeo"
_PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Qt headless (deve essere impostato prima di importare qgis/PyQt)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if sys.platform == 'darwin':
    _qgis_plugins = '/Applications/QGIS.app/Contents/PlugIns/platforms'
    if os.path.isdir(_qgis_plugins):
        os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', _qgis_plugins)
    _proj = '/Applications/QGIS.app/Contents/Resources/proj'
    if os.path.isdir(_proj):
        os.environ.setdefault('PROJ_LIB', _proj)

# Moduli Processing di QGIS (per i test degli algoritmi)
for _candidate in ('/Applications/QGIS.app/Contents/Resources/python/plugins',
                   '/usr/share/qgis/python/plugins'):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.append(_candidate)
