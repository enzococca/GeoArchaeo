"""
Controllo statico stile "Qt6 compatibility check" del repository plugin QGIS.

Segnala accessi a enum non scoped (es. Qt.UserRole invece di Qt.ItemDataRole.UserRole),
exec_()/print_() e QVariant.Type.X. Usa le classi PyQt5/QGIS reali per sapere quali
attributi sono valori di enum e a quale enum scoped appartengono.

Uso: python tests/qt6_enum_check.py [file.py ...]   (default: tutti i .py del plugin)
"""
import os
import re
import sys
import enum

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qgis.PyQt import QtCore, QtGui, QtWidgets, QtPrintSupport  # noqa: E402
from qgis import core as qgis_core, gui as qgis_gui  # noqa: E402

CLASS_SOURCES = [QtCore, QtGui, QtWidgets, QtPrintSupport, qgis_core, qgis_gui]
PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTR_RE = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\.([A-Z][A-Za-z0-9_]*)\b(?!\s*\()')
IGNORED_PARENTS = {'QVariant'}       # QVariant.Int e' la forma raccomandata da QGIS


def _enum_members(cls):
    """{nome_membro_non_scoped: nome_enum_scoped} per una classe Qt/QGIS."""
    members = {}
    try:
        names = dir(cls)
    except Exception:
        return members
    for name in names:
        if not name[:1].isupper():
            continue
        try:
            value = getattr(cls, name)
        except Exception:
            continue
        value_type = type(value)
        if isinstance(value, enum.Enum) or (value_type.__module__.startswith('PyQt') and
                                            value_type.__name__ != 'builtin_function_or_method'
                                            and not isinstance(value, type)):
            scope = value_type.__name__
            if scope != name and hasattr(cls, scope) and getattr(cls, scope) is value_type:
                members[name] = scope
    return members


# Membri che l'introspezione non vede (enum QGIS gia' convertiti) ma che il
# controllo del repository segnala.
EXTRA = {
    'Qgis': {'Info': 'MessageLevel', 'Warning': 'MessageLevel', 'Critical': 'MessageLevel',
             'Success': 'MessageLevel', 'Float32': 'DataType'},
    'QgsWkbTypes': {'Point': 'Type', 'PointGeometry': 'GeometryType'},
    'QgsMapLayerProxyModel': {'PointLayer': 'Filter', 'PolygonLayer': 'Filter'},
    'QgsFieldProxyModel': {'Numeric': 'Filter'},
    'QgsProcessing': {'TypeVectorPoint': 'SourceType', 'TypeVectorPolygon': 'SourceType',
                      'TypeVectorAnyGeometry': 'SourceType', 'TypeVectorLine': 'SourceType'},
    'QgsProcessingParameterField': {'Numeric': 'DataType', 'DateTime': 'DataType', 'String': 'DataType'},
    'QgsProcessingParameterNumber': {'Double': 'Type', 'Integer': 'Type'},
    'QgsColorRampShader': {'Interpolated': 'Type', 'Discrete': 'Type', 'Exact': 'Type'},
}


def build_index():
    index = {k: dict(v) for k, v in EXTRA.items()}
    for module in CLASS_SOURCES:
        for cls_name in dir(module):
            if not cls_name[:1] == 'Q':
                continue
            cls = getattr(module, cls_name)
            if not isinstance(cls, type):
                continue
            members = _enum_members(cls)
            if members:
                index.setdefault(cls_name, {}).update(members)
    return index


def check_file(path, index):
    issues = []
    with open(path, encoding='utf-8') as handle:
        for lineno, line in enumerate(handle, 1):
            code = line.split('#', 1)[0]
            for match in ATTR_RE.finditer(code):
                parent, member = match.groups()
                if parent in IGNORED_PARENTS:
                    continue
                scope = index.get(parent, {}).get(member)
                if scope:
                    issues.append((lineno, match.start() + 1, f"Enum error, add '{scope}' before '{member}'"))
            if re.search(r'\.exec_\s*\(', code):
                issues.append((lineno, code.find('.exec_') + 1, "This member should be renamed to 'exec'"))
            if re.search(r'\.print_\s*\(', code):
                issues.append((lineno, code.find('.print_') + 1, "This member should be renamed to 'print'"))
            if 'QVariant.Type.' in code:
                issues.append((lineno, code.find('QVariant.Type.') + 1,
                               "QVariant.Type.XXX doesn't exist, it should be QVariant.XXX"))
    return issues


def plugin_files():
    files = []
    for root, dirs, names in os.walk(PLUGIN_DIR):
        dirs[:] = [d for d in dirs if d not in ('.git', '.idea', '__pycache__', 'tests', '.pytest_cache')]
        for name in names:
            if name.endswith('.py') and name != 'convert_guide_to_pdf.py':
                files.append(os.path.join(root, name))
    return sorted(files)


def main(paths=None):
    index = build_index()
    total = 0
    for path in (paths or plugin_files()):
        for lineno, col, message in check_file(path, index):
            total += 1
            print(f"{os.path.relpath(path, PLUGIN_DIR)}\t{lineno}\t{col}\t{message}")
    print(f"Issues found: {total}")
    return total


if __name__ == '__main__':
    sys.exit(1 if main(sys.argv[1:]) else 0)
