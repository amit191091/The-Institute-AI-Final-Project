import importlib
m = importlib.import_module('app.loaders')
print('module:', m)
print('keys:', sorted([k for k in m.__dict__.keys() if not k.startswith('_')])[:200])
print('has load_elements?', hasattr(m, 'load_elements'))
print('has load_many?', hasattr(m, 'load_many'))
try:
    from app.loaders import load_elements, load_many
    print('direct import OK')
except Exception as e:
    print('direct import failed:', e)
