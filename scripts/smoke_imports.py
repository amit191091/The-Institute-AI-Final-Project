import importlib, os, sys
print('PY', sys.version)
os.environ.pop('OPENAI_API_KEY', None)
mods=['rag_app.indexing','rag_app.retrieve','rag_app.agents']
for m in mods:
    try:
        importlib.import_module(m)
        print('IMPORTED', m)
    except Exception as e:
        print('ERR', m, e)
