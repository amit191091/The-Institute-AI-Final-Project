# 🔄 RAG Directory Structure Migration

## 📋 Summary of Changes

This document summarizes the changes made to move all RAG-related files from the project root to a dedicated `RAG/` directory structure.

## 🎯 Problem Solved

Previously, the RAG system was creating files in the project root:
- `data/` folder at project root
- `index/` folder at project root  
- `logs/` folder at project root

This created a cluttered project structure and made it unclear which files belonged to the RAG system.

## ✅ Solution Implemented

### 1. **Updated Configuration** (`RAG/app/config.py`)
```python
@dataclass(frozen=True)
class Settings:
    DATA_DIR: Path = Path("RAG/data")      # Was: Path("RAG/data")
    INDEX_DIR: Path = Path("RAG/index")    # Was: Path("index")
    LOGS_DIR: Path = Path("RAG/logs")      # NEW: Added logs configuration
```

### 2. **Updated All Hardcoded Paths**
- **Logger** (`RAG/app/logger.py`): Now uses `settings.LOGS_DIR`
- **Pipeline** (`RAG/app/pipeline.py`): All log paths updated to use configuration
- **Indexing** (`RAG/app/indexing.py`): Uses `settings.INDEX_DIR` when no env var set
- **Loaders** (`RAG/app/loaders.py`): Updated element dump paths
- **Gradio Apps**: All UI components updated to use configured paths
- **Prompt Optimizer**: Updated evaluation file paths

### 3. **New Directory Structure**
```
Final_project/
├── RAG/
│   ├── data/           # All data files (PDFs, DOCXs, etc.)
│   ├── index/          # Vector database and indexes
│   ├── logs/           # All log files and evaluation results
│   └── app/            # RAG application code
├── Pictures and Vibrations database/  # Picture analysis
└── [other project files]
```

## 🚀 How to Use

### Option 1: Automatic Migration (Recommended)
Run the migration script to move existing files:
```bash
python migrate_paths.py
```

### Option 2: Manual Migration
1. Create the new directories:
   ```bash
   mkdir -p RAG/data RAG/index RAG/logs
   ```

2. Move existing files:
   ```bash
   # Move data files
   mv "Gear wear Failure.pdf" RAG/data/
   mv "Gear wear Failure.docx" RAG/data/
   mv "Database.docx" RAG/data/
   
   # Move index files
   mv index/* RAG/index/
   rmdir index
   
   # Move log files
   mv logs/* RAG/logs/
   rmdir logs
   ```

## 📁 File Locations After Migration

### Data Files
- **Before**: `Final_project/Gear wear Failure.pdf`
- **After**: `Final_project/RAG/data/Gear wear Failure.pdf`

### Index Files
- **Before**: `Final_project/index/`
- **After**: `Final_project/RAG/index/`

### Log Files
- **Before**: `Final_project/logs/`
- **After**: `Final_project/RAG/logs/`

## 🔧 Configuration Options

The system still supports environment variables for custom paths:
- `RAG_CHROMA_DIR`: Custom Chroma database directory
- `RAG_CHROMA_COLLECTION`: Custom collection name
- `RAG_DUMP_ELEMENTS`: Enable element dumping
- `RAG_USE_NORMALIZED`: Use normalized data
- `RAG_USE_NORMALIZED_GRAPH`: Use normalized graph
- `RAG_IMPORT_NORMALIZED_GRAPH`: Import to Neo4j

## ✅ Benefits

1. **Cleaner Project Structure**: RAG files are now contained within the RAG directory
2. **Better Organization**: Clear separation between RAG and other project components
3. **Easier Maintenance**: All RAG-related files in one place
4. **Consistent Configuration**: All paths managed through the settings class
5. **Backward Compatibility**: Environment variables still work for custom configurations

## 🧪 Testing

After migration, test the system:
```bash
# Run the RAG system
python Main.py

# Check that files are created in the correct locations
ls -la RAG/data/
ls -la RAG/index/
ls -la RAG/logs/
```

## 🔄 Rollback (If Needed)

If you need to rollback, you can:
1. Revert the configuration changes in `RAG/app/config.py`
2. Revert the path changes in the affected files
3. Move files back to the project root

## 📝 Notes

- The system maintains backward compatibility with existing environment variables
- Legacy support for root PDF files is preserved
- All existing functionality remains unchanged
- The migration is non-destructive and can be easily reversed
