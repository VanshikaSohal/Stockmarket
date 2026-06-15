# Archive — Legacy Rule-Based Modules

This directory contains the original rule-based matching and simulation
modules that were the primary inference engine before the NLP semantic
mapping system was introduced.

**Nothing here is deleted.** Every file is preserved intact with its
original internal code so it can be referenced, tested, or reinstated.

---

## Why these files are here

| File | Rule-based pattern | Reason archived |
|------|--------------------|-----------------|
| `api/search.py` | Hardcoded mock results, static confidence values | Replaced by `/api/semantic-search` (NLP) |
| `api/terminology.py` | 3-entry hardcoded dict with manual confidence numbers | Not scalable; NLP covers full corpus |
| `api/outbreak.py` | Hardcoded risk dicts, manual growth-rate formulas | Domain-specific simulation, not mapping |
| `src/services/ai_assistant.py` | `AYUSH_AI_DATABASE` dict + substring `in` matching | Duplicate of master_portal logic; NLP supersedes |
| `src/services/multilingual_support.py` | 4-term static `MULTILINGUAL_TRANSLATIONS` dict | Replaced by root `multilingual_support.py` (22 languages) |
| `src/services/research_analytics.py` | Hardcoded sample research data, `random.randint` analytics | Simulation/demo data only |
| `src/services/telemedicine_integration.py` | Hardcoded practitioner list, mock ABHA verification | Demo stub, no live integration |
| `pipeline/phase3_ml_ai/ai_engine.py` | Hardcoded symptom dicts + `np.random.normal` fake ML | No trained model; confidence scores are manual constants |
| `pipeline/phase1_data_foundation/namaste_processor.py` | Hardcoded column-name maps, static coverage dict | Excel files (Ayurveda.xls etc.) not present in repo |

---

## How the archive is used in production

`api_new/complete_portal.py` imports `NAMASTEProcessor` and `AdvancedAIEngine`
from this archive as an **optional fallback** only — they are never used
for primary inference. The import block is wrapped in `try/except` and
both symbols default to `None` if unavailable:

```python
# complete_portal.py
try:
    sys.path.insert(0, str(project_root / "archive" / "pipeline" / "phase1_data_foundation"))
    sys.path.insert(0, str(project_root / "pipeline" / "phase3_ml_ai"))
    from namaste_processor import NAMASTEProcessor   # archive copy
    from ai_engine import AdvancedAIEngine           # archive copy (simulated)
except ImportError:
    NAMASTEProcessor = None
    AdvancedAIEngine = None
```

All other archived files are **not imported by any active module**.

---

## Primary inference pipeline (what replaced these)

```
Query
  └─> nlp/semantic_mapper.py          (cosine similarity, all-MiniLM-L6-v2)
        └─> nlp/embedding_store.py    (34,662 ICD-11 embeddings, .npy)
        └─> nlp/ontology_fallback.py  (CSV lookup when score < 0.45)
```

Build embeddings once before serving:
```bash
pip install -r nlp/requirements_nlp.txt
python -m nlp.embedding_store
```

Evaluate:
```bash
python -m nlp.evaluate --sample 200 --qualitative
```
