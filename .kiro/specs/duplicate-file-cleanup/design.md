# Design Document

## Overview

The duplicate file cleanup system will systematically identify, analyze, and consolidate duplicate files in the ScrollIntel codebase. The system will use a multi-phase approach: discovery, analysis, planning, execution, and validation. The design prioritizes safety and maintains functionality throughout the consolidation process.

## Architecture

### Core Components

```
duplicate-cleanup/
├── analyzer/
│   ├── file_scanner.py          # Discovers duplicate files
│   ├── content_analyzer.py      # Analyzes file content and differences
│   ├── dependency_mapper.py     # Maps file dependencies and imports
│   └── conflict_detector.py     # Identifies potential consolidation conflicts
├── consolidator/
│   ├── strategy_selector.py     # Selects appropriate consolidation strategy
│   ├── file_merger.py          # Merges compatible files
│   ├── import_updater.py       # Updates import statements
│   └── reference_updater.py    # Updates file references
├── validators/
│   ├── syntax_validator.py     # Validates Python syntax
│   ├── import_validator.py     # Validates import resolution
│   ├── test_runner.py          # Runs tests to validate functionality
│   └── api_validator.py        # Validates API endpoints
└── reporters/
    ├── analysis_reporter.py    # Generates analysis reports
    ├── consolidation_reporter.py # Reports consolidation results
    └── validation_reporter.py  # Reports validation results
```

## Components and Interfaces

### 1. File Scanner

**Purpose:** Discovers all duplicate files in the codebase

**Interface:**
```python
class FileScanner:
    def scan_directory(self, root_path: str) -> List[DuplicateGroup]
    def categorize_duplicates(self, duplicates: List[DuplicateGroup]) -> Dict[str, List[DuplicateGroup]]
    def calculate_file_hash(self, file_path: str) -> str
```

**Key Features:**
- Scans entire codebase recursively
- Groups files by name and content hash
- Categorizes duplicates by file type (config, routes, models, etc.)
- Excludes common files like `__init__.py` from consolidation

### 2. Content Analyzer

**Purpose:** Analyzes file content to determine consolidation strategy

**Interface:**
```python
class ContentAnalyzer:
    def analyze_differences(self, file_group: DuplicateGroup) -> AnalysisResult
    def detect_identical_content(self, files: List[str]) -> bool
    def identify_merge_conflicts(self, files: List[str]) -> List[Conflict]
    def suggest_consolidation_strategy(self, analysis: AnalysisResult) -> ConsolidationStrategy
```

**Analysis Types:**
- **Identical Files:** Same content, can be safely merged
- **Similar Files:** Similar structure, may need manual review
- **Conflicting Files:** Different implementations, need careful handling

### 3. Dependency Mapper

**Purpose:** Maps all file dependencies and import relationships

**Interface:**
```python
class DependencyMapper:
    def map_imports(self, file_path: str) -> List[ImportStatement]
    def find_references(self, file_path: str) -> List[Reference]
    def build_dependency_graph(self, files: List[str]) -> DependencyGraph
    def detect_circular_dependencies(self, graph: DependencyGraph) -> List[CircularDependency]
```

**Capabilities:**
- Parses Python AST to extract imports
- Finds string references to file paths
- Builds comprehensive dependency graph
- Detects potential circular dependencies

### 4. Strategy Selector

**Purpose:** Selects appropriate consolidation strategy for each duplicate group

**Strategies:**
1. **Direct Merge:** For identical files
2. **Hierarchical Merge:** For config files with inheritance
3. **Namespace Merge:** For API routes with different namespaces
4. **Adapter Pattern:** For conflicting implementations
5. **Manual Review:** For complex conflicts

**Interface:**
```python
class StrategySelector:
    def select_strategy(self, duplicate_group: DuplicateGroup, analysis: AnalysisResult) -> Strategy
    def validate_strategy(self, strategy: Strategy) -> ValidationResult
```

### 5. File Merger

**Purpose:** Executes file consolidation based on selected strategy

**Interface:**
```python
class FileMerger:
    def merge_identical_files(self, files: List[str], target: str) -> MergeResult
    def merge_config_files(self, files: List[str], target: str) -> MergeResult
    def merge_route_files(self, files: List[str], target: str) -> MergeResult
    def create_backup(self, files: List[str]) -> str
```

**Merge Strategies:**
- **Config Files:** Create hierarchical configuration with environment overrides
- **Route Files:** Merge compatible routes, namespace conflicting ones
- **Database Files:** Create unified database abstraction layer
- **Test Files:** Merge test cases, remove redundancy

### 6. Import Updater

**Purpose:** Updates all import statements after file consolidation

**Interface:**
```python
class ImportUpdater:
    def find_import_statements(self, file_path: str) -> List[ImportStatement]
    def update_imports(self, old_path: str, new_path: str) -> UpdateResult
    def validate_imports(self, file_path: str) -> ValidationResult
```

**Update Types:**
- Absolute imports: `from scrollintel.core.config import Config`
- Relative imports: `from ..core.config import Config`
- Dynamic imports: `importlib.import_module()`
- String references in configuration files

## Data Models

### DuplicateGroup
```python
@dataclass
class DuplicateGroup:
    name: str
    files: List[FilePath]
    content_hash: str
    file_type: FileType
    analysis_result: Optional[AnalysisResult] = None
```

### AnalysisResult
```python
@dataclass
class AnalysisResult:
    identical_content: bool
    similarity_score: float
    conflicts: List[Conflict]
    merge_complexity: MergeComplexity
    recommended_strategy: ConsolidationStrategy
```

### ConsolidationPlan
```python
@dataclass
class ConsolidationPlan:
    duplicate_groups: List[DuplicateGroup]
    consolidation_order: List[str]
    import_updates: List[ImportUpdate]
    validation_steps: List[ValidationStep]
    rollback_plan: RollbackPlan
```

## Error Handling

### Error Categories
1. **File System Errors:** Permission issues, missing files
2. **Syntax Errors:** Invalid Python syntax after consolidation
3. **Import Errors:** Broken import statements
4. **Test Failures:** Functionality broken after consolidation
5. **Circular Dependencies:** Import cycles created during consolidation

### Error Recovery
- **Automatic Rollback:** Restore original files if validation fails
- **Partial Rollback:** Rollback specific changes that cause issues
- **Manual Intervention:** Flag complex issues for manual resolution
- **Incremental Processing:** Process files in small batches to isolate issues

### Validation Pipeline
```python
def validate_consolidation(plan: ConsolidationPlan) -> ValidationResult:
    # 1. Syntax validation
    syntax_result = validate_python_syntax(plan.affected_files)
    
    # 2. Import validation
    import_result = validate_imports(plan.affected_files)
    
    # 3. Test validation
    test_result = run_test_suite(plan.test_files)
    
    # 4. API validation
    api_result = validate_api_endpoints(plan.api_files)
    
    return ValidationResult(syntax_result, import_result, test_result, api_result)
```

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock file system operations
- Test error handling and edge cases
- Validate consolidation strategies

### Integration Tests
- Test end-to-end consolidation workflow
- Test with real duplicate files from codebase
- Validate import updates work correctly
- Test rollback functionality

### Validation Tests
- Run existing test suite after consolidation
- Validate API endpoints still work
- Check that all imports resolve
- Verify configuration loading works

### Safety Tests
- Test backup and restore functionality
- Validate rollback mechanisms
- Test partial failure scenarios
- Verify no data loss occurs

## Implementation Phases

### Phase 1: Discovery and Analysis
1. Implement file scanner to find duplicates
2. Implement content analyzer to categorize duplicates
3. Build dependency mapper to understand relationships
4. Generate comprehensive analysis report

### Phase 2: Strategy Development
1. Implement strategy selector
2. Develop consolidation strategies for each file type
3. Create consolidation planning system
4. Implement conflict detection

### Phase 3: Safe Consolidation
1. Implement file merger with backup system
2. Implement import updater
3. Implement reference updater
4. Create validation pipeline

### Phase 4: Validation and Reporting
1. Implement comprehensive validation system
2. Create detailed reporting system
3. Implement rollback mechanisms
4. Add monitoring and logging

## Risk Mitigation

### High-Risk Areas
1. **Configuration Files:** Multiple config.py files with different settings
2. **Database Files:** Different database connection patterns
3. **API Routes:** Potential endpoint conflicts
4. **Import Chains:** Complex dependency relationships

### Mitigation Strategies
1. **Comprehensive Backup:** Backup all files before any changes
2. **Incremental Processing:** Process files in small, manageable batches
3. **Extensive Validation:** Multiple validation layers before committing changes
4. **Rollback Capability:** Ability to quickly restore original state
5. **Manual Review Points:** Flag complex cases for human review

## Success Metrics

### Quantitative Metrics
- Number of duplicate files identified and consolidated
- Reduction in total file count
- Improvement in test execution time
- Reduction in import statement count

### Qualitative Metrics
- Improved code maintainability
- Clearer project structure
- Reduced developer confusion
- Easier onboarding for new developers