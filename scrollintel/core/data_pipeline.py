"""
Real-time Data Validation, Cleaning, and Enrichment Pipeline.
Enterprise-grade data processing with streaming capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from abc import ABC, abstractmethod
import hashlib
import statistics
from collections import defaultdict, deque

from ..core.data_connector import DataRecord

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Data pipeline processing stages"""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    CLEANING = "cleaning"
    ENRICHMENT = "enrichment"
    TRANSFORMATION = "transformation"
    QUALITY_CHECK = "quality_check"
    OUTPUT = "output"


class DataQualityLevel(Enum):
    """Data quality assessment levels"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class ProcessingMode(Enum):
    """Data processing modes"""
    BATCH = "batch"
    STREAMING = "streaming"
    MICRO_BATCH = "micro_batch"
    REAL_TIME = "real_time"


@dataclass
class QualityMetrics:
    """Data quality metrics"""
    completeness: float = 0.0  # Percentage of non-null values
    accuracy: float = 0.0      # Percentage of accurate values
    consistency: float = 0.0   # Percentage of consistent values
    validity: float = 0.0      # Percentage of valid format values
    uniqueness: float = 0.0    # Percentage of unique values
    timeliness: float = 0.0    # Percentage of timely data
    overall_score: float = 0.0 # Overall quality score


@dataclass
class ProcessingStats:
    """Pipeline processing statistics"""
    records_processed: int = 0
    records_passed: int = 0
    records_failed: int = 0
    records_enriched: int = 0
    processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)


class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.stats = ProcessingStats()
        self.enabled = True
    
    @abstractmethod
    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        """Process a single data record"""
        pass
    
    async def process_batch(self, records: List[DataRecord]) -> List[DataRecord]:
        """Process a batch of records"""
        processed_records = []
        for record in records:
            if self.enabled:
                processed_record = await self.process(record)
                if processed_record:
                    processed_records.append(processed_record)
            else:
                processed_records.append(record)
        return processed_records
    
    def get_stats(self) -> ProcessingStats:
        """Get processing statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = ProcessingStats()


class SchemaValidator(DataProcessor):
    """Schema validation processor"""
    
    def __init__(self, schema: Dict[str, Any], config: Dict[str, Any] = None):
        super().__init__("schema_validator", config)
        self.schema = schema
        self.required_fields = schema.get('required', [])
        self.field_types = schema.get('properties', {})
        self.validation_errors = deque(maxlen=1000)  # Keep last 1000 errors
    
    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        """Validate record against schema"""
        start_time = datetime.utcnow()
        self.stats.records_processed += 1
        
        try:
            data = record.data
            errors = []
            
            # Check required fields
            for field in self.required_fields:
                if field not in data or data[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Check field types and constraints
            for field_name, field_config in self.field_types.items():
                if field_name in data and data[field_name] is not None:
                    value = data[field_name]
                    field_type = field_config.get('type')
                    
                    # Type validation
                    if field_type == 'string' and not isinstance(value, str):
                        errors.append(f"Field {field_name} must be string, got {type(value).__name__}")
                    elif field_type == 'number' and not isinstance(value, (int, float)):
                        errors.append(f"Field {field_name} must be number, got {type(value).__name__}")
                    elif field_type == 'integer' and not isinstance(value, int):
                        errors.append(f"Field {field_name} must be integer, got {type(value).__name__}")
                    elif field_type == 'boolean' and not isinstance(value, bool):
                        errors.append(f"Field {field_name} must be boolean, got {type(value).__name__}")
                    
                    # Additional constraints
                    if isinstance(value, str):
                        min_length = field_config.get('minLength')
                        max_length = field_config.get('maxLength')
                        pattern = field_config.get('pattern')
                        
                        if min_length and len(value) < min_length:
                            errors.append(f"Field {field_name} too short (min: {min_length})")
                        if max_length and len(value) > max_length:
                            errors.append(f"Field {field_name} too long (max: {max_length})")
                        if pattern and not re.match(pattern, value):
                            errors.append(f"Field {field_name} doesn't match pattern: {pattern}")
                    
                    elif isinstance(value, (int, float)):
                        minimum = field_config.get('minimum')
                        maximum = field_config.get('maximum')
                        
                        if minimum is not None and value < minimum:
                            errors.append(f"Field {field_name} below minimum: {minimum}")
                        if maximum is not None and value > maximum:
                            errors.append(f"Field {field_name} above maximum: {maximum}")
            
            if errors:
                self.validation_errors.append({
                    'record_id': record.record_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'errors': errors
                })
                self.stats.records_failed += 1
                
                # Decide whether to reject or pass with warnings
                if self.config.get('strict_mode', True):
                    return None
                else:
                    # Add validation warnings to metadata
                    record.metadata['validation_warnings'] = errors
            
            self.stats.records_passed += 1
            
            # Update processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats.processing_time_ms += processing_time
            
            return record
            
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            self.stats.records_failed += 1
            return None


class DataCleaner(DataProcessor):
    """Data cleaning and normalization processor"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("data_cleaner", config)
        self.cleaning_rules = config.get('cleaning_rules', {}) if config else {}
        self.null_strategies = config.get('null_strategies', {}) if config else {}
        self.outlier_detection = config.get('outlier_detection', {}) if config else {}
    
    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        """Clean and normalize record data"""
        start_time = datetime.utcnow()
        self.stats.records_processed += 1
        
        try:
            cleaned_data = record.data.copy()
            cleaning_applied = []
            
            # Apply field-specific cleaning rules
            for field_name, value in cleaned_data.items():
                if field_name in self.cleaning_rules:
                    rules = self.cleaning_rules[field_name]
                    
                    # String cleaning
                    if isinstance(value, str):
                        if rules.get('trim', True):
                            value = value.strip()
                        if rules.get('lowercase'):
                            value = value.lower()
                        if rules.get('uppercase'):
                            value = value.upper()
                        if rules.get('remove_special_chars'):
                            value = re.sub(r'[^\w\s]', '', value)
                        if rules.get('normalize_whitespace'):
                            value = re.sub(r'\s+', ' ', value)
                        
                        # Custom regex replacements
                        replacements = rules.get('replacements', [])
                        for replacement in replacements:
                            pattern = replacement.get('pattern')
                            replacement_text = replacement.get('replacement', '')
                            if pattern:
                                value = re.sub(pattern, replacement_text, value)
                    
                    # Numeric cleaning
                    elif isinstance(value, (int, float)):
                        # Handle outliers
                        if field_name in self.outlier_detection:
                            outlier_config = self.outlier_detection[field_name]
                            min_val = outlier_config.get('min')
                            max_val = outlier_config.get('max')
                            
                            if min_val is not None and value < min_val:
                                if outlier_config.get('action') == 'cap':
                                    value = min_val
                                    cleaning_applied.append(f"Capped {field_name} to minimum")
                                elif outlier_config.get('action') == 'null':
                                    value = None
                                    cleaning_applied.append(f"Nullified outlier {field_name}")
                            
                            if max_val is not None and value > max_val:
                                if outlier_config.get('action') == 'cap':
                                    value = max_val
                                    cleaning_applied.append(f"Capped {field_name} to maximum")
                                elif outlier_config.get('action') == 'null':
                                    value = None
                                    cleaning_applied.append(f"Nullified outlier {field_name}")
                    
                    cleaned_data[field_name] = value
            
            # Handle null values
            for field_name, strategy in self.null_strategies.items():
                if field_name in cleaned_data and cleaned_data[field_name] is None:
                    if strategy == 'remove_record':
                        self.stats.records_failed += 1
                        return None
                    elif strategy == 'default_value':
                        default_val = self.null_strategies.get(f'{field_name}_default')
                        cleaned_data[field_name] = default_val
                        cleaning_applied.append(f"Applied default value to {field_name}")
                    elif strategy == 'interpolate':
                        # Simple interpolation (in production, use more sophisticated methods)
                        cleaned_data[field_name] = 0
                        cleaning_applied.append(f"Interpolated {field_name}")
            
            # Create cleaned record
            cleaned_record = DataRecord(
                source_id=record.source_id,
                record_id=record.record_id,
                data=cleaned_data,
                timestamp=record.timestamp,
                metadata={
                    **record.metadata,
                    'cleaning_applied': cleaning_applied,
                    'cleaned_at': datetime.utcnow().isoformat()
                }
            )
            
            self.stats.records_passed += 1
            
            # Update processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats.processing_time_ms += processing_time
            
            return cleaned_record
            
        except Exception as e:
            logger.error(f"Data cleaning error: {e}")
            self.stats.records_failed += 1
            return None


class DataEnricher(DataProcessor):
    """Data enrichment processor with external lookups and calculations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("data_enricher", config)
        self.enrichment_rules = config.get('enrichment_rules', []) if config else []
        self.lookup_cache = {}
        self.cache_ttl = config.get('cache_ttl', 3600) if config else 3600  # 1 hour
        self.external_apis = config.get('external_apis', {}) if config else {}
    
    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        """Enrich record with additional data"""
        start_time = datetime.utcnow()
        self.stats.records_processed += 1
        
        try:
            enriched_data = record.data.copy()
            enrichments_applied = []
            
            for rule in self.enrichment_rules:
                enrichment_type = rule.get('type')
                source_field = rule.get('source_field')
                target_field = rule.get('target_field')
                
                if source_field not in enriched_data:
                    continue
                
                source_value = enriched_data[source_field]
                
                if enrichment_type == 'lookup':
                    # Lookup enrichment
                    lookup_table = rule.get('lookup_table', {})
                    enriched_value = lookup_table.get(str(source_value))
                    if enriched_value:
                        enriched_data[target_field] = enriched_value
                        enrichments_applied.append(f"Lookup: {target_field}")
                
                elif enrichment_type == 'calculation':
                    # Calculation-based enrichment
                    formula = rule.get('formula')
                    if formula:
                        try:
                            # Simple calculation engine
                            enriched_value = await self._calculate_value(enriched_data, formula)
                            enriched_data[target_field] = enriched_value
                            enrichments_applied.append(f"Calculation: {target_field}")
                        except Exception as e:
                            logger.warning(f"Calculation failed for {target_field}: {e}")
                
                elif enrichment_type == 'geolocation':
                    # Geolocation enrichment
                    if isinstance(source_value, str) and source_value:
                        geo_data = await self._get_geolocation(source_value)
                        if geo_data:
                            enriched_data[target_field] = geo_data
                            enrichments_applied.append(f"Geolocation: {target_field}")
                
                elif enrichment_type == 'external_api':
                    # External API enrichment
                    api_config = rule.get('api_config', {})
                    if api_config:
                        api_data = await self._call_external_api(source_value, api_config)
                        if api_data:
                            enriched_data[target_field] = api_data
                            enrichments_applied.append(f"External API: {target_field}")
                
                elif enrichment_type == 'ml_inference':
                    # ML model inference
                    model_config = rule.get('model_config', {})
                    if model_config:
                        prediction = await self._ml_inference(enriched_data, model_config)
                        if prediction is not None:
                            enriched_data[target_field] = prediction
                            enrichments_applied.append(f"ML Inference: {target_field}")
                
                elif enrichment_type == 'data_quality_score':
                    # Calculate data quality score
                    quality_score = await self._calculate_quality_score(enriched_data)
                    enriched_data[target_field] = quality_score
                    enrichments_applied.append(f"Quality Score: {target_field}")
            
            # Add standard enrichments
            enriched_data.update({
                'processing_timestamp': datetime.utcnow().isoformat(),
                'record_hash': hashlib.md5(json.dumps(enriched_data, sort_keys=True).encode()).hexdigest(),
                'enrichment_version': '1.0'
            })
            
            # Create enriched record
            enriched_record = DataRecord(
                source_id=record.source_id,
                record_id=record.record_id,
                data=enriched_data,
                timestamp=record.timestamp,
                metadata={
                    **record.metadata,
                    'enrichments_applied': enrichments_applied,
                    'enriched_at': datetime.utcnow().isoformat(),
                    'enrichment_count': len(enrichments_applied)
                }
            )
            
            self.stats.records_passed += 1
            self.stats.records_enriched += 1
            
            # Update processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats.processing_time_ms += processing_time
            
            return enriched_record
            
        except Exception as e:
            logger.error(f"Data enrichment error: {e}")
            self.stats.records_failed += 1
            return record  # Return original record on enrichment failure
    
    async def _calculate_value(self, data: Dict[str, Any], formula: str) -> Any:
        """Calculate value using formula"""
        # Simple formula evaluation (in production, use a proper expression evaluator)
        try:
            # Replace field references
            for field, value in data.items():
                if isinstance(value, (int, float)):
                    formula = formula.replace(f'{{{field}}}', str(value))
            
            # Evaluate mathematical expressions only
            if all(c in '0123456789+-*/.() ' for c in formula):
                return eval(formula)
            else:
                return None
        except Exception:
            return None
    
    async def _get_geolocation(self, address: str) -> Optional[Dict[str, Any]]:
        """Get geolocation data for address"""
        # Simulate geolocation lookup
        cache_key = f"geo:{address}"
        
        if cache_key in self.lookup_cache:
            cached_data, timestamp = self.lookup_cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl:
                return cached_data
        
        # Mock geolocation data
        geo_data = {
            'latitude': 40.7128 + hash(address) % 100 * 0.01,
            'longitude': -74.0060 + hash(address) % 100 * 0.01,
            'city': 'New York',
            'state': 'NY',
            'country': 'US',
            'postal_code': f'{10000 + hash(address) % 90000:05d}'
        }
        
        self.lookup_cache[cache_key] = (geo_data, datetime.utcnow())
        return geo_data
    
    async def _call_external_api(self, value: Any, api_config: Dict[str, Any]) -> Optional[Any]:
        """Call external API for enrichment"""
        # Simulate external API call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        api_name = api_config.get('name', 'unknown')
        return f"{api_name}_enriched_{value}"
    
    async def _ml_inference(self, data: Dict[str, Any], model_config: Dict[str, Any]) -> Optional[Any]:
        """Perform ML model inference"""
        # Simulate ML inference
        await asyncio.sleep(0.05)  # Simulate inference time
        
        model_name = model_config.get('model_name', 'default')
        features = [data.get(field, 0) for field in model_config.get('features', [])]
        
        # Simple mock prediction
        if features:
            prediction = sum(hash(str(f)) % 100 for f in features) / len(features) / 100
            return round(prediction, 3)
        
        return None
    
    async def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate data quality score for the record"""
        total_fields = len(data)
        if total_fields == 0:
            return 0.0
        
        quality_factors = []
        
        # Completeness: percentage of non-null fields
        non_null_fields = sum(1 for v in data.values() if v is not None and v != '')
        completeness = non_null_fields / total_fields
        quality_factors.append(completeness)
        
        # Consistency: check for consistent data types and formats
        consistency_score = 1.0  # Start with perfect score
        for field, value in data.items():
            if isinstance(value, str):
                # Check for consistent string formatting
                if value != value.strip():
                    consistency_score -= 0.1
                if re.search(r'\s{2,}', value):  # Multiple spaces
                    consistency_score -= 0.05
        
        quality_factors.append(max(0, consistency_score))
        
        # Validity: check for valid formats (simplified)
        validity_score = 1.0
        for field, value in data.items():
            if 'email' in field.lower() and isinstance(value, str):
                if not re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
                    validity_score -= 0.2
            elif 'phone' in field.lower() and isinstance(value, str):
                if not re.match(r'^\+?[\d\s\-\(\)]+$', value):
                    validity_score -= 0.2
        
        quality_factors.append(max(0, validity_score))
        
        # Overall quality score
        return round(statistics.mean(quality_factors), 3)


class QualityAssessment(DataProcessor):
    """Data quality assessment processor"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("quality_assessment", config)
        self.quality_thresholds = config.get('quality_thresholds', {}) if config else {}
        self.quality_history = deque(maxlen=10000)  # Keep last 10k quality scores
    
    async def process(self, record: DataRecord) -> Optional[DataRecord]:
        """Assess and score data quality"""
        start_time = datetime.utcnow()
        self.stats.records_processed += 1
        
        try:
            quality_metrics = await self._assess_quality(record.data)
            
            # Add quality assessment to record metadata
            record.metadata.update({
                'quality_metrics': quality_metrics.__dict__,
                'quality_level': self._get_quality_level(quality_metrics.overall_score),
                'quality_assessed_at': datetime.utcnow().isoformat()
            })
            
            # Store quality history
            self.quality_history.append({
                'record_id': record.record_id,
                'timestamp': datetime.utcnow(),
                'quality_score': quality_metrics.overall_score
            })
            
            # Check if record meets quality thresholds
            min_quality = self.quality_thresholds.get('minimum_score', 0.5)
            if quality_metrics.overall_score < min_quality:
                logger.warning(f"Record {record.record_id} below quality threshold: {quality_metrics.overall_score}")
                
                if self.config.get('reject_low_quality', False):
                    self.stats.records_failed += 1
                    return None
            
            self.stats.records_passed += 1
            
            # Update quality metrics in stats
            self.stats.quality_metrics = quality_metrics
            
            # Update processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats.processing_time_ms += processing_time
            
            return record
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            self.stats.records_failed += 1
            return record
    
    async def _assess_quality(self, data: Dict[str, Any]) -> QualityMetrics:
        """Assess data quality across multiple dimensions"""
        metrics = QualityMetrics()
        
        total_fields = len(data)
        if total_fields == 0:
            return metrics
        
        # Completeness
        non_null_count = sum(1 for v in data.values() if v is not None and v != '')
        metrics.completeness = non_null_count / total_fields
        
        # Accuracy (simplified - in production, use reference data)
        accurate_count = 0
        for field, value in data.items():
            if value is not None:
                # Simple accuracy checks
                if isinstance(value, str) and len(value.strip()) > 0:
                    accurate_count += 1
                elif isinstance(value, (int, float)) and not (value != value):  # Not NaN
                    accurate_count += 1
                elif isinstance(value, bool):
                    accurate_count += 1
        
        metrics.accuracy = accurate_count / total_fields if total_fields > 0 else 0
        
        # Consistency
        consistency_issues = 0
        for field, value in data.items():
            if isinstance(value, str):
                # Check for inconsistent formatting
                if value != value.strip():
                    consistency_issues += 1
                if re.search(r'\s{2,}', value):
                    consistency_issues += 1
        
        metrics.consistency = max(0, 1 - (consistency_issues / total_fields))
        
        # Validity
        valid_count = 0
        for field, value in data.items():
            if value is None:
                continue
            
            # Field-specific validation
            if 'email' in field.lower() and isinstance(value, str):
                if re.match(r'^[^@]+@[^@]+\.[^@]+$', value):
                    valid_count += 1
            elif 'phone' in field.lower() and isinstance(value, str):
                if re.match(r'^\+?[\d\s\-\(\)]+$', value):
                    valid_count += 1
            elif 'date' in field.lower() and isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace('Z', '+00:00'))
                    valid_count += 1
                except ValueError:
                    pass
            else:
                # General validity (non-empty, reasonable length)
                if isinstance(value, str) and 0 < len(value) < 1000:
                    valid_count += 1
                elif isinstance(value, (int, float)) and -1e10 < value < 1e10:
                    valid_count += 1
                elif isinstance(value, bool):
                    valid_count += 1
        
        metrics.validity = valid_count / non_null_count if non_null_count > 0 else 0
        
        # Uniqueness (simplified - would need historical data in production)
        unique_values = len(set(str(v) for v in data.values() if v is not None))
        metrics.uniqueness = unique_values / non_null_count if non_null_count > 0 else 0
        
        # Timeliness (check if timestamps are recent)
        timeliness_score = 1.0
        for field, value in data.items():
            if 'timestamp' in field.lower() or 'date' in field.lower():
                if isinstance(value, str):
                    try:
                        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        age_hours = (datetime.utcnow() - dt.replace(tzinfo=None)).total_seconds() / 3600
                        if age_hours > 24:  # Older than 24 hours
                            timeliness_score *= 0.8
                    except ValueError:
                        timeliness_score *= 0.9
        
        metrics.timeliness = timeliness_score
        
        # Overall score (weighted average)
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.15,
            'validity': 0.20,
            'uniqueness': 0.05,
            'timeliness': 0.10
        }
        
        metrics.overall_score = (
            metrics.completeness * weights['completeness'] +
            metrics.accuracy * weights['accuracy'] +
            metrics.consistency * weights['consistency'] +
            metrics.validity * weights['validity'] +
            metrics.uniqueness * weights['uniqueness'] +
            metrics.timeliness * weights['timeliness']
        )
        
        return metrics
    
    def _get_quality_level(self, score: float) -> DataQualityLevel:
        """Get quality level based on score"""
        if score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif score >= 0.7:
            return DataQualityLevel.GOOD
        elif score >= 0.5:
            return DataQualityLevel.FAIR
        else:
            return DataQualityLevel.POOR
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Get quality trends over time"""
        if not self.quality_history:
            return {}
        
        recent_scores = [entry['quality_score'] for entry in list(self.quality_history)[-100:]]
        
        return {
            'average_quality': statistics.mean(recent_scores),
            'quality_trend': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable',
            'min_quality': min(recent_scores),
            'max_quality': max(recent_scores),
            'quality_variance': statistics.variance(recent_scores) if len(recent_scores) > 1 else 0
        }


class DataPipeline:
    """Enterprise data processing pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.processors: List[DataProcessor] = []
        self.processing_mode = ProcessingMode(config.get('processing_mode', 'streaming'))
        self.batch_size = config.get('batch_size', 1000)
        self.buffer = deque()
        self.pipeline_stats = ProcessingStats()
        self.error_handlers: List[Callable] = []
        self.monitoring_enabled = config.get('monitoring_enabled', True)
    
    def add_processor(self, processor: DataProcessor):
        """Add a processor to the pipeline"""
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.name}")
    
    def add_error_handler(self, handler: Callable):
        """Add an error handler"""
        self.error_handlers.append(handler)
    
    async def process_record(self, record: DataRecord) -> Optional[DataRecord]:
        """Process a single record through the pipeline"""
        start_time = datetime.utcnow()
        current_record = record
        
        try:
            for processor in self.processors:
                if current_record is None:
                    break
                
                current_record = await processor.process(current_record)
                
                if current_record is None:
                    logger.debug(f"Record {record.record_id} filtered out by {processor.name}")
                    break
            
            # Update pipeline stats
            self.pipeline_stats.records_processed += 1
            if current_record is not None:
                self.pipeline_stats.records_passed += 1
            else:
                self.pipeline_stats.records_failed += 1
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.pipeline_stats.processing_time_ms += processing_time
            
            return current_record
            
        except Exception as e:
            logger.error(f"Pipeline processing error for record {record.record_id}: {e}")
            
            # Call error handlers
            for handler in self.error_handlers:
                try:
                    await handler(record, e)
                except Exception as handler_error:
                    logger.error(f"Error handler failed: {handler_error}")
            
            self.pipeline_stats.records_failed += 1
            return None
    
    async def process_batch(self, records: List[DataRecord]) -> List[DataRecord]:
        """Process a batch of records"""
        processed_records = []
        
        if self.processing_mode == ProcessingMode.BATCH:
            # Process entire batch through each processor
            current_batch = records
            for processor in self.processors:
                current_batch = await processor.process_batch(current_batch)
            processed_records = current_batch
        else:
            # Process records individually
            for record in records:
                processed_record = await self.process_record(record)
                if processed_record:
                    processed_records.append(processed_record)
        
        return processed_records
    
    async def process_stream(self, record_stream: AsyncGenerator[DataRecord, None]) -> AsyncGenerator[DataRecord, None]:
        """Process a stream of records"""
        if self.processing_mode == ProcessingMode.REAL_TIME:
            # Process records as they arrive
            async for record in record_stream:
                processed_record = await self.process_record(record)
                if processed_record:
                    yield processed_record
        else:
            # Buffer records and process in micro-batches
            async for record in record_stream:
                self.buffer.append(record)
                
                if len(self.buffer) >= self.batch_size:
                    batch = [self.buffer.popleft() for _ in range(self.batch_size)]
                    processed_batch = await self.process_batch(batch)
                    for processed_record in processed_batch:
                        yield processed_record
            
            # Process remaining records in buffer
            if self.buffer:
                remaining_batch = list(self.buffer)
                self.buffer.clear()
                processed_batch = await self.process_batch(remaining_batch)
                for processed_record in processed_batch:
                    yield processed_record
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        processor_stats = {}
        for processor in self.processors:
            processor_stats[processor.name] = processor.get_stats().__dict__
        
        # Calculate throughput
        if self.pipeline_stats.processing_time_ms > 0:
            self.pipeline_stats.throughput_per_second = (
                self.pipeline_stats.records_processed / 
                (self.pipeline_stats.processing_time_ms / 1000)
            )
        
        # Calculate error rate
        if self.pipeline_stats.records_processed > 0:
            self.pipeline_stats.error_rate = (
                self.pipeline_stats.records_failed / 
                self.pipeline_stats.records_processed
            )
        
        return {
            'pipeline_stats': self.pipeline_stats.__dict__,
            'processor_stats': processor_stats,
            'processing_mode': self.processing_mode.value,
            'active_processors': len(self.processors)
        }
    
    def reset_stats(self):
        """Reset all pipeline statistics"""
        self.pipeline_stats = ProcessingStats()
        for processor in self.processors:
            processor.reset_stats()


# Factory function to create a standard enterprise pipeline
def create_enterprise_pipeline(schema: Dict[str, Any], config: Dict[str, Any] = None) -> DataPipeline:
    """Create a standard enterprise data pipeline"""
    pipeline_config = config or {}
    
    pipeline = DataPipeline(pipeline_config)
    
    # Add standard processors
    pipeline.add_processor(SchemaValidator(schema, pipeline_config.get('validation', {})))
    pipeline.add_processor(DataCleaner(pipeline_config.get('cleaning', {})))
    pipeline.add_processor(DataEnricher(pipeline_config.get('enrichment', {})))
    pipeline.add_processor(QualityAssessment(pipeline_config.get('quality', {})))
    
    return pipeline