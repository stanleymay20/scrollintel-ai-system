import os
"""
Elasticsearch Integration for Data Product Registry

Provides advanced search capabilities including full-text search,
semantic search, and faceted search for data products.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

from scrollintel.models.data_product_models import DataProduct

logger = logging.getLogger(__name__)


class DataProductIndexer:
    """Elasticsearch indexer for data products"""
    
    def __init__(self, elasticsearch_url: str = os.getenv("API_URL", os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))):
        self.es = Elasticsearch([elasticsearch_url])
        self.index_name = "data_products"
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create index with proper mapping if it doesn't exist"""
        
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "name": {
                            "type": "text",
                            "analyzer": "standard",
                            "fields": {
                                "keyword": {"type": "keyword"}
                            }
                        },
                        "description": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "owner": {"type": "keyword"},
                        "access_level": {"type": "keyword"},
                        "verification_status": {"type": "keyword"},
                        "compliance_tags": {"type": "keyword"},
                        "quality_score": {"type": "float"},
                        "bias_score": {"type": "float"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"},
                        "freshness_timestamp": {"type": "date"},
                        "schema_definition": {
                            "type": "object",
                            "enabled": False  # Don't index schema structure
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "tags": {"type": "keyword"},
                                "category": {"type": "keyword"},
                                "domain": {"type": "keyword"}
                            }
                        },
                        "searchable_content": {
                            "type": "text",
                            "analyzer": "standard"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "data_product_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "stop", "snowball"]
                            }
                        }
                    }
                }
            }
            
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created Elasticsearch index: {self.index_name}")
    
    def index_data_product(self, data_product: DataProduct) -> bool:
        """Index a single data product"""
        
        try:
            doc = self._prepare_document(data_product)
            
            response = self.es.index(
                index=self.index_name,
                id=str(data_product.id),
                body=doc
            )
            
            logger.debug(f"Indexed data product {data_product.id}: {response['result']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index data product {data_product.id}: {e}")
            return False
    
    def bulk_index_data_products(self, data_products: List[DataProduct]) -> Tuple[int, int]:
        """Bulk index multiple data products"""
        
        actions = []
        for data_product in data_products:
            doc = self._prepare_document(data_product)
            action = {
                "_index": self.index_name,
                "_id": str(data_product.id),
                "_source": doc
            }
            actions.append(action)
        
        try:
            success_count, failed_items = bulk(self.es, actions)
            logger.info(f"Bulk indexed {success_count} data products, {len(failed_items)} failed")
            return success_count, len(failed_items)
            
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return 0, len(data_products)
    
    def delete_data_product(self, product_id: str) -> bool:
        """Remove data product from index"""
        
        try:
            self.es.delete(index=self.index_name, id=product_id)
            logger.debug(f"Deleted data product {product_id} from index")
            return True
            
        except NotFoundError:
            logger.warning(f"Data product {product_id} not found in index")
            return False
        except Exception as e:
            logger.error(f"Failed to delete data product {product_id}: {e}")
            return False
    
    def search_data_products(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Full-text search with filters"""
        
        search_body = {
            "query": self._build_query(query, filters),
            "from": offset,
            "size": limit,
            "highlight": {
                "fields": {
                    "name": {},
                    "description": {},
                    "searchable_content": {}
                }
            }
        }
        
        # Add sorting
        if sort_by:
            search_body["sort"] = [{sort_by: {"order": sort_order}}]
        else:
            search_body["sort"] = [{"_score": {"order": "desc"}}]
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            
            hits = response["hits"]["hits"]
            total_count = response["hits"]["total"]["value"]
            
            results = []
            for hit in hits:
                result = hit["_source"]
                result["_score"] = hit["_score"]
                if "highlight" in hit:
                    result["_highlight"] = hit["highlight"]
                results.append(result)
            
            return results, total_count
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [], 0
    
    def faceted_search(
        self,
        query: Optional[str] = None,
        facets: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Faceted search with aggregations"""
        
        facets = facets or ["owner", "access_level", "verification_status", "compliance_tags"]
        
        search_body = {
            "query": self._build_query(query, filters),
            "size": limit,
            "aggs": {}
        }
        
        # Add facet aggregations
        for facet in facets:
            search_body["aggs"][f"{facet}_facet"] = {
                "terms": {
                    "field": facet,
                    "size": 20
                }
            }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            
            # Extract results
            hits = [hit["_source"] for hit in response["hits"]["hits"]]
            total_count = response["hits"]["total"]["value"]
            
            # Extract facets
            facet_results = {}
            for facet in facets:
                facet_key = f"{facet}_facet"
                if facet_key in response["aggregations"]:
                    facet_results[facet] = [
                        {"value": bucket["key"], "count": bucket["doc_count"]}
                        for bucket in response["aggregations"][facet_key]["buckets"]
                    ]
            
            return {
                "results": hits,
                "total_count": total_count,
                "facets": facet_results
            }
            
        except Exception as e:
            logger.error(f"Faceted search failed: {e}")
            return {"results": [], "total_count": 0, "facets": {}}
    
    def suggest_data_products(
        self,
        partial_query: str,
        limit: int = 10
    ) -> List[str]:
        """Auto-suggest data product names"""
        
        search_body = {
            "suggest": {
                "data_product_suggest": {
                    "prefix": partial_query,
                    "completion": {
                        "field": "name.suggest",
                        "size": limit
                    }
                }
            }
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            suggestions = response["suggest"]["data_product_suggest"][0]["options"]
            return [suggestion["text"] for suggestion in suggestions]
            
        except Exception as e:
            logger.error(f"Suggestion failed: {e}")
            return []
    
    def get_similar_products(
        self,
        product_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar data products using More Like This query"""
        
        search_body = {
            "query": {
                "more_like_this": {
                    "fields": ["name", "description", "searchable_content"],
                    "like": [
                        {
                            "_index": self.index_name,
                            "_id": product_id
                        }
                    ],
                    "min_term_freq": 1,
                    "max_query_terms": 12
                }
            },
            "size": limit
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            return [hit["_source"] for hit in response["hits"]["hits"]]
            
        except Exception as e:
            logger.error(f"Similar products search failed: {e}")
            return []
    
    def _prepare_document(self, data_product: DataProduct) -> Dict[str, Any]:
        """Prepare data product for indexing"""
        
        # Create searchable content by combining relevant fields
        searchable_content = " ".join(filter(None, [
            data_product.name,
            data_product.description,
            data_product.owner,
            " ".join(data_product.compliance_tags or []),
            json.dumps(data_product.product_metadata or {})
        ]))
        
        doc = {
            "id": str(data_product.id),
            "name": data_product.name,
            "description": data_product.description,
            "owner": data_product.owner,
            "access_level": data_product.access_level,
            "verification_status": data_product.verification_status,
            "compliance_tags": data_product.compliance_tags or [],
            "quality_score": data_product.quality_score,
            "bias_score": data_product.bias_score,
            "created_at": data_product.created_at.isoformat(),
            "updated_at": data_product.updated_at.isoformat(),
            "freshness_timestamp": data_product.freshness_timestamp.isoformat(),
            "schema_definition": data_product.schema_definition,
            "metadata": data_product.product_metadata or {},
            "searchable_content": searchable_content
        }
        
        return doc
    
    def _build_query(
        self,
        query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build Elasticsearch query with filters"""
        
        if not query and not filters:
            return {"match_all": {}}
        
        bool_query = {"bool": {"must": [], "filter": []}}
        
        # Add text query
        if query:
            bool_query["bool"]["must"].append({
                "multi_match": {
                    "query": query,
                    "fields": ["name^3", "description^2", "searchable_content"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
        
        # Add filters
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    bool_query["bool"]["filter"].append({
                        "terms": {field: value}
                    })
                else:
                    bool_query["bool"]["filter"].append({
                        "term": {field: value}
                    })
        
        return bool_query if bool_query["bool"]["must"] or bool_query["bool"]["filter"] else {"match_all": {}}