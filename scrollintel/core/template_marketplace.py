"""
Template Marketplace for Community Dashboard Templates

Provides marketplace functionality for sharing and discovering community templates.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from enum import Enum

from .template_engine import DashboardTemplate, IndustryType, TemplateCategory

class MarketplaceStatus(Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FEATURED = "featured"
    DEPRECATED = "deprecated"

class LicenseType(Enum):
    FREE = "free"
    PREMIUM = "premium"
    OPEN_SOURCE = "open_source"
    COMMERCIAL = "commercial"

@dataclass
class TemplateRating:
    """User rating for marketplace template"""
    rating_id: str
    template_id: str
    user_id: str
    rating: int  # 1-5 stars
    review: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime] = None

@dataclass
class MarketplaceTemplate:
    """Template in marketplace with additional metadata"""
    marketplace_id: str
    template: DashboardTemplate
    author_id: str
    author_name: str
    status: MarketplaceStatus
    license_type: LicenseType
    price: float = 0.0
    download_count: int = 0
    rating_average: float = 0.0
    rating_count: int = 0
    featured_order: Optional[int] = None
    submission_date: datetime = None
    approval_date: Optional[datetime] = None
    last_updated: datetime = None
    screenshots: List[str] = None
    demo_url: Optional[str] = None
    documentation_url: Optional[str] = None
    support_email: Optional[str] = None
    changelog: List[Dict[str, Any]] = None
    requirements: List[str] = None
    compatible_versions: List[str] = None

@dataclass
class MarketplaceCollection:
    """Curated collection of templates"""
    collection_id: str
    name: str
    description: str
    curator_id: str
    template_ids: List[str]
    is_featured: bool = False
    created_at: datetime = None
    updated_at: Optional[datetime] = None
    tags: List[str] = None

class TemplateMarketplace:
    """Marketplace for community dashboard templates"""
    
    def __init__(self):
        self.marketplace_templates: Dict[str, MarketplaceTemplate] = {}
        self.ratings: Dict[str, List[TemplateRating]] = {}  # template_id -> ratings
        self.collections: Dict[str, MarketplaceCollection] = {}
        self.user_downloads: Dict[str, Set[str]] = {}  # user_id -> template_ids
        self.featured_templates: List[str] = []
        self._initialize_sample_templates()
    
    def _initialize_sample_templates(self):
        """Initialize marketplace with sample templates"""
        
        # Sample featured template
        from .template_engine import WidgetConfig
        
        sample_widgets = [
            WidgetConfig(
                id="sample_kpi_grid",
                type="kpi_grid",
                title="Executive KPI Overview",
                position={"x": 0, "y": 0, "width": 12, "height": 3},
                data_source="executive_metrics",
                visualization_config={
                    "kpis": ["revenue", "growth_rate", "customer_satisfaction", "team_productivity"],
                    "layout": "grid_4x1"
                },
                filters=[]
            )
        ]
        
        sample_template = DashboardTemplate(
            id="community_executive_v1",
            name="Community Executive Dashboard",
            description="Popular executive dashboard template from the community",
            industry=IndustryType.GENERIC,
            category=TemplateCategory.EXECUTIVE,
            widgets=sample_widgets,
            layout_config={"grid_size": 12, "row_height": 60},
            default_filters=[],
            metadata={"community_contributed": True},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["executive", "kpi", "community"]
        )
        
        marketplace_template = MarketplaceTemplate(
            marketplace_id="mp_" + str(uuid.uuid4()),
            template=sample_template,
            author_id="community_user_1",
            author_name="Dashboard Expert",
            status=MarketplaceStatus.FEATURED,
            license_type=LicenseType.FREE,
            submission_date=datetime.now(),
            approval_date=datetime.now(),
            last_updated=datetime.now(),
            screenshots=["screenshot1.png", "screenshot2.png"],
            requirements=["ScrollIntel v2.0+"],
            compatible_versions=["2.0", "2.1", "2.2"]
        )
        
        self.marketplace_templates[marketplace_template.marketplace_id] = marketplace_template
        self.featured_templates.append(marketplace_template.marketplace_id)
    
    def submit_template(self, 
                       template: DashboardTemplate,
                       author_id: str,
                       author_name: str,
                       license_type: LicenseType = LicenseType.FREE,
                       price: float = 0.0,
                       screenshots: Optional[List[str]] = None,
                       demo_url: Optional[str] = None,
                       documentation_url: Optional[str] = None,
                       support_email: Optional[str] = None,
                       requirements: Optional[List[str]] = None) -> str:
        """Submit template to marketplace"""
        
        marketplace_id = "mp_" + str(uuid.uuid4())
        
        marketplace_template = MarketplaceTemplate(
            marketplace_id=marketplace_id,
            template=template,
            author_id=author_id,
            author_name=author_name,
            status=MarketplaceStatus.PENDING_REVIEW,
            license_type=license_type,
            price=price,
            submission_date=datetime.now(),
            last_updated=datetime.now(),
            screenshots=screenshots or [],
            demo_url=demo_url,
            documentation_url=documentation_url,
            support_email=support_email,
            requirements=requirements or [],
            changelog=[{
                "version": template.version,
                "date": datetime.now().isoformat(),
                "changes": ["Initial submission"]
            }]
        )
        
        self.marketplace_templates[marketplace_id] = marketplace_template
        return marketplace_id
    
    def approve_template(self, marketplace_id: str, reviewer_id: str) -> bool:
        """Approve template for marketplace"""
        
        template = self.marketplace_templates.get(marketplace_id)
        if not template or template.status != MarketplaceStatus.PENDING_REVIEW:
            return False
        
        template.status = MarketplaceStatus.APPROVED
        template.approval_date = datetime.now()
        return True
    
    def reject_template(self, marketplace_id: str, reviewer_id: str, reason: str) -> bool:
        """Reject template submission"""
        
        template = self.marketplace_templates.get(marketplace_id)
        if not template or template.status != MarketplaceStatus.PENDING_REVIEW:
            return False
        
        template.status = MarketplaceStatus.REJECTED
        # Store rejection reason in metadata
        if not template.template.metadata:
            template.template.metadata = {}
        template.template.metadata['rejection_reason'] = reason
        template.template.metadata['rejected_by'] = reviewer_id
        template.template.metadata['rejected_at'] = datetime.now().isoformat()
        
        return True
    
    def feature_template(self, marketplace_id: str, featured_order: int = 0) -> bool:
        """Feature template in marketplace"""
        
        template = self.marketplace_templates.get(marketplace_id)
        if not template or template.status != MarketplaceStatus.APPROVED:
            return False
        
        template.status = MarketplaceStatus.FEATURED
        template.featured_order = featured_order
        
        if marketplace_id not in self.featured_templates:
            self.featured_templates.append(marketplace_id)
        
        # Sort featured templates by order
        self.featured_templates.sort(key=lambda tid: 
            self.marketplace_templates[tid].featured_order or 999)
        
        return True
    
    def download_template(self, marketplace_id: str, user_id: str) -> Optional[DashboardTemplate]:
        """Download template from marketplace"""
        
        template = self.marketplace_templates.get(marketplace_id)
        if not template or template.status not in [MarketplaceStatus.APPROVED, MarketplaceStatus.FEATURED]:
            return None
        
        # Increment download count
        template.download_count += 1
        
        # Track user download
        if user_id not in self.user_downloads:
            self.user_downloads[user_id] = set()
        self.user_downloads[user_id].add(marketplace_id)
        
        # Return copy of template
        return template.template
    
    def rate_template(self, 
                     marketplace_id: str,
                     user_id: str,
                     rating: int,
                     review: Optional[str] = None) -> bool:
        """Rate marketplace template"""
        
        if rating < 1 or rating > 5:
            return False
        
        template = self.marketplace_templates.get(marketplace_id)
        if not template:
            return False
        
        rating_id = str(uuid.uuid4())
        
        template_rating = TemplateRating(
            rating_id=rating_id,
            template_id=marketplace_id,
            user_id=user_id,
            rating=rating,
            review=review,
            created_at=datetime.now()
        )
        
        if marketplace_id not in self.ratings:
            self.ratings[marketplace_id] = []
        
        # Remove existing rating from same user
        self.ratings[marketplace_id] = [
            r for r in self.ratings[marketplace_id] if r.user_id != user_id
        ]
        
        self.ratings[marketplace_id].append(template_rating)
        
        # Update template rating statistics
        self._update_template_rating_stats(marketplace_id)
        
        return True
    
    def search_templates(self, 
                        query: Optional[str] = None,
                        industry: Optional[IndustryType] = None,
                        category: Optional[TemplateCategory] = None,
                        license_type: Optional[LicenseType] = None,
                        min_rating: Optional[float] = None,
                        tags: Optional[List[str]] = None,
                        sort_by: str = "popularity") -> List[MarketplaceTemplate]:
        """Search marketplace templates"""
        
        templates = [t for t in self.marketplace_templates.values() 
                    if t.status in [MarketplaceStatus.APPROVED, MarketplaceStatus.FEATURED]]
        
        # Apply filters
        if query:
            query_lower = query.lower()
            templates = [t for t in templates if 
                        query_lower in t.template.name.lower() or
                        query_lower in t.template.description.lower() or
                        any(query_lower in tag.lower() for tag in (t.template.tags or []))]
        
        if industry:
            templates = [t for t in templates if t.template.industry == industry]
        
        if category:
            templates = [t for t in templates if t.template.category == category]
        
        if license_type:
            templates = [t for t in templates if t.license_type == license_type]
        
        if min_rating:
            templates = [t for t in templates if t.rating_average >= min_rating]
        
        if tags:
            templates = [t for t in templates if 
                        any(tag in (t.template.tags or []) for tag in tags)]
        
        # Sort results
        if sort_by == "popularity":
            templates.sort(key=lambda t: t.download_count, reverse=True)
        elif sort_by == "rating":
            templates.sort(key=lambda t: t.rating_average, reverse=True)
        elif sort_by == "newest":
            templates.sort(key=lambda t: t.submission_date, reverse=True)
        elif sort_by == "name":
            templates.sort(key=lambda t: t.template.name)
        
        return templates
    
    def get_featured_templates(self) -> List[MarketplaceTemplate]:
        """Get featured templates"""
        return [self.marketplace_templates[tid] for tid in self.featured_templates 
                if tid in self.marketplace_templates]
    
    def get_template_ratings(self, marketplace_id: str) -> List[TemplateRating]:
        """Get ratings for template"""
        return self.ratings.get(marketplace_id, [])
    
    def create_collection(self, 
                         name: str,
                         description: str,
                         curator_id: str,
                         template_ids: List[str],
                         tags: Optional[List[str]] = None) -> str:
        """Create curated collection"""
        
        collection_id = "col_" + str(uuid.uuid4())
        
        collection = MarketplaceCollection(
            collection_id=collection_id,
            name=name,
            description=description,
            curator_id=curator_id,
            template_ids=template_ids,
            created_at=datetime.now(),
            tags=tags or []
        )
        
        self.collections[collection_id] = collection
        return collection_id
    
    def get_collections(self, featured_only: bool = False) -> List[MarketplaceCollection]:
        """Get collections"""
        collections = list(self.collections.values())
        
        if featured_only:
            collections = [c for c in collections if c.is_featured]
        
        return collections
    
    def get_user_downloads(self, user_id: str) -> List[MarketplaceTemplate]:
        """Get templates downloaded by user"""
        downloaded_ids = self.user_downloads.get(user_id, set())
        return [self.marketplace_templates[tid] for tid in downloaded_ids 
                if tid in self.marketplace_templates]
    
    def get_author_templates(self, author_id: str) -> List[MarketplaceTemplate]:
        """Get templates by author"""
        return [t for t in self.marketplace_templates.values() 
                if t.author_id == author_id]
    
    def update_template(self, 
                       marketplace_id: str,
                       template: DashboardTemplate,
                       changelog_entry: str) -> bool:
        """Update existing marketplace template"""
        
        marketplace_template = self.marketplace_templates.get(marketplace_id)
        if not marketplace_template:
            return False
        
        # Update template
        marketplace_template.template = template
        marketplace_template.last_updated = datetime.now()
        
        # Add changelog entry
        if not marketplace_template.changelog:
            marketplace_template.changelog = []
        
        marketplace_template.changelog.append({
            "version": template.version,
            "date": datetime.now().isoformat(),
            "changes": [changelog_entry]
        })
        
        return True
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        
        total_templates = len(self.marketplace_templates)
        approved_templates = len([t for t in self.marketplace_templates.values() 
                                 if t.status in [MarketplaceStatus.APPROVED, MarketplaceStatus.FEATURED]])
        featured_templates = len([t for t in self.marketplace_templates.values() 
                                if t.status == MarketplaceStatus.FEATURED])
        
        total_downloads = sum(t.download_count for t in self.marketplace_templates.values())
        
        # Industry distribution
        industry_stats = {}
        for template in self.marketplace_templates.values():
            industry = template.template.industry.value
            industry_stats[industry] = industry_stats.get(industry, 0) + 1
        
        return {
            "total_templates": total_templates,
            "approved_templates": approved_templates,
            "featured_templates": featured_templates,
            "total_downloads": total_downloads,
            "industry_distribution": industry_stats,
            "total_collections": len(self.collections),
            "total_ratings": sum(len(ratings) for ratings in self.ratings.values())
        }
    
    def _update_template_rating_stats(self, marketplace_id: str):
        """Update template rating statistics"""
        
        template = self.marketplace_templates.get(marketplace_id)
        ratings = self.ratings.get(marketplace_id, [])
        
        if not template or not ratings:
            return
        
        total_rating = sum(r.rating for r in ratings)
        template.rating_average = total_rating / len(ratings)
        template.rating_count = len(ratings)