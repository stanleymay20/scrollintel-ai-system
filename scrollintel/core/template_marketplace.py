"""
Template marketplace for community-contributed dashboard templates.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import uuid
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, Float
from sqlalchemy.orm import Session
from scrollintel.models.database import Base, get_db
from scrollintel.core.template_engine import template_engine, IndustryType, TemplateCategory


class MarketplaceStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FEATURED = "featured"
    DEPRECATED = "deprecated"


class TemplateMarketplaceEntry(Base):
    """Database model for marketplace template entries."""
    __tablename__ = "marketplace_templates"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    author_id = Column(String, nullable=False)
    author_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    industry = Column(String, nullable=False)
    tags = Column(Text, nullable=False)  # JSON array
    version = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    download_count = Column(Integer, default=0)
    rating_average = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    price = Column(Float, default=0.0)  # 0 for free templates
    license_type = Column(String, default="MIT")
    preview_images = Column(Text, nullable=True)  # JSON array of image URLs
    demo_url = Column(String, nullable=True)
    documentation_url = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(String, nullable=True)
    is_featured = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)


class TemplateRating(Base):
    """Database model for template ratings."""
    __tablename__ = "template_ratings"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    review = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class TemplateDownload(Base):
    """Database model for template downloads."""
    __tablename__ = "template_downloads"
    
    id = Column(String, primary_key=True)
    template_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    downloaded_at = Column(DateTime, default=datetime.utcnow)
    version = Column(String, nullable=False)


class TemplateMarketplace:
    """Manager for template marketplace operations."""
    
    def __init__(self):
        self.db_session = None
    
    def _get_db_session(self) -> Session:
        """Get database session."""
        if not self.db_session:
            self.db_session = next(get_db())
        return self.db_session
    
    def submit_template(
        self,
        template_id: str,
        title: str,
        description: str,
        author_id: str,
        author_name: str,
        category: str,
        industry: str,
        tags: List[str],
        version: str = "1.0.0",
        price: float = 0.0,
        license_type: str = "MIT",
        preview_images: List[str] = None,
        demo_url: str = None,
        documentation_url: str = None,
        source_url: str = None
    ) -> str:
        """Submit a template to the marketplace."""
        db = self._get_db_session()
        
        entry_id = f"marketplace_{uuid.uuid4().hex[:12]}"
        
        marketplace_entry = TemplateMarketplaceEntry(
            id=entry_id,
            template_id=template_id,
            title=title,
            description=description,
            author_id=author_id,
            author_name=author_name,
            category=category,
            industry=industry,
            tags=json.dumps(tags),
            version=version,
            status=MarketplaceStatus.PENDING.value,
            price=price,
            license_type=license_type,
            preview_images=json.dumps(preview_images or []),
            demo_url=demo_url,
            documentation_url=documentation_url,
            source_url=source_url
        )
        
        db.add(marketplace_entry)
        db.commit()
        
        return entry_id
    
    def approve_template(self, entry_id: str, approved_by: str) -> bool:
        """Approve a template for the marketplace."""
        db = self._get_db_session()
        
        entry = db.query(TemplateMarketplaceEntry)\
            .filter(TemplateMarketplaceEntry.id == entry_id)\
            .first()
        
        if not entry:
            return False
        
        entry.status = MarketplaceStatus.APPROVED.value
        entry.approved_at = datetime.utcnow()
        entry.approved_by = approved_by
        entry.updated_at = datetime.utcnow()
        
        db.commit()
        return True
    
    def reject_template(self, entry_id: str, rejected_by: str, reason: str = "") -> bool:
        """Reject a template submission."""
        db = self._get_db_session()
        
        entry = db.query(TemplateMarketplaceEntry)\
            .filter(TemplateMarketplaceEntry.id == entry_id)\
            .first()
        
        if not entry:
            return False
        
        entry.status = MarketplaceStatus.REJECTED.value
        entry.updated_at = datetime.utcnow()
        
        # TODO: Store rejection reason
        
        db.commit()
        return True
    
    def feature_template(self, entry_id: str) -> bool:
        """Feature a template in the marketplace."""
        db = self._get_db_session()
        
        entry = db.query(TemplateMarketplaceEntry)\
            .filter(TemplateMarketplaceEntry.id == entry_id)\
            .first()
        
        if not entry or entry.status != MarketplaceStatus.APPROVED.value:
            return False
        
        entry.is_featured = True
        entry.status = MarketplaceStatus.FEATURED.value
        entry.updated_at = datetime.utcnow()
        
        db.commit()
        return True
    
    def search_templates(
        self,
        query: str = "",
        category: str = None,
        industry: str = None,
        tags: List[str] = None,
        min_rating: float = 0.0,
        max_price: float = None,
        sort_by: str = "popularity",  # popularity, rating, newest, price
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Search templates in the marketplace."""
        db = self._get_db_session()
        
        query_obj = db.query(TemplateMarketplaceEntry)\
            .filter(
                TemplateMarketplaceEntry.status.in_([
                    MarketplaceStatus.APPROVED.value,
                    MarketplaceStatus.FEATURED.value
                ]),
                TemplateMarketplaceEntry.is_active == True
            )
        
        # Apply filters
        if query:
            query_obj = query_obj.filter(
                TemplateMarketplaceEntry.title.contains(query) |
                TemplateMarketplaceEntry.description.contains(query)
            )
        
        if category:
            query_obj = query_obj.filter(TemplateMarketplaceEntry.category == category)
        
        if industry:
            query_obj = query_obj.filter(TemplateMarketplaceEntry.industry == industry)
        
        if min_rating > 0:
            query_obj = query_obj.filter(TemplateMarketplaceEntry.rating_average >= min_rating)
        
        if max_price is not None:
            query_obj = query_obj.filter(TemplateMarketplaceEntry.price <= max_price)
        
        # Apply sorting
        if sort_by == "popularity":
            query_obj = query_obj.order_by(TemplateMarketplaceEntry.download_count.desc())
        elif sort_by == "rating":
            query_obj = query_obj.order_by(TemplateMarketplaceEntry.rating_average.desc())
        elif sort_by == "newest":
            query_obj = query_obj.order_by(TemplateMarketplaceEntry.created_at.desc())
        elif sort_by == "price":
            query_obj = query_obj.order_by(TemplateMarketplaceEntry.price.asc())
        
        # Apply pagination
        templates = query_obj.limit(limit).offset(offset).all()
        
        return [self._format_marketplace_entry(entry) for entry in templates]
    
    def get_featured_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get featured templates."""
        db = self._get_db_session()
        
        featured = db.query(TemplateMarketplaceEntry)\
            .filter(
                TemplateMarketplaceEntry.is_featured == True,
                TemplateMarketplaceEntry.is_active == True
            )\
            .order_by(TemplateMarketplaceEntry.download_count.desc())\
            .limit(limit)\
            .all()
        
        return [self._format_marketplace_entry(entry) for entry in featured]
    
    def get_template_details(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a marketplace template."""
        db = self._get_db_session()
        
        entry = db.query(TemplateMarketplaceEntry)\
            .filter(TemplateMarketplaceEntry.template_id == template_id)\
            .first()
        
        if not entry:
            return None
        
        # Get ratings
        ratings = db.query(TemplateRating)\
            .filter(
                TemplateRating.template_id == template_id,
                TemplateRating.is_active == True
            )\
            .order_by(TemplateRating.created_at.desc())\
            .limit(10)\
            .all()
        
        details = self._format_marketplace_entry(entry)
        details["ratings"] = [
            {
                "id": rating.id,
                "user_id": rating.user_id,
                "rating": rating.rating,
                "review": rating.review,
                "created_at": rating.created_at.isoformat()
            }
            for rating in ratings
        ]
        
        return details
    
    def download_template(self, template_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Download a template from the marketplace."""
        db = self._get_db_session()
        
        entry = db.query(TemplateMarketplaceEntry)\
            .filter(TemplateMarketplaceEntry.template_id == template_id)\
            .first()
        
        if not entry or entry.status not in [MarketplaceStatus.APPROVED.value, MarketplaceStatus.FEATURED.value]:
            return None
        
        # Record download
        download_id = f"download_{uuid.uuid4().hex[:12]}"
        download_record = TemplateDownload(
            id=download_id,
            template_id=template_id,
            user_id=user_id,
            version=entry.version
        )
        
        db.add(download_record)
        
        # Increment download count
        entry.download_count += 1
        entry.updated_at = datetime.utcnow()
        
        db.commit()
        
        # Get template data from template engine
        template = template_engine.get_template(template_id)
        if not template:
            return None
        
        return {
            "template_data": template_engine.export_template(template_id),
            "marketplace_info": self._format_marketplace_entry(entry),
            "download_id": download_id
        }
    
    def rate_template(
        self,
        template_id: str,
        user_id: str,
        rating: int,
        review: str = ""
    ) -> bool:
        """Rate a template."""
        db = self._get_db_session()
        
        if rating < 1 or rating > 5:
            return False
        
        # Check if user already rated this template
        existing_rating = db.query(TemplateRating)\
            .filter(
                TemplateRating.template_id == template_id,
                TemplateRating.user_id == user_id,
                TemplateRating.is_active == True
            )\
            .first()
        
        if existing_rating:
            # Update existing rating
            existing_rating.rating = rating
            existing_rating.review = review
            existing_rating.updated_at = datetime.utcnow()
        else:
            # Create new rating
            rating_id = f"rating_{uuid.uuid4().hex[:12]}"
            new_rating = TemplateRating(
                id=rating_id,
                template_id=template_id,
                user_id=user_id,
                rating=rating,
                review=review
            )
            db.add(new_rating)
        
        db.commit()
        
        # Update template rating average
        self._update_template_rating(template_id)
        
        return True
    
    def _update_template_rating(self, template_id: str):
        """Update the average rating for a template."""
        db = self._get_db_session()
        
        # Calculate new average
        ratings = db.query(TemplateRating)\
            .filter(
                TemplateRating.template_id == template_id,
                TemplateRating.is_active == True
            )\
            .all()
        
        if not ratings:
            return
        
        total_rating = sum(r.rating for r in ratings)
        average_rating = total_rating / len(ratings)
        
        # Update marketplace entry
        entry = db.query(TemplateMarketplaceEntry)\
            .filter(TemplateMarketplaceEntry.template_id == template_id)\
            .first()
        
        if entry:
            entry.rating_average = round(average_rating, 2)
            entry.rating_count = len(ratings)
            entry.updated_at = datetime.utcnow()
            db.commit()
    
    def get_user_downloads(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's download history."""
        db = self._get_db_session()
        
        downloads = db.query(TemplateDownload)\
            .filter(TemplateDownload.user_id == user_id)\
            .order_by(TemplateDownload.downloaded_at.desc())\
            .all()
        
        return [
            {
                "id": download.id,
                "template_id": download.template_id,
                "version": download.version,
                "downloaded_at": download.downloaded_at.isoformat()
            }
            for download in downloads
        ]
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        db = self._get_db_session()
        
        total_templates = db.query(TemplateMarketplaceEntry)\
            .filter(TemplateMarketplaceEntry.is_active == True)\
            .count()
        
        approved_templates = db.query(TemplateMarketplaceEntry)\
            .filter(
                TemplateMarketplaceEntry.status.in_([
                    MarketplaceStatus.APPROVED.value,
                    MarketplaceStatus.FEATURED.value
                ]),
                TemplateMarketplaceEntry.is_active == True
            )\
            .count()
        
        total_downloads = db.query(TemplateDownload).count()
        
        return {
            "total_templates": total_templates,
            "approved_templates": approved_templates,
            "pending_templates": total_templates - approved_templates,
            "total_downloads": total_downloads,
            "categories": self._get_category_stats(),
            "industries": self._get_industry_stats()
        }
    
    def _get_category_stats(self) -> Dict[str, int]:
        """Get template count by category."""
        db = self._get_db_session()
        
        categories = {}
        for category in TemplateCategory:
            count = db.query(TemplateMarketplaceEntry)\
                .filter(
                    TemplateMarketplaceEntry.category == category.value,
                    TemplateMarketplaceEntry.status.in_([
                        MarketplaceStatus.APPROVED.value,
                        MarketplaceStatus.FEATURED.value
                    ]),
                    TemplateMarketplaceEntry.is_active == True
                )\
                .count()
            categories[category.value] = count
        
        return categories
    
    def _get_industry_stats(self) -> Dict[str, int]:
        """Get template count by industry."""
        db = self._get_db_session()
        
        industries = {}
        for industry in IndustryType:
            count = db.query(TemplateMarketplaceEntry)\
                .filter(
                    TemplateMarketplaceEntry.industry == industry.value,
                    TemplateMarketplaceEntry.status.in_([
                        MarketplaceStatus.APPROVED.value,
                        MarketplaceStatus.FEATURED.value
                    ]),
                    TemplateMarketplaceEntry.is_active == True
                )\
                .count()
            industries[industry.value] = count
        
        return industries
    
    def _format_marketplace_entry(self, entry: TemplateMarketplaceEntry) -> Dict[str, Any]:
        """Format marketplace entry for API response."""
        return {
            "id": entry.id,
            "template_id": entry.template_id,
            "title": entry.title,
            "description": entry.description,
            "author_id": entry.author_id,
            "author_name": entry.author_name,
            "category": entry.category,
            "industry": entry.industry,
            "tags": json.loads(entry.tags),
            "version": entry.version,
            "status": entry.status,
            "download_count": entry.download_count,
            "rating_average": entry.rating_average,
            "rating_count": entry.rating_count,
            "price": entry.price,
            "license_type": entry.license_type,
            "preview_images": json.loads(entry.preview_images or "[]"),
            "demo_url": entry.demo_url,
            "documentation_url": entry.documentation_url,
            "source_url": entry.source_url,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "is_featured": entry.is_featured
        }


# Global marketplace instance
marketplace = TemplateMarketplace()