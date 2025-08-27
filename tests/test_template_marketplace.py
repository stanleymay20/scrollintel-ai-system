"""
Unit tests for Template Marketplace
"""

import pytest
from datetime import datetime
from scrollintel.core.template_marketplace import (
    TemplateMarketplace,
    MarketplaceTemplate,
    TemplateRating,
    MarketplaceCollection,
    MarketplaceStatus,
    LicenseType
)
from scrollintel.core.template_engine import (
    DashboardTemplate,
    WidgetConfig,
    IndustryType,
    TemplateCategory
)


class TestTemplateMarketplace:
    """Test cases for TemplateMarketplace"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.marketplace = TemplateMarketplace()
        
        # Sample template
        self.sample_widget = WidgetConfig(
            id="sample_widget",
            type="metric_card",
            title="Sample Widget",
            position={"x": 0, "y": 0, "width": 4, "height": 2},
            data_source="sample_source",
            visualization_config={},
            filters=[]
        )
        
        self.sample_template = DashboardTemplate(
            id="sample_template",
            name="Sample Dashboard",
            description="Sample dashboard for testing",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.EXECUTIVE,
            widgets=[self.sample_widget],
            layout_config={"grid_size": 12, "row_height": 60, "margin": [10, 10]},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["sample", "test"]
        )
    
    def test_initialization(self):
        """Test marketplace initialization"""
        # Should have at least one sample template
        assert len(self.marketplace.marketplace_templates) >= 1
        assert len(self.marketplace.featured_templates) >= 1
        
        # Check sample template exists
        sample_exists = any(
            t.status == MarketplaceStatus.FEATURED 
            for t in self.marketplace.marketplace_templates.values()
        )
        assert sample_exists
    
    def test_submit_template(self):
        """Test submitting template to marketplace"""
        initial_count = len(self.marketplace.marketplace_templates)
        
        # Submit template
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="test_author",
            author_name="Test Author",
            license_type=LicenseType.FREE,
            screenshots=["screenshot1.png"],
            demo_url="https://demo.example.com",
            requirements=["ScrollIntel v2.0+"]
        )
        
        assert marketplace_id is not None
        assert marketplace_id.startswith("mp_")
        assert len(self.marketplace.marketplace_templates) == initial_count + 1
        
        # Verify submitted template
        submitted = self.marketplace.marketplace_templates[marketplace_id]
        assert submitted.template.name == "Sample Dashboard"
        assert submitted.author_id == "test_author"
        assert submitted.author_name == "Test Author"
        assert submitted.status == MarketplaceStatus.PENDING_REVIEW
        assert submitted.license_type == LicenseType.FREE
        assert submitted.price == 0.0
        assert len(submitted.screenshots) == 1
        assert submitted.demo_url == "https://demo.example.com"
        assert "ScrollIntel v2.0+" in submitted.requirements
        assert len(submitted.changelog) == 1
    
    def test_submit_premium_template(self):
        """Test submitting premium template"""
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="premium_author",
            author_name="Premium Author",
            license_type=LicenseType.PREMIUM,
            price=29.99
        )
        
        submitted = self.marketplace.marketplace_templates[marketplace_id]
        assert submitted.license_type == LicenseType.PREMIUM
        assert submitted.price == 29.99
    
    def test_approve_template(self):
        """Test approving template"""
        # Submit template
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="test_author",
            author_name="Test Author"
        )
        
        # Approve template
        success = self.marketplace.approve_template(marketplace_id, "reviewer_1")
        assert success is True
        
        approved = self.marketplace.marketplace_templates[marketplace_id]
        assert approved.status == MarketplaceStatus.APPROVED
        assert approved.approval_date is not None
        
        # Test approving non-existent template
        assert self.marketplace.approve_template("non_existent", "reviewer") is False
        
        # Test approving already approved template
        assert self.marketplace.approve_template(marketplace_id, "reviewer_2") is False
    
    def test_reject_template(self):
        """Test rejecting template"""
        # Submit template
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="test_author",
            author_name="Test Author"
        )
        
        # Reject template
        success = self.marketplace.reject_template(
            marketplace_id, 
            "reviewer_1", 
            "Does not meet quality standards"
        )
        assert success is True
        
        rejected = self.marketplace.marketplace_templates[marketplace_id]
        assert rejected.status == MarketplaceStatus.REJECTED
        assert rejected.template.metadata["rejection_reason"] == "Does not meet quality standards"
        assert rejected.template.metadata["rejected_by"] == "reviewer_1"
        
        # Test rejecting non-existent template
        assert self.marketplace.reject_template("non_existent", "reviewer", "reason") is False
    
    def test_feature_template(self):
        """Test featuring template"""
        # Submit and approve template
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="test_author",
            author_name="Test Author"
        )
        self.marketplace.approve_template(marketplace_id, "reviewer")
        
        initial_featured_count = len(self.marketplace.featured_templates)
        
        # Feature template
        success = self.marketplace.feature_template(marketplace_id, featured_order=1)
        assert success is True
        
        featured = self.marketplace.marketplace_templates[marketplace_id]
        assert featured.status == MarketplaceStatus.FEATURED
        assert featured.featured_order == 1
        assert marketplace_id in self.marketplace.featured_templates
        assert len(self.marketplace.featured_templates) == initial_featured_count + 1
        
        # Test featuring non-approved template
        pending_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="pending_author",
            author_name="Pending Author"
        )
        assert self.marketplace.feature_template(pending_id) is False
    
    def test_download_template(self):
        """Test downloading template"""
        # Submit and approve template
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="test_author",
            author_name="Test Author"
        )
        self.marketplace.approve_template(marketplace_id, "reviewer")
        
        initial_downloads = self.marketplace.marketplace_templates[marketplace_id].download_count
        
        # Download template
        downloaded = self.marketplace.download_template(marketplace_id, "user_1")
        
        assert downloaded is not None
        assert downloaded.name == "Sample Dashboard"
        assert downloaded.id == "sample_template"
        
        # Verify download tracking
        marketplace_template = self.marketplace.marketplace_templates[marketplace_id]
        assert marketplace_template.download_count == initial_downloads + 1
        assert "user_1" in self.marketplace.user_downloads
        assert marketplace_id in self.marketplace.user_downloads["user_1"]
        
        # Test downloading non-existent template
        assert self.marketplace.download_template("non_existent", "user_1") is None
        
        # Test downloading pending template
        pending_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="pending_author",
            author_name="Pending Author"
        )
        assert self.marketplace.download_template(pending_id, "user_1") is None
    
    def test_rate_template(self):
        """Test rating template"""
        # Submit and approve template
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="test_author",
            author_name="Test Author"
        )
        self.marketplace.approve_template(marketplace_id, "reviewer")
        
        # Rate template
        success = self.marketplace.rate_template(
            marketplace_id,
            "user_1",
            5,
            "Excellent template!"
        )
        assert success is True
        
        # Verify rating
        ratings = self.marketplace.get_template_ratings(marketplace_id)
        assert len(ratings) == 1
        assert ratings[0].rating == 5
        assert ratings[0].review == "Excellent template!"
        assert ratings[0].user_id == "user_1"
        
        # Verify rating statistics update
        marketplace_template = self.marketplace.marketplace_templates[marketplace_id]
        assert marketplace_template.rating_average == 5.0
        assert marketplace_template.rating_count == 1
        
        # Add another rating
        self.marketplace.rate_template(marketplace_id, "user_2", 4, "Good template")
        
        updated_template = self.marketplace.marketplace_templates[marketplace_id]
        assert updated_template.rating_average == 4.5  # (5 + 4) / 2
        assert updated_template.rating_count == 2
        
        # Test updating existing rating
        self.marketplace.rate_template(marketplace_id, "user_1", 3, "Changed my mind")
        
        final_ratings = self.marketplace.get_template_ratings(marketplace_id)
        assert len(final_ratings) == 2  # Should still be 2 ratings
        user_1_rating = next(r for r in final_ratings if r.user_id == "user_1")
        assert user_1_rating.rating == 3
        assert user_1_rating.review == "Changed my mind"
        
        # Test invalid rating
        assert self.marketplace.rate_template(marketplace_id, "user_3", 6) is False
        assert self.marketplace.rate_template(marketplace_id, "user_3", 0) is False
    
    def test_search_templates(self):
        """Test searching marketplace templates"""
        # Create diverse templates
        tech_template = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="tech_author",
            author_name="Tech Author"
        )
        self.marketplace.approve_template(tech_template, "reviewer")
        
        finance_template_data = DashboardTemplate(
            id="finance_template",
            name="Finance Dashboard",
            description="Financial metrics and KPIs",
            industry=IndustryType.FINANCE,
            category=TemplateCategory.FINANCIAL,
            widgets=[],
            layout_config={"grid_size": 12},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=["finance", "kpi"]
        )
        
        finance_template = self.marketplace.submit_template(
            template=finance_template_data,
            author_id="finance_author",
            author_name="Finance Author",
            license_type=LicenseType.PREMIUM,
            price=19.99
        )
        self.marketplace.approve_template(finance_template, "reviewer")
        
        # Rate templates
        self.marketplace.rate_template(tech_template, "user_1", 5)
        self.marketplace.rate_template(finance_template, "user_1", 4)
        
        # Search by query
        results = self.marketplace.search_templates(query="Finance")
        assert len(results) >= 1
        assert any("Finance" in t.template.name for t in results)
        
        # Search by industry
        tech_results = self.marketplace.search_templates(industry=IndustryType.TECHNOLOGY)
        assert len(tech_results) >= 1
        assert all(t.template.industry == IndustryType.TECHNOLOGY for t in tech_results)
        
        # Search by category
        financial_results = self.marketplace.search_templates(category=TemplateCategory.FINANCIAL)
        assert len(financial_results) >= 1
        assert all(t.template.category == TemplateCategory.FINANCIAL for t in financial_results)
        
        # Search by license type
        premium_results = self.marketplace.search_templates(license_type=LicenseType.PREMIUM)
        assert len(premium_results) >= 1
        assert all(t.license_type == LicenseType.PREMIUM for t in premium_results)
        
        # Search by minimum rating
        high_rated = self.marketplace.search_templates(min_rating=4.5)
        assert len(high_rated) >= 1
        assert all(t.rating_average >= 4.5 for t in high_rated)
        
        # Search by tags
        finance_tagged = self.marketplace.search_templates(tags=["finance"])
        assert len(finance_tagged) >= 1
        assert all("finance" in (t.template.tags or []) for t in finance_tagged)
        
        # Test sorting
        popularity_sorted = self.marketplace.search_templates(sort_by="popularity")
        rating_sorted = self.marketplace.search_templates(sort_by="rating")
        name_sorted = self.marketplace.search_templates(sort_by="name")
        
        assert len(popularity_sorted) >= 2
        assert len(rating_sorted) >= 2
        assert len(name_sorted) >= 2
    
    def test_get_featured_templates(self):
        """Test getting featured templates"""
        # Feature some templates
        template_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="featured_author",
            author_name="Featured Author"
        )
        self.marketplace.approve_template(template_id, "reviewer")
        self.marketplace.feature_template(template_id, featured_order=1)
        
        featured = self.marketplace.get_featured_templates()
        assert len(featured) >= 1
        assert all(t.status == MarketplaceStatus.FEATURED for t in featured)
        
        # Check that featured templates include our new one
        featured_ids = [t.marketplace_id for t in featured]
        assert template_id in featured_ids
    
    def test_create_collection(self):
        """Test creating template collection"""
        # Create some templates
        template1_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="author_1",
            author_name="Author 1"
        )
        
        template2_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="author_2",
            author_name="Author 2"
        )
        
        # Create collection
        collection_id = self.marketplace.create_collection(
            name="Executive Dashboards",
            description="Collection of executive dashboard templates",
            curator_id="curator_1",
            template_ids=[template1_id, template2_id],
            tags=["executive", "curated"]
        )
        
        assert collection_id is not None
        assert collection_id.startswith("col_")
        assert collection_id in self.marketplace.collections
        
        collection = self.marketplace.collections[collection_id]
        assert collection.name == "Executive Dashboards"
        assert collection.curator_id == "curator_1"
        assert len(collection.template_ids) == 2
        assert template1_id in collection.template_ids
        assert template2_id in collection.template_ids
        assert "executive" in collection.tags
    
    def test_get_collections(self):
        """Test getting collections"""
        # Create collections
        collection1_id = self.marketplace.create_collection(
            name="Collection 1",
            description="First collection",
            curator_id="curator_1",
            template_ids=[]
        )
        
        collection2_id = self.marketplace.create_collection(
            name="Collection 2",
            description="Second collection",
            curator_id="curator_2",
            template_ids=[]
        )
        
        # Mark one as featured
        self.marketplace.collections[collection1_id].is_featured = True
        
        # Get all collections
        all_collections = self.marketplace.get_collections()
        assert len(all_collections) >= 2
        
        # Get featured collections only
        featured_collections = self.marketplace.get_collections(featured_only=True)
        assert len(featured_collections) >= 1
        assert all(c.is_featured for c in featured_collections)
    
    def test_get_user_downloads(self):
        """Test getting user downloads"""
        # Submit and approve templates
        template1_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="author_1",
            author_name="Author 1"
        )
        self.marketplace.approve_template(template1_id, "reviewer")
        
        template2_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="author_2",
            author_name="Author 2"
        )
        self.marketplace.approve_template(template2_id, "reviewer")
        
        # Download templates
        self.marketplace.download_template(template1_id, "user_1")
        self.marketplace.download_template(template2_id, "user_1")
        
        # Get user downloads
        downloads = self.marketplace.get_user_downloads("user_1")
        assert len(downloads) == 2
        
        download_ids = [d.marketplace_id for d in downloads]
        assert template1_id in download_ids
        assert template2_id in download_ids
        
        # Test user with no downloads
        no_downloads = self.marketplace.get_user_downloads("user_2")
        assert len(no_downloads) == 0
    
    def test_get_author_templates(self):
        """Test getting templates by author"""
        # Submit templates by same author
        template1_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="prolific_author",
            author_name="Prolific Author"
        )
        
        template2_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="prolific_author",
            author_name="Prolific Author"
        )
        
        # Submit template by different author
        template3_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="other_author",
            author_name="Other Author"
        )
        
        # Get templates by prolific author
        author_templates = self.marketplace.get_author_templates("prolific_author")
        assert len(author_templates) == 2
        
        author_ids = [t.marketplace_id for t in author_templates]
        assert template1_id in author_ids
        assert template2_id in author_ids
        assert template3_id not in author_ids
        
        # Test author with no templates
        no_templates = self.marketplace.get_author_templates("nonexistent_author")
        assert len(no_templates) == 0
    
    def test_update_template(self):
        """Test updating marketplace template"""
        # Submit template
        marketplace_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="author",
            author_name="Author"
        )
        
        original_updated = self.marketplace.marketplace_templates[marketplace_id].last_updated
        
        # Update template
        updated_template = DashboardTemplate(
            id="updated_template",
            name="Updated Dashboard",
            description="Updated description",
            industry=IndustryType.TECHNOLOGY,
            category=TemplateCategory.EXECUTIVE,
            widgets=[],
            layout_config={"grid_size": 16},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version="2.0.0"
        )
        
        success = self.marketplace.update_template(
            marketplace_id,
            updated_template,
            "Major update with new features"
        )
        
        assert success is True
        
        # Verify update
        marketplace_template = self.marketplace.marketplace_templates[marketplace_id]
        assert marketplace_template.template.name == "Updated Dashboard"
        assert marketplace_template.template.version == "2.0.0"
        assert marketplace_template.last_updated > original_updated
        assert len(marketplace_template.changelog) == 2  # Original + update
        assert "Major update with new features" in marketplace_template.changelog[1]["changes"]
        
        # Test updating non-existent template
        assert self.marketplace.update_template("non_existent", updated_template, "update") is False
    
    def test_get_marketplace_stats(self):
        """Test getting marketplace statistics"""
        # Submit various templates
        tech_id = self.marketplace.submit_template(
            template=self.sample_template,
            author_id="tech_author",
            author_name="Tech Author"
        )
        self.marketplace.approve_template(tech_id, "reviewer")
        
        finance_template = DashboardTemplate(
            id="finance_stats",
            name="Finance Stats",
            description="Finance dashboard",
            industry=IndustryType.FINANCE,
            category=TemplateCategory.FINANCIAL,
            widgets=[],
            layout_config={"grid_size": 12},
            default_filters=[],
            metadata={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        finance_id = self.marketplace.submit_template(
            template=finance_template,
            author_id="finance_author",
            author_name="Finance Author"
        )
        self.marketplace.approve_template(finance_id, "reviewer")
        self.marketplace.feature_template(finance_id)
        
        # Download and rate
        self.marketplace.download_template(tech_id, "user_1")
        self.marketplace.download_template(finance_id, "user_1")
        self.marketplace.rate_template(tech_id, "user_1", 5)
        
        # Create collection
        self.marketplace.create_collection(
            name="Test Collection",
            description="Test",
            curator_id="curator",
            template_ids=[tech_id]
        )
        
        # Get stats
        stats = self.marketplace.get_marketplace_stats()
        
        assert stats["total_templates"] >= 3  # Including sample template
        assert stats["approved_templates"] >= 2
        assert stats["featured_templates"] >= 2  # Including sample template
        assert stats["total_downloads"] >= 2
        assert stats["total_collections"] >= 1
        assert stats["total_ratings"] >= 1
        
        # Check industry distribution
        assert "technology" in stats["industry_distribution"]
        assert "finance" in stats["industry_distribution"]
        assert stats["industry_distribution"]["technology"] >= 1
        assert stats["industry_distribution"]["finance"] >= 1


if __name__ == "__main__":
    pytest.main([__file__])