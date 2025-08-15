"""
Tests for the template marketplace functionality.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from scrollintel.core.template_marketplace import (
    TemplateMarketplace,
    MarketplaceStatus,
    TemplateMarketplaceEntry,
    TemplateRating,
    TemplateDownload
)


class TestTemplateMarketplace:
    """Test cases for TemplateMarketplace."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.marketplace = TemplateMarketplace()
        # Mock database session
        self.mock_db = Mock()
        self.marketplace.db_session = self.mock_db
    
    def test_submit_template(self):
        """Test submitting a template to marketplace."""
        entry_id = self.marketplace.submit_template(
            template_id="test_template",
            title="Test Dashboard Template",
            description="A comprehensive test dashboard",
            author_id="user123",
            author_name="Test User",
            category="executive",
            industry="technology",
            tags=["dashboard", "executive", "technology"],
            version="1.0.0",
            price=0.0,
            license_type="MIT"
        )
        
        assert entry_id.startswith("marketplace_")
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
        
        # Check the created entry
        added_entry = self.mock_db.add.call_args[0][0]
        assert isinstance(added_entry, TemplateMarketplaceEntry)
        assert added_entry.template_id == "test_template"
        assert added_entry.title == "Test Dashboard Template"
        assert added_entry.author_id == "user123"
        assert added_entry.status == MarketplaceStatus.PENDING.value
        assert added_entry.price == 0.0
    
    def test_approve_template(self):
        """Test approving a template submission."""
        # Mock existing entry
        mock_entry = Mock()
        mock_entry.status = MarketplaceStatus.PENDING.value
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
        
        result = self.marketplace.approve_template("entry123", "admin_user")
        
        assert result is True
        assert mock_entry.status == MarketplaceStatus.APPROVED.value
        assert mock_entry.approved_by == "admin_user"
        assert mock_entry.approved_at is not None
        self.mock_db.commit.assert_called_once()
    
    def test_approve_nonexistent_template(self):
        """Test approving non-existent template."""
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = self.marketplace.approve_template("nonexistent", "admin_user")
        
        assert result is False
        self.mock_db.commit.assert_not_called()
    
    def test_reject_template(self):
        """Test rejecting a template submission."""
        mock_entry = Mock()
        mock_entry.status = MarketplaceStatus.PENDING.value
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
        
        result = self.marketplace.reject_template("entry123", "admin_user", "Quality issues")
        
        assert result is True
        assert mock_entry.status == MarketplaceStatus.REJECTED.value
        assert mock_entry.updated_at is not None
        self.mock_db.commit.assert_called_once()
    
    def test_feature_template(self):
        """Test featuring an approved template."""
        mock_entry = Mock()
        mock_entry.status = MarketplaceStatus.APPROVED.value
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
        
        result = self.marketplace.feature_template("entry123")
        
        assert result is True
        assert mock_entry.is_featured is True
        assert mock_entry.status == MarketplaceStatus.FEATURED.value
        self.mock_db.commit.assert_called_once()
    
    def test_feature_unapproved_template(self):
        """Test featuring an unapproved template fails."""
        mock_entry = Mock()
        mock_entry.status = MarketplaceStatus.PENDING.value
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
        
        result = self.marketplace.feature_template("entry123")
        
        assert result is False
        self.mock_db.commit.assert_not_called()
    
    @patch('scrollintel.core.template_marketplace.template_engine')
    def test_download_template(self, mock_template_engine):
        """Test downloading a template from marketplace."""
        # Mock marketplace entry
        mock_entry = Mock()
        mock_entry.template_id = "test_template"
        mock_entry.status = MarketplaceStatus.APPROVED.value
        mock_entry.version = "1.0.0"
        mock_entry.download_count = 5
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
        
        # Mock template engine
        mock_template_engine.get_template.return_value = Mock()
        mock_template_engine.export_template.return_value = {"template": "data"}
        
        result = self.marketplace.download_template("test_template", "user123")
        
        assert result is not None
        assert "template_data" in result
        assert "marketplace_info" in result
        assert "download_id" in result
        
        # Check download record was created
        self.mock_db.add.assert_called()
        added_download = self.mock_db.add.call_args[0][0]
        assert isinstance(added_download, TemplateDownload)
        assert added_download.template_id == "test_template"
        assert added_download.user_id == "user123"
        
        # Check download count was incremented
        assert mock_entry.download_count == 6
        self.mock_db.commit.assert_called()
    
    def test_download_unapproved_template(self):
        """Test downloading unapproved template fails."""
        mock_entry = Mock()
        mock_entry.status = MarketplaceStatus.PENDING.value
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_entry
        
        result = self.marketplace.download_template("test_template", "user123")
        
        assert result is None
        self.mock_db.add.assert_not_called()
    
    def test_rate_template_new_rating(self):
        """Test rating a template for the first time."""
        # Mock no existing rating
        self.mock_db.query.return_value.filter.return_value.first.return_value = None
        
        # Mock rating calculation
        mock_ratings = [Mock(rating=4), Mock(rating=5), Mock(rating=3)]
        self.mock_db.query.return_value.filter.return_value.all.return_value = mock_ratings
        
        mock_entry = Mock()
        mock_entry.rating_average = 0.0
        mock_entry.rating_count = 0
        
        # Setup query chain for rating update
        query_mock = Mock()
        query_mock.filter.return_value.first.return_value = mock_entry
        self.mock_db.query.return_value = query_mock
        
        result = self.marketplace.rate_template("test_template", "user123", 4, "Great template!")
        
        assert result is True
        self.mock_db.add.assert_called()
        
        # Check new rating was created
        added_rating = self.mock_db.add.call_args[0][0]
        assert isinstance(added_rating, TemplateRating)
        assert added_rating.template_id == "test_template"
        assert added_rating.user_id == "user123"
        assert added_rating.rating == 4
        assert added_rating.review == "Great template!"
    
    def test_rate_template_update_existing(self):
        """Test updating an existing rating."""
        mock_existing_rating = Mock()
        mock_existing_rating.rating = 3
        mock_existing_rating.review = "Old review"
        self.mock_db.query.return_value.filter.return_value.first.return_value = mock_existing_rating
        
        result = self.marketplace.rate_template("test_template", "user123", 5, "Updated review!")
        
        assert result is True
        assert mock_existing_rating.rating == 5
        assert mock_existing_rating.review == "Updated review!"
        assert mock_existing_rating.updated_at is not None
        self.mock_db.add.assert_not_called()  # Should update, not add
    
    def test_rate_template_invalid_rating(self):
        """Test rating with invalid score."""
        result = self.marketplace.rate_template("test_template", "user123", 6, "Invalid rating")
        assert result is False
        
        result = self.marketplace.rate_template("test_template", "user123", 0, "Invalid rating")
        assert result is False
    
    def test_search_templates_basic(self):
        """Test basic template search."""
        mock_templates = [
            Mock(
                id="entry1",
                template_id="template1",
                title="Executive Dashboard",
                description="Dashboard for executives",
                author_name="Author 1",
                category="executive",
                industry="technology",
                tags='["executive", "technology"]',
                version="1.0.0",
                status=MarketplaceStatus.APPROVED.value,
                download_count=10,
                rating_average=4.5,
                rating_count=5,
                price=0.0,
                license_type="MIT",
                preview_images="[]",
                demo_url=None,
                documentation_url=None,
                source_url=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_featured=False
            )
        ]
        
        # Mock query chain
        query_mock = Mock()
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.offset.return_value = query_mock
        query_mock.all.return_value = mock_templates
        self.mock_db.query.return_value = query_mock
        
        results = self.marketplace.search_templates(query="Executive")
        
        assert len(results) == 1
        assert results[0]["title"] == "Executive Dashboard"
        assert results[0]["template_id"] == "template1"
    
    def test_get_featured_templates(self):
        """Test getting featured templates."""
        mock_templates = [
            Mock(
                id="entry1",
                template_id="template1",
                title="Featured Dashboard",
                is_featured=True,
                download_count=100
            )
        ]
        
        query_mock = Mock()
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.limit.return_value = query_mock
        query_mock.all.return_value = mock_templates
        self.mock_db.query.return_value = query_mock
        
        # Mock the _format_marketplace_entry method
        self.marketplace._format_marketplace_entry = Mock(return_value={
            "id": "entry1",
            "template_id": "template1",
            "title": "Featured Dashboard"
        })
        
        results = self.marketplace.get_featured_templates(limit=5)
        
        assert len(results) == 1
        assert results[0]["title"] == "Featured Dashboard"
    
    def test_get_template_details(self):
        """Test getting detailed template information."""
        mock_entry = Mock()
        mock_entry.template_id = "test_template"
        mock_entry.title = "Test Template"
        
        mock_ratings = [
            Mock(
                id="rating1",
                user_id="user1",
                rating=5,
                review="Excellent!",
                created_at=datetime.utcnow()
            )
        ]
        
        # Setup query mocks
        entry_query = Mock()
        entry_query.filter.return_value.first.return_value = mock_entry
        
        rating_query = Mock()
        rating_query.filter.return_value = rating_query
        rating_query.order_by.return_value = rating_query
        rating_query.limit.return_value = rating_query
        rating_query.all.return_value = mock_ratings
        
        self.mock_db.query.side_effect = [entry_query, rating_query]
        
        # Mock format method
        self.marketplace._format_marketplace_entry = Mock(return_value={
            "id": "entry1",
            "template_id": "test_template",
            "title": "Test Template"
        })
        
        result = self.marketplace.get_template_details("test_template")
        
        assert result is not None
        assert result["title"] == "Test Template"
        assert "ratings" in result
        assert len(result["ratings"]) == 1
        assert result["ratings"][0]["rating"] == 5
    
    def test_get_user_downloads(self):
        """Test getting user's download history."""
        mock_downloads = [
            Mock(
                id="download1",
                template_id="template1",
                version="1.0.0",
                downloaded_at=datetime.utcnow()
            )
        ]
        
        query_mock = Mock()
        query_mock.filter.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        query_mock.all.return_value = mock_downloads
        self.mock_db.query.return_value = query_mock
        
        results = self.marketplace.get_user_downloads("user123")
        
        assert len(results) == 1
        assert results[0]["id"] == "download1"
        assert results[0]["template_id"] == "template1"
        assert results[0]["version"] == "1.0.0"
    
    def test_get_marketplace_stats(self):
        """Test getting marketplace statistics."""
        # Mock count queries
        count_mock = Mock()
        count_mock.filter.return_value.count.return_value = 10
        count_mock.count.return_value = 50
        self.mock_db.query.return_value = count_mock
        
        # Mock category and industry stats methods
        self.marketplace._get_category_stats = Mock(return_value={"executive": 5, "operational": 3})
        self.marketplace._get_industry_stats = Mock(return_value={"technology": 8, "finance": 2})
        
        stats = self.marketplace.get_marketplace_stats()
        
        assert "total_templates" in stats
        assert "approved_templates" in stats
        assert "pending_templates" in stats
        assert "total_downloads" in stats
        assert "categories" in stats
        assert "industries" in stats
    
    def test_format_marketplace_entry(self):
        """Test formatting marketplace entry for API response."""
        mock_entry = Mock()
        mock_entry.id = "entry1"
        mock_entry.template_id = "template1"
        mock_entry.title = "Test Template"
        mock_entry.description = "A test template"
        mock_entry.author_id = "user123"
        mock_entry.author_name = "Test User"
        mock_entry.category = "executive"
        mock_entry.industry = "technology"
        mock_entry.tags = '["test", "dashboard"]'
        mock_entry.version = "1.0.0"
        mock_entry.status = "approved"
        mock_entry.download_count = 5
        mock_entry.rating_average = 4.2
        mock_entry.rating_count = 3
        mock_entry.price = 0.0
        mock_entry.license_type = "MIT"
        mock_entry.preview_images = '["image1.jpg"]'
        mock_entry.demo_url = "https://demo.example.com"
        mock_entry.documentation_url = "https://docs.example.com"
        mock_entry.source_url = "https://github.com/example/template"
        mock_entry.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_entry.updated_at = datetime(2023, 1, 2, 12, 0, 0)
        mock_entry.is_featured = True
        
        result = self.marketplace._format_marketplace_entry(mock_entry)
        
        assert result["id"] == "entry1"
        assert result["template_id"] == "template1"
        assert result["title"] == "Test Template"
        assert result["tags"] == ["test", "dashboard"]
        assert result["preview_images"] == ["image1.jpg"]
        assert result["rating_average"] == 4.2
        assert result["is_featured"] is True
        assert result["created_at"] == "2023-01-01T12:00:00"


if __name__ == "__main__":
    pytest.main([__file__])