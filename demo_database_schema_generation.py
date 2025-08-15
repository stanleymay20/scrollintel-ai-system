"""
Demo script for database schema generation system.
"""
import uuid
import json
from datetime import datetime

from scrollintel.models.database_schema_models import (
    SchemaGenerationRequest, DatabaseType
)
from scrollintel.models.code_generation_models import (
    Requirements, ParsedRequirement, Entity as RequirementEntity,
    Relationship, RequirementType, Intent, EntityType, ConfidenceLevel
)
from scrollintel.engines.database_schema_generator import DatabaseSchemaGenerator


def create_sample_requirements():
    """Create sample requirements for e-commerce system."""
    
    # Create entities
    user_entity = RequirementEntity(
        id=str(uuid.uuid4()),
        name="User",
        type=EntityType.DATA_ENTITY,
        description="User entity for customer authentication and profile management",
        confidence=0.95,
        source_text="Users need to register, login, and manage their profiles",
        position=(0, 55),
        attributes={
            "fields": ["username", "email", "password", "first_name", "last_name", "phone"],
            "authentication": True,
            "profile_management": True
        }
    )
    
    product_entity = RequirementEntity(
        id=str(uuid.uuid4()),
        name="Product",
        type=EntityType.DATA_ENTITY,
        description="Product entity for catalog management",
        confidence=0.9,
        source_text="Products need to be cataloged with name, description, price, and inventory",
        position=(56, 120),
        attributes={
            "fields": ["name", "description", "price", "sku", "inventory_count", "category"],
            "catalog_management": True
        }
    )
    
    order_entity = RequirementEntity(
        id=str(uuid.uuid4()),
        name="Order",
        type=EntityType.DATA_ENTITY,
        description="Order entity for purchase tracking",
        confidence=0.92,
        source_text="Orders need to track customer purchases with items, quantities, and totals",
        position=(121, 190),
        attributes={
            "fields": ["order_number", "customer_id", "status", "total_amount", "order_date"],
            "purchase_tracking": True
        }
    )
    
    order_item_entity = RequirementEntity(
        id=str(uuid.uuid4()),
        name="OrderItem",
        type=EntityType.DATA_ENTITY,
        description="Order item entity for individual line items",
        confidence=0.88,
        source_text="Each order contains multiple items with quantities and prices",
        position=(191, 250),
        attributes={
            "fields": ["order_id", "product_id", "quantity", "unit_price", "total_price"]
        }
    )
    
    category_entity = RequirementEntity(
        id=str(uuid.uuid4()),
        name="Category",
        type=EntityType.DATA_ENTITY,
        description="Product category entity for organization",
        confidence=0.85,
        source_text="Products are organized into categories for better navigation",
        position=(251, 310),
        attributes={
            "fields": ["name", "description", "parent_category_id"]
        }
    )
    
    # Create relationships
    user_order_relationship = Relationship(
        id=str(uuid.uuid4()),
        source_entity_id="Order",
        target_entity_id="User",
        relationship_type="many_to_one",
        description="Orders belong to users (customers)",
        confidence=0.95
    )
    
    order_item_order_relationship = Relationship(
        id=str(uuid.uuid4()),
        source_entity_id="OrderItem",
        target_entity_id="Order",
        relationship_type="many_to_one",
        description="Order items belong to orders",
        confidence=0.98
    )
    
    order_item_product_relationship = Relationship(
        id=str(uuid.uuid4()),
        source_entity_id="OrderItem",
        target_entity_id="Product",
        relationship_type="many_to_one",
        description="Order items reference products",
        confidence=0.95
    )
    
    product_category_relationship = Relationship(
        id=str(uuid.uuid4()),
        source_entity_id="Product",
        target_entity_id="Category",
        relationship_type="many_to_one",
        description="Products belong to categories",
        confidence=0.9
    )
    
    category_parent_relationship = Relationship(
        id=str(uuid.uuid4()),
        source_entity_id="Category",
        target_entity_id="Category",
        relationship_type="many_to_one",
        description="Categories can have parent categories (hierarchical)",
        confidence=0.8
    )
    
    # Create parsed requirements
    auth_requirement = ParsedRequirement(
        id=str(uuid.uuid4()),
        original_text="Users must be able to register, login, and manage their profiles securely",
        structured_text="The system shall provide user authentication and profile management capabilities",
        requirement_type=RequirementType.FUNCTIONAL,
        intent=Intent.CREATE_APPLICATION,
        entities=[user_entity],
        acceptance_criteria=[
            "Users can register with email and password",
            "Users can login with valid credentials",
            "Users can update their profile information",
            "Passwords are securely hashed and stored"
        ],
        priority=1,
        complexity=3,
        confidence=ConfidenceLevel.HIGH
    )
    
    catalog_requirement = ParsedRequirement(
        id=str(uuid.uuid4()),
        original_text="Products must be cataloged with detailed information and organized by categories",
        structured_text="The system shall provide product catalog management with categorization",
        requirement_type=RequirementType.FUNCTIONAL,
        intent=Intent.CREATE_APPLICATION,
        entities=[product_entity, category_entity],
        acceptance_criteria=[
            "Products have name, description, price, and SKU",
            "Products are assigned to categories",
            "Categories can be hierarchical",
            "Inventory levels are tracked"
        ],
        priority=1,
        complexity=4,
        confidence=ConfidenceLevel.HIGH
    )
    
    order_requirement = ParsedRequirement(
        id=str(uuid.uuid4()),
        original_text="Orders must track customer purchases with line items and calculate totals",
        structured_text="The system shall provide order management with line item tracking",
        requirement_type=RequirementType.FUNCTIONAL,
        intent=Intent.CREATE_APPLICATION,
        entities=[order_entity, order_item_entity],
        acceptance_criteria=[
            "Orders are associated with customers",
            "Orders contain multiple line items",
            "Line items reference products with quantities",
            "Order totals are calculated automatically",
            "Order status is tracked through fulfillment"
        ],
        priority=1,
        complexity=5,
        confidence=ConfidenceLevel.HIGH
    )
    
    # Create requirements container
    requirements = Requirements(
        id=str(uuid.uuid4()),
        project_name="E-Commerce Platform",
        raw_text="""
        Build an e-commerce platform where users can register, login, and manage their profiles.
        Products need to be cataloged with detailed information including name, description, price, 
        SKU, and inventory levels. Products should be organized into hierarchical categories.
        
        Customers should be able to place orders containing multiple items. Each order should track
        the customer, order date, status, and total amount. Order items should reference products
        with quantities and calculate line totals.
        
        The system needs to handle high traffic with good performance and maintain data integrity.
        Security is important for user authentication and payment processing.
        """,
        parsed_requirements=[auth_requirement, catalog_requirement, order_requirement],
        entities=[user_entity, product_entity, order_entity, order_item_entity, category_entity],
        relationships=[
            user_order_relationship,
            order_item_order_relationship,
            order_item_product_relationship,
            product_category_relationship,
            category_parent_relationship
        ],
        completeness_score=0.85,
        processing_status="completed"
    )
    
    return requirements


def demo_postgresql_schema_generation():
    """Demo PostgreSQL schema generation."""
    print("🐘 PostgreSQL Schema Generation Demo")
    print("=" * 50)
    
    # Create generator
    generator = DatabaseSchemaGenerator()
    
    # Create requirements
    requirements = create_sample_requirements()
    
    # Create generation request
    request = SchemaGenerationRequest(
        requirements_id=requirements.id,
        database_type=DatabaseType.POSTGRESQL,
        performance_requirements={
            "max_query_time": "100ms",
            "max_concurrent_connections": 1000,
            "expected_read_write_ratio": "80:20"
        },
        scalability_requirements={
            "max_concurrent_users": 50000,
            "expected_data_growth": "1TB/year",
            "peak_transactions_per_second": 10000
        },
        security_requirements={
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "audit_logging": True,
            "row_level_security": False
        },
        optimization_level="high"
    )
    
    # Generate schema
    print("Generating PostgreSQL schema...")
    result = generator.generate_schema(request, requirements)
    
    if result.success:
        print(f"✅ Schema generated successfully in {result.generation_time:.2f} seconds")
        print(f"📊 Generated {len(result.schema.entities)} entities")
        print(f"🔗 Established {len(result.schema.relationships)} relationships")
        print(f"⚡ Created {len(result.schema.optimizations)} optimizations")
        
        # Display entities
        print("\n📋 Generated Entities:")
        for entity in result.schema.entities:
            print(f"  • {entity.name} ({entity.table_name})")
            print(f"    - Fields: {len(entity.fields)}")
            print(f"    - Indexes: {len(entity.indexes)}")
            print(f"    - Constraints: {len(entity.constraints)}")
        
        # Display SQL preview
        if "create_schema.sql" in result.sql_scripts:
            sql_preview = result.sql_scripts["create_schema.sql"][:500]
            print(f"\n📝 SQL Preview (first 500 chars):")
            print("-" * 30)
            print(sql_preview)
            if len(result.sql_scripts["create_schema.sql"]) > 500:
                print("... (truncated)")
        
        # Validate schema
        print("\n🔍 Validating schema...")
        validation = generator.validate_schema(result.schema)
        
        if validation.is_valid:
            print("✅ Schema validation passed")
        else:
            print("❌ Schema validation failed")
            for error in validation.errors:
                print(f"  • Error: {error}")
        
        print(f"📈 Performance Score: {validation.performance_score:.2f}")
        print(f"📐 Normalization Score: {validation.normalization_score:.2f}")
        print(f"🔒 Security Score: {validation.security_score:.2f}")
        
        if validation.suggestions:
            print("\n💡 Optimization Suggestions:")
            for suggestion in validation.suggestions[:3]:  # Show first 3
                print(f"  • {suggestion}")
    
    else:
        print("❌ Schema generation failed")
        for error in result.errors:
            print(f"  • {error}")
    
    return result


def demo_mysql_schema_generation():
    """Demo MySQL schema generation."""
    print("\n🐬 MySQL Schema Generation Demo")
    print("=" * 50)
    
    generator = DatabaseSchemaGenerator()
    requirements = create_sample_requirements()
    
    request = SchemaGenerationRequest(
        requirements_id=requirements.id,
        database_type=DatabaseType.MYSQL,
        performance_requirements={
            "max_query_time": "50ms",
            "innodb_buffer_pool_size": "8GB"
        },
        optimization_level="medium"
    )
    
    print("Generating MySQL schema...")
    result = generator.generate_schema(request, requirements)
    
    if result.success:
        print(f"✅ MySQL schema generated in {result.generation_time:.2f} seconds")
        
        # Show MySQL-specific SQL
        if "create_schema.sql" in result.sql_scripts:
            sql_lines = result.sql_scripts["create_schema.sql"].split('\n')
            mysql_features = [line for line in sql_lines if 'ENGINE=InnoDB' in line or 'AUTO_INCREMENT' in line]
            
            if mysql_features:
                print("\n🔧 MySQL-specific features detected:")
                for feature in mysql_features[:3]:
                    print(f"  • {feature.strip()}")
    
    return result


def demo_mongodb_schema_generation():
    """Demo MongoDB schema generation."""
    print("\n🍃 MongoDB Schema Generation Demo")
    print("=" * 50)
    
    generator = DatabaseSchemaGenerator()
    requirements = create_sample_requirements()
    
    request = SchemaGenerationRequest(
        requirements_id=requirements.id,
        database_type=DatabaseType.MONGODB,
        performance_requirements={
            "max_document_size": "16MB",
            "sharding_strategy": "hash"
        },
        optimization_level="medium"
    )
    
    print("Generating MongoDB schema...")
    result = generator.generate_schema(request, requirements)
    
    if result.success:
        print(f"✅ MongoDB schema generated in {result.generation_time:.2f} seconds")
        
        # Show MongoDB-specific JavaScript
        js_files = [k for k in result.sql_scripts.keys() if k.endswith('.js')]
        if js_files:
            js_preview = result.sql_scripts[js_files[0]][:300]
            print(f"\n📝 JavaScript Preview:")
            print("-" * 30)
            print(js_preview)
    
    return result


def demo_migration_generation():
    """Demo migration generation between schema versions."""
    print("\n🔄 Migration Generation Demo")
    print("=" * 50)
    
    generator = DatabaseSchemaGenerator()
    requirements = create_sample_requirements()
    
    # Generate initial schema
    request_v1 = SchemaGenerationRequest(
        requirements_id=requirements.id,
        database_type=DatabaseType.POSTGRESQL,
        optimization_level="low"
    )
    
    result_v1 = generator.generate_schema(request_v1, requirements)
    
    if not result_v1.success:
        print("❌ Failed to generate initial schema")
        return
    
    # Modify requirements for v2 (add new entity)
    review_entity = RequirementEntity(
        id=str(uuid.uuid4()),
        name="Review",
        type=EntityType.DATA_ENTITY,
        description="Product review entity",
        confidence=0.9,
        source_text="Customers can leave reviews for products",
        position=(0, 40)
    )
    
    requirements.entities.append(review_entity)
    
    # Generate v2 schema
    request_v2 = SchemaGenerationRequest(
        requirements_id=requirements.id,
        database_type=DatabaseType.POSTGRESQL,
        optimization_level="medium"
    )
    
    result_v2 = generator.generate_schema(request_v2, requirements)
    
    if result_v2.success:
        # Generate migration
        print("Generating migration from v1 to v2...")
        migration = generator.generate_migration(result_v1.schema, result_v2.schema)
        
        print(f"✅ Migration generated: {migration.version}")
        print(f"📝 Description: {migration.description}")
        print(f"🔧 Operations: {len(migration.operations)}")
        
        if migration.operations:
            print("\n📋 Migration Operations:")
            for op in migration.operations:
                print(f"  • {op.operation_type}: {op.entity_name}")
        
        # Show migration SQL preview
        if migration.up_sql:
            sql_preview = migration.up_sql[:200]
            print(f"\n⬆️  Forward Migration SQL (preview):")
            print("-" * 30)
            print(sql_preview)
            if len(migration.up_sql) > 200:
                print("... (truncated)")


def demo_performance_analysis():
    """Demo performance analysis and optimization suggestions."""
    print("\n⚡ Performance Analysis Demo")
    print("=" * 50)
    
    generator = DatabaseSchemaGenerator()
    requirements = create_sample_requirements()
    
    # Generate schema with high performance requirements
    request = SchemaGenerationRequest(
        requirements_id=requirements.id,
        database_type=DatabaseType.POSTGRESQL,
        performance_requirements={
            "max_query_time": "10ms",  # Very strict
            "max_concurrent_connections": 10000,
            "expected_read_write_ratio": "90:10",
            "peak_transactions_per_second": 50000
        },
        scalability_requirements={
            "max_concurrent_users": 1000000,
            "expected_data_growth": "10TB/year"
        },
        optimization_level="high"
    )
    
    result = generator.generate_schema(request, requirements)
    
    if result.success:
        print(f"✅ High-performance schema generated")
        print(f"⚡ Optimizations created: {len(result.schema.optimizations)}")
        
        # Show optimizations
        if result.schema.optimizations:
            print("\n🚀 Performance Optimizations:")
            for opt in result.schema.optimizations:
                impact_emoji = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "💡"}
                emoji = impact_emoji.get(opt.impact_level, "📈")
                print(f"  {emoji} {opt.optimization_type.title()}: {opt.description}")
                print(f"    Impact: {opt.impact_level}")
                if opt.estimated_improvement:
                    print(f"    Improvement: {opt.estimated_improvement}")
        
        # Validate for performance
        validation = generator.validate_schema(result.schema)
        print(f"\n📊 Performance Metrics:")
        print(f"  • Performance Score: {validation.performance_score:.2f}/1.0")
        print(f"  • Normalization Score: {validation.normalization_score:.2f}/1.0")
        print(f"  • Security Score: {validation.security_score:.2f}/1.0")


def demo_documentation_generation():
    """Demo documentation generation."""
    print("\n📚 Documentation Generation Demo")
    print("=" * 50)
    
    generator = DatabaseSchemaGenerator()
    requirements = create_sample_requirements()
    
    request = SchemaGenerationRequest(
        requirements_id=requirements.id,
        database_type=DatabaseType.POSTGRESQL,
        optimization_level="medium"
    )
    
    result = generator.generate_schema(request, requirements)
    
    if result.success and result.documentation:
        print("✅ Documentation generated")
        
        # Show documentation preview
        doc_lines = result.documentation.split('\n')
        preview_lines = doc_lines[:20]  # First 20 lines
        
        print("\n📖 Documentation Preview:")
        print("-" * 40)
        for line in preview_lines:
            print(line)
        
        if len(doc_lines) > 20:
            print(f"... ({len(doc_lines) - 20} more lines)")
        
        print(f"\n📏 Total documentation length: {len(result.documentation)} characters")


def main():
    """Run all database schema generation demos."""
    print("🗄️  Database Schema Generation System Demo")
    print("=" * 60)
    print("This demo showcases the automated database schema generation")
    print("capabilities for different database types and use cases.")
    print()
    
    try:
        # Run demos
        demo_postgresql_schema_generation()
        demo_mysql_schema_generation()
        demo_mongodb_schema_generation()
        demo_migration_generation()
        demo_performance_analysis()
        demo_documentation_generation()
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Multi-database support (PostgreSQL, MySQL, MongoDB)")
        print("  ✅ Entity and relationship extraction from requirements")
        print("  ✅ Automatic field generation with appropriate types")
        print("  ✅ Index and constraint creation")
        print("  ✅ Performance optimization suggestions")
        print("  ✅ Schema validation and quality scoring")
        print("  ✅ Migration generation between versions")
        print("  ✅ SQL script generation")
        print("  ✅ Comprehensive documentation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()