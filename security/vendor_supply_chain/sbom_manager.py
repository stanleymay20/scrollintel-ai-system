"""
Software Bill of Materials (SBOM) Tracking and Management System
Implements comprehensive SBOM generation, tracking, and analysis
"""

import asyncio
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET

class SBOMFormat(Enum):
    SPDX_JSON = "spdx-json"
    SPDX_XML = "spdx-xml"
    CYCLONE_DX_JSON = "cyclonedx-json"
    CYCLONE_DX_XML = "cyclonedx-xml"
    SWID = "swid"

class ComponentType(Enum):
    APPLICATION = "application"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    CONTAINER = "container"
    OPERATING_SYSTEM = "operating-system"
    DEVICE = "device"
    FIRMWARE = "firmware"
    FILE = "file"

class LicenseType(Enum):
    PERMISSIVE = "permissive"
    COPYLEFT = "copyleft"
    PROPRIETARY = "proprietary"
    PUBLIC_DOMAIN = "public_domain"
    UNKNOWN = "unknown"

@dataclass
class Component:
    component_id: str
    name: str
    version: str
    component_type: ComponentType
    supplier: Optional[str]
    author: Optional[str]
    publisher: Optional[str]
    namespace: Optional[str]
    description: Optional[str]
    scope: str  # required, optional, excluded
    hashes: Dict[str, str]  # algorithm -> hash
    licenses: List[str]
    license_type: LicenseType
    copyright: Optional[str]
    cpe: Optional[str]  # Common Platform Enumeration
    purl: Optional[str]  # Package URL
    external_references: List[Dict[str, str]]
    dependencies: List[str]  # Component IDs
    vulnerabilities: List[str]  # CVE IDs
    created_at: datetime
    modified_at: datetime

@dataclass
class SBOM:
    sbom_id: str
    name: str
    version: str
    format: SBOMFormat
    spec_version: str
    data_license: str
    document_namespace: str
    creation_info: Dict[str, Any]
    components: List[Component]
    relationships: List[Dict[str, str]]
    annotations: List[Dict[str, str]]
    metadata: Dict[str, Any]
    created_at: datetime
    modified_at: datetime
    vendor_id: Optional[str] = None
    software_name: Optional[str] = None

class SBOMManager:
    def __init__(self, config_path: str = "security/config/sbom_config.yaml"):
        self.config = self._load_config(config_path)
        self.sbom_storage = {}  # In production, use proper database
        self.component_registry = {}
        self.license_database = self._initialize_license_database()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load SBOM configuration"""
        default_config = {
            "default_format": SBOMFormat.SPDX_JSON.value,
            "spec_versions": {
                "spdx": "2.3",
                "cyclonedx": "1.4"
            },
            "data_license": "CC0-1.0",
            "namespace_prefix": "https://scrollintel.com/sbom/",
            "component_analysis": {
                "enabled": True,
                "vulnerability_check": True,
                "license_compliance": True,
                "dependency_analysis": True
            },
            "storage": {
                "retention_period": 2555,  # 7 years in days
                "backup_enabled": True,
                "encryption_enabled": True
            }
        }
        
        try:
            # In production, load from actual config file
            return default_config
        except Exception:
            return default_config
    
    def _initialize_license_database(self) -> Dict[str, Any]:
        """Initialize license database with common licenses"""
        return {
            "MIT": {
                "type": LicenseType.PERMISSIVE,
                "commercial_use": True,
                "distribution": True,
                "modification": True,
                "private_use": True,
                "patent_use": False,
                "trademark_use": False
            },
            "Apache-2.0": {
                "type": LicenseType.PERMISSIVE,
                "commercial_use": True,
                "distribution": True,
                "modification": True,
                "private_use": True,
                "patent_use": True,
                "trademark_use": False
            },
            "GPL-3.0": {
                "type": LicenseType.COPYLEFT,
                "commercial_use": True,
                "distribution": True,
                "modification": True,
                "private_use": True,
                "patent_use": True,
                "trademark_use": False
            },
            "BSD-3-Clause": {
                "type": LicenseType.PERMISSIVE,
                "commercial_use": True,
                "distribution": True,
                "modification": True,
                "private_use": True,
                "patent_use": False,
                "trademark_use": False
            }
        }
    
    async def generate_sbom(self, vendor_id: str, software_path: str, 
                          software_name: str, software_version: str,
                          sbom_format: SBOMFormat = SBOMFormat.SPDX_JSON) -> SBOM:
        """Generate SBOM for software package"""
        sbom_id = self._generate_sbom_id(vendor_id, software_name, software_version)
        
        # Create SBOM structure
        sbom = SBOM(
            sbom_id=sbom_id,
            name=f"{software_name}-{software_version}",
            version=software_version,
            format=sbom_format,
            spec_version=self.config["spec_versions"]["spdx"],
            data_license=self.config["data_license"],
            document_namespace=f"{self.config['namespace_prefix']}{sbom_id}",
            creation_info=self._create_creation_info(),
            components=[],
            relationships=[],
            annotations=[],
            metadata={
                "vendor_id": vendor_id,
                "software_name": software_name,
                "analysis_timestamp": datetime.now().isoformat()
            },
            created_at=datetime.now(),
            modified_at=datetime.now(),
            vendor_id=vendor_id,
            software_name=software_name
        )
        
        # Analyze software and extract components
        components = await self._analyze_software_components(software_path, software_name, software_version)
        sbom.components = components
        
        # Build relationships
        relationships = await self._build_component_relationships(components)
        sbom.relationships = relationships
        
        # Perform additional analysis if enabled
        if self.config["component_analysis"]["enabled"]:
            await self._enhance_component_analysis(sbom)
        
        # Store SBOM
        self.sbom_storage[sbom_id] = sbom
        
        return sbom
    
    def _create_creation_info(self) -> Dict[str, Any]:
        """Create SBOM creation information"""
        return {
            "created": datetime.now().isoformat(),
            "creators": [
                "Tool: ScrollIntel SBOM Manager v1.0",
                "Organization: ScrollIntel Security Team"
            ],
            "license_list_version": "3.19"
        }
    
    async def _analyze_software_components(self, software_path: str, 
                                         software_name: str, software_version: str) -> List[Component]:
        """Analyze software to extract components"""
        components = []
        
        # Add main software component
        main_component = await self._create_main_component(software_name, software_version, software_path)
        components.append(main_component)
        
        # Extract dependencies
        dependencies = await self._extract_dependencies(software_path)
        for dep in dependencies:
            component = await self._create_dependency_component(dep)
            components.append(component)
        
        # Extract embedded libraries
        embedded_libs = await self._extract_embedded_libraries(software_path)
        for lib in embedded_libs:
            component = await self._create_library_component(lib)
            components.append(component)
        
        return components
    
    async def _create_main_component(self, software_name: str, software_version: str, 
                                   software_path: str) -> Component:
        """Create main software component"""
        component_id = self._generate_component_id(software_name, software_version)
        
        # Calculate file hashes
        hashes = await self._calculate_file_hashes(software_path)
        
        return Component(
            component_id=component_id,
            name=software_name,
            version=software_version,
            component_type=ComponentType.APPLICATION,
            supplier=None,
            author=None,
            publisher=None,
            namespace=f"scrollintel.com/{software_name}",
            description=f"Main application component for {software_name}",
            scope="required",
            hashes=hashes,
            licenses=[],
            license_type=LicenseType.UNKNOWN,
            copyright=None,
            cpe=None,
            purl=f"pkg:generic/{software_name}@{software_version}",
            external_references=[],
            dependencies=[],
            vulnerabilities=[],
            created_at=datetime.now(),
            modified_at=datetime.now()
        )
    
    async def _extract_dependencies(self, software_path: str) -> List[Dict[str, Any]]:
        """Extract software dependencies"""
        dependencies = []
        
        # Look for dependency files
        dependency_files = [
            ("package.json", self._parse_npm_dependencies),
            ("requirements.txt", self._parse_pip_dependencies),
            ("pom.xml", self._parse_maven_dependencies),
            ("build.gradle", self._parse_gradle_dependencies),
            ("Gemfile", self._parse_gem_dependencies),
            ("composer.json", self._parse_composer_dependencies)
        ]
        
        software_dir = Path(software_path).parent if Path(software_path).is_file() else Path(software_path)
        
        for filename, parser in dependency_files:
            dep_file = software_dir / filename
            if dep_file.exists():
                file_deps = await parser(dep_file)
                dependencies.extend(file_deps)
        
        return dependencies
    
    async def _parse_npm_dependencies(self, package_file: Path) -> List[Dict[str, Any]]:
        """Parse NPM package.json dependencies"""
        dependencies = []
        
        try:
            with open(package_file, 'r') as f:
                data = json.load(f)
            
            for dep_type in ["dependencies", "devDependencies", "peerDependencies"]:
                deps = data.get(dep_type, {})
                for name, version in deps.items():
                    dependencies.append({
                        "name": name,
                        "version": version.strip("^~>=<"),
                        "type": "npm",
                        "scope": "required" if dep_type == "dependencies" else "optional"
                    })
        
        except Exception as e:
            print(f"Failed to parse NPM dependencies: {e}")
        
        return dependencies
    
    async def _parse_pip_dependencies(self, requirements_file: Path) -> List[Dict[str, Any]]:
        """Parse Python requirements.txt dependencies"""
        dependencies = []
        
        try:
            with open(requirements_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        name, version = line.split('==', 1)
                        dependencies.append({
                            "name": name.strip(),
                            "version": version.strip(),
                            "type": "pip",
                            "scope": "required"
                        })
        
        except Exception as e:
            print(f"Failed to parse pip dependencies: {e}")
        
        return dependencies
    
    async def _parse_maven_dependencies(self, pom_file: Path) -> List[Dict[str, Any]]:
        """Parse Maven pom.xml dependencies"""
        dependencies = []
        
        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()
            
            # Handle XML namespace
            namespace = {'maven': 'http://maven.apache.org/POM/4.0.0'}
            
            deps = root.findall('.//maven:dependency', namespace)
            for dep in deps:
                group_id = dep.find('maven:groupId', namespace)
                artifact_id = dep.find('maven:artifactId', namespace)
                version = dep.find('maven:version', namespace)
                scope = dep.find('maven:scope', namespace)
                
                if group_id is not None and artifact_id is not None:
                    dependencies.append({
                        "name": f"{group_id.text}:{artifact_id.text}",
                        "version": version.text if version is not None else "unknown",
                        "type": "maven",
                        "scope": scope.text if scope is not None else "required"
                    })
        
        except Exception as e:
            print(f"Failed to parse Maven dependencies: {e}")
        
        return dependencies
    
    async def _parse_gradle_dependencies(self, build_file: Path) -> List[Dict[str, Any]]:
        """Parse Gradle build.gradle dependencies"""
        dependencies = []
        
        try:
            with open(build_file, 'r') as f:
                content = f.read()
            
            # Simple regex-based parsing (in production, use proper Gradle parser)
            import re
            dep_pattern = r"implementation\s+['\"]([^:]+):([^:]+):([^'\"]+)['\"]"
            matches = re.findall(dep_pattern, content)
            
            for group, artifact, version in matches:
                dependencies.append({
                    "name": f"{group}:{artifact}",
                    "version": version,
                    "type": "gradle",
                    "scope": "required"
                })
        
        except Exception as e:
            print(f"Failed to parse Gradle dependencies: {e}")
        
        return dependencies
    
    async def _parse_gem_dependencies(self, gemfile: Path) -> List[Dict[str, Any]]:
        """Parse Ruby Gemfile dependencies"""
        dependencies = []
        
        try:
            with open(gemfile, 'r') as f:
                content = f.read()
            
            # Simple regex-based parsing
            import re
            gem_pattern = r"gem\s+['\"]([^'\"]+)['\"](?:,\s*['\"]([^'\"]+)['\"])?"
            matches = re.findall(gem_pattern, content)
            
            for name, version in matches:
                dependencies.append({
                    "name": name,
                    "version": version if version else "latest",
                    "type": "gem",
                    "scope": "required"
                })
        
        except Exception as e:
            print(f"Failed to parse Gem dependencies: {e}")
        
        return dependencies
    
    async def _parse_composer_dependencies(self, composer_file: Path) -> List[Dict[str, Any]]:
        """Parse PHP composer.json dependencies"""
        dependencies = []
        
        try:
            with open(composer_file, 'r') as f:
                data = json.load(f)
            
            for dep_type in ["require", "require-dev"]:
                deps = data.get(dep_type, {})
                for name, version in deps.items():
                    if not name.startswith("php"):  # Skip PHP version requirements
                        dependencies.append({
                            "name": name,
                            "version": version.strip("^~>=<"),
                            "type": "composer",
                            "scope": "required" if dep_type == "require" else "optional"
                        })
        
        except Exception as e:
            print(f"Failed to parse Composer dependencies: {e}")
        
        return dependencies
    
    async def _extract_embedded_libraries(self, software_path: str) -> List[Dict[str, Any]]:
        """Extract embedded libraries from software"""
        libraries = []
        
        # Look for common library patterns
        library_patterns = [
            r"jquery-(\d+\.\d+\.\d+)",
            r"bootstrap-(\d+\.\d+\.\d+)",
            r"react-(\d+\.\d+\.\d+)",
            r"angular-(\d+\.\d+\.\d+)"
        ]
        
        try:
            software_dir = Path(software_path).parent if Path(software_path).is_file() else Path(software_path)
            
            for root, dirs, files in software_dir.rglob("*"):
                for file in files:
                    file_path = Path(root) / file
                    
                    # Check filename for library patterns
                    for pattern in library_patterns:
                        import re
                        match = re.search(pattern, file_path.name, re.IGNORECASE)
                        if match:
                            lib_name = pattern.split('-')[0].replace(r'\d+\.\d+\.\d+', '').strip('()')
                            version = match.group(1)
                            
                            libraries.append({
                                "name": lib_name,
                                "version": version,
                                "type": "embedded",
                                "file_path": str(file_path)
                            })
        
        except Exception as e:
            print(f"Failed to extract embedded libraries: {e}")
        
        return libraries
    
    async def _create_dependency_component(self, dependency: Dict[str, Any]) -> Component:
        """Create component from dependency information"""
        component_id = self._generate_component_id(dependency["name"], dependency["version"])
        
        # Determine component type based on dependency type
        type_mapping = {
            "npm": ComponentType.LIBRARY,
            "pip": ComponentType.LIBRARY,
            "maven": ComponentType.LIBRARY,
            "gradle": ComponentType.LIBRARY,
            "gem": ComponentType.LIBRARY,
            "composer": ComponentType.LIBRARY
        }
        
        component_type = type_mapping.get(dependency["type"], ComponentType.LIBRARY)
        
        # Generate PURL
        purl = self._generate_purl(dependency)
        
        return Component(
            component_id=component_id,
            name=dependency["name"],
            version=dependency["version"],
            component_type=component_type,
            supplier=None,
            author=None,
            publisher=None,
            namespace=None,
            description=f"{dependency['type']} dependency",
            scope=dependency.get("scope", "required"),
            hashes={},
            licenses=[],
            license_type=LicenseType.UNKNOWN,
            copyright=None,
            cpe=None,
            purl=purl,
            external_references=[],
            dependencies=[],
            vulnerabilities=[],
            created_at=datetime.now(),
            modified_at=datetime.now()
        )
    
    async def _create_library_component(self, library: Dict[str, Any]) -> Component:
        """Create component from embedded library information"""
        component_id = self._generate_component_id(library["name"], library["version"])
        
        return Component(
            component_id=component_id,
            name=library["name"],
            version=library["version"],
            component_type=ComponentType.LIBRARY,
            supplier=None,
            author=None,
            publisher=None,
            namespace=None,
            description="Embedded library",
            scope="required",
            hashes={},
            licenses=[],
            license_type=LicenseType.UNKNOWN,
            copyright=None,
            cpe=None,
            purl=f"pkg:generic/{library['name']}@{library['version']}",
            external_references=[],
            dependencies=[],
            vulnerabilities=[],
            created_at=datetime.now(),
            modified_at=datetime.now()
        )
    
    def _generate_purl(self, dependency: Dict[str, Any]) -> str:
        """Generate Package URL (PURL) for dependency"""
        type_mapping = {
            "npm": "npm",
            "pip": "pypi",
            "maven": "maven",
            "gradle": "maven",
            "gem": "gem",
            "composer": "composer"
        }
        
        purl_type = type_mapping.get(dependency["type"], "generic")
        name = dependency["name"]
        version = dependency["version"]
        
        if purl_type == "maven" and ":" in name:
            namespace, name = name.split(":", 1)
            return f"pkg:{purl_type}/{namespace}/{name}@{version}"
        else:
            return f"pkg:{purl_type}/{name}@{version}"
    
    async def _calculate_file_hashes(self, file_path: str) -> Dict[str, str]:
        """Calculate file hashes"""
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Calculate multiple hash algorithms
            hashes["sha1"] = hashlib.sha1(content).hexdigest()
            hashes["sha256"] = hashlib.sha256(content).hexdigest()
            hashes["md5"] = hashlib.md5(content).hexdigest()
        
        except Exception as e:
            print(f"Failed to calculate hashes for {file_path}: {e}")
        
        return hashes
    
    async def _build_component_relationships(self, components: List[Component]) -> List[Dict[str, str]]:
        """Build relationships between components"""
        relationships = []
        
        # Find main component (application type)
        main_component = None
        for component in components:
            if component.component_type == ComponentType.APPLICATION:
                main_component = component
                break
        
        if main_component:
            # Create DEPENDS_ON relationships
            for component in components:
                if component.component_id != main_component.component_id:
                    relationships.append({
                        "spdx_element_id": main_component.component_id,
                        "relationship_type": "DEPENDS_ON",
                        "related_spdx_element": component.component_id
                    })
        
        return relationships
    
    async def _enhance_component_analysis(self, sbom: SBOM):
        """Enhance component analysis with additional data"""
        if self.config["component_analysis"]["vulnerability_check"]:
            await self._check_component_vulnerabilities(sbom)
        
        if self.config["component_analysis"]["license_compliance"]:
            await self._analyze_component_licenses(sbom)
        
        if self.config["component_analysis"]["dependency_analysis"]:
            await self._analyze_dependency_relationships(sbom)
    
    async def _check_component_vulnerabilities(self, sbom: SBOM):
        """Check components for known vulnerabilities"""
        for component in sbom.components:
            # In production, query vulnerability databases
            # For now, simulate vulnerability check
            if "vulnerable" in component.name.lower():
                component.vulnerabilities.append("CVE-2023-12345")
    
    async def _analyze_component_licenses(self, sbom: SBOM):
        """Analyze component licenses"""
        for component in sbom.components:
            # In production, query license databases
            # For now, assign common licenses based on component type
            if component.component_type == ComponentType.LIBRARY:
                component.licenses = ["MIT"]
                component.license_type = LicenseType.PERMISSIVE
    
    async def _analyze_dependency_relationships(self, sbom: SBOM):
        """Analyze dependency relationships"""
        # Build dependency graph
        dependency_graph = {}
        
        for component in sbom.components:
            dependency_graph[component.component_id] = component.dependencies
        
        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies(dependency_graph)
        if circular_deps:
            sbom.annotations.append({
                "annotator": "SBOM Manager",
                "annotation_type": "WARNING",
                "annotation_comment": f"Circular dependencies detected: {circular_deps}",
                "annotation_date": datetime.now().isoformat()
            })
    
    def _detect_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Detect circular dependencies in dependency graph"""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if dfs(neighbor):
                    cycles.append(node)
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dependency_graph:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def _generate_component_id(self, name: str, version: str) -> str:
        """Generate unique component ID"""
        content = f"{name}_{version}"
        return f"SPDXRef-{hashlib.sha256(content.encode()).hexdigest()[:16]}"
    
    def _generate_sbom_id(self, vendor_id: str, software_name: str, software_version: str) -> str:
        """Generate unique SBOM ID"""
        content = f"{vendor_id}_{software_name}_{software_version}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def export_sbom(self, sbom_id: str, format: SBOMFormat, output_path: str) -> bool:
        """Export SBOM to specified format"""
        if sbom_id not in self.sbom_storage:
            return False
        
        sbom = self.sbom_storage[sbom_id]
        
        try:
            if format == SBOMFormat.SPDX_JSON:
                await self._export_spdx_json(sbom, output_path)
            elif format == SBOMFormat.SPDX_XML:
                await self._export_spdx_xml(sbom, output_path)
            elif format == SBOMFormat.CYCLONE_DX_JSON:
                await self._export_cyclonedx_json(sbom, output_path)
            elif format == SBOMFormat.CYCLONE_DX_XML:
                await self._export_cyclonedx_xml(sbom, output_path)
            else:
                return False
            
            return True
        
        except Exception as e:
            print(f"Failed to export SBOM: {e}")
            return False
    
    async def _export_spdx_json(self, sbom: SBOM, output_path: str):
        """Export SBOM in SPDX JSON format"""
        spdx_data = {
            "spdxVersion": f"SPDX-{sbom.spec_version}",
            "dataLicense": sbom.data_license,
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": sbom.name,
            "documentNamespace": sbom.document_namespace,
            "creationInfo": sbom.creation_info,
            "packages": [],
            "relationships": sbom.relationships,
            "annotations": sbom.annotations
        }
        
        # Convert components to SPDX packages
        for component in sbom.components:
            package = {
                "SPDXID": component.component_id,
                "name": component.name,
                "versionInfo": component.version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": component.copyright or "NOASSERTION",
                "externalRefs": component.external_references,
                "checksums": [
                    {"algorithm": algo.upper(), "checksumValue": value}
                    for algo, value in component.hashes.items()
                ]
            }
            
            if component.supplier:
                package["supplier"] = f"Organization: {component.supplier}"
            
            spdx_data["packages"].append(package)
        
        with open(output_path, 'w') as f:
            json.dump(spdx_data, f, indent=2, default=str)
    
    async def _export_spdx_xml(self, sbom: SBOM, output_path: str):
        """Export SBOM in SPDX XML format"""
        # Create XML structure
        root = ET.Element("spdx:SpdxDocument")
        root.set("xmlns:spdx", "http://spdx.org/rdf/terms#")
        
        # Add document information
        doc_info = ET.SubElement(root, "spdx:creationInfo")
        ET.SubElement(doc_info, "spdx:created").text = sbom.creation_info["created"]
        
        # Add packages
        for component in sbom.components:
            package = ET.SubElement(root, "spdx:Package")
            package.set("rdf:about", component.component_id)
            ET.SubElement(package, "spdx:name").text = component.name
            ET.SubElement(package, "spdx:versionInfo").text = component.version
        
        # Write XML to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    async def _export_cyclonedx_json(self, sbom: SBOM, output_path: str):
        """Export SBOM in CycloneDX JSON format"""
        cyclonedx_data = {
            "bomFormat": "CycloneDX",
            "specVersion": self.config["spec_versions"]["cyclonedx"],
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "tools": [
                    {
                        "vendor": "ScrollIntel",
                        "name": "SBOM Manager",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "name": sbom.software_name or sbom.name,
                    "version": sbom.version
                }
            },
            "components": []
        }
        
        # Convert components to CycloneDX format
        for component in sbom.components:
            if component.component_type != ComponentType.APPLICATION:  # Skip main component
                cyclonedx_component = {
                    "type": component.component_type.value,
                    "name": component.name,
                    "version": component.version,
                    "scope": component.scope,
                    "hashes": [
                        {"alg": algo.upper(), "content": value}
                        for algo, value in component.hashes.items()
                    ],
                    "licenses": [{"license": {"name": license}} for license in component.licenses],
                    "purl": component.purl
                }
                
                cyclonedx_data["components"].append(cyclonedx_component)
        
        with open(output_path, 'w') as f:
            json.dump(cyclonedx_data, f, indent=2, default=str)
    
    async def _export_cyclonedx_xml(self, sbom: SBOM, output_path: str):
        """Export SBOM in CycloneDX XML format"""
        # Create XML structure
        root = ET.Element("bom")
        root.set("xmlns", "http://cyclonedx.org/schema/bom/1.4")
        root.set("serialNumber", f"urn:uuid:{uuid.uuid4()}")
        root.set("version", "1")
        
        # Add metadata
        metadata = ET.SubElement(root, "metadata")
        timestamp = ET.SubElement(metadata, "timestamp")
        timestamp.text = datetime.now().isoformat()
        
        # Add components
        components = ET.SubElement(root, "components")
        for component in sbom.components:
            if component.component_type != ComponentType.APPLICATION:
                comp_elem = ET.SubElement(components, "component")
                comp_elem.set("type", component.component_type.value)
                ET.SubElement(comp_elem, "name").text = component.name
                ET.SubElement(comp_elem, "version").text = component.version
        
        # Write XML to file
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    async def compare_sboms(self, sbom_id1: str, sbom_id2: str) -> Dict[str, Any]:
        """Compare two SBOMs and identify differences"""
        if sbom_id1 not in self.sbom_storage or sbom_id2 not in self.sbom_storage:
            return {"error": "One or both SBOMs not found"}
        
        sbom1 = self.sbom_storage[sbom_id1]
        sbom2 = self.sbom_storage[sbom_id2]
        
        # Compare components
        components1 = {(c.name, c.version): c for c in sbom1.components}
        components2 = {(c.name, c.version): c for c in sbom2.components}
        
        added = set(components2.keys()) - set(components1.keys())
        removed = set(components1.keys()) - set(components2.keys())
        common = set(components1.keys()) & set(components2.keys())
        
        modified = []
        for key in common:
            comp1, comp2 = components1[key], components2[key]
            if comp1.hashes != comp2.hashes or comp1.licenses != comp2.licenses:
                modified.append(key)
        
        return {
            "comparison_timestamp": datetime.now().isoformat(),
            "sbom1": {"id": sbom_id1, "name": sbom1.name},
            "sbom2": {"id": sbom_id2, "name": sbom2.name},
            "summary": {
                "total_components_sbom1": len(components1),
                "total_components_sbom2": len(components2),
                "added_components": len(added),
                "removed_components": len(removed),
                "modified_components": len(modified)
            },
            "details": {
                "added": list(added),
                "removed": list(removed),
                "modified": list(modified)
            }
        }
    
    async def get_sbom_analytics(self, sbom_id: str) -> Dict[str, Any]:
        """Get analytics for SBOM"""
        if sbom_id not in self.sbom_storage:
            return {"error": "SBOM not found"}
        
        sbom = self.sbom_storage[sbom_id]
        
        # Component type distribution
        type_distribution = {}
        for component in sbom.components:
            comp_type = component.component_type.value
            type_distribution[comp_type] = type_distribution.get(comp_type, 0) + 1
        
        # License distribution
        license_distribution = {}
        for component in sbom.components:
            for license in component.licenses:
                license_distribution[license] = license_distribution.get(license, 0) + 1
        
        # Vulnerability summary
        vulnerable_components = [c for c in sbom.components if c.vulnerabilities]
        total_vulnerabilities = sum(len(c.vulnerabilities) for c in sbom.components)
        
        return {
            "sbom_id": sbom_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_components": len(sbom.components),
                "vulnerable_components": len(vulnerable_components),
                "total_vulnerabilities": total_vulnerabilities,
                "unique_licenses": len(license_distribution)
            },
            "distributions": {
                "component_types": type_distribution,
                "licenses": license_distribution
            },
            "risk_assessment": {
                "vulnerability_risk": "high" if total_vulnerabilities > 10 else "medium" if total_vulnerabilities > 0 else "low",
                "license_risk": self._assess_license_risk(license_distribution)
            }
        }
    
    def _assess_license_risk(self, license_distribution: Dict[str, int]) -> str:
        """Assess license compliance risk"""
        high_risk_licenses = ["GPL-3.0", "AGPL-3.0", "SSPL-1.0"]
        
        for license in license_distribution:
            if license in high_risk_licenses:
                return "high"
        
        return "low"