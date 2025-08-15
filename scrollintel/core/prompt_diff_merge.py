"""
Advanced diff and merge capabilities for prompt content.
"""
import difflib
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ConflictType(Enum):
    """Types of merge conflicts."""
    CONTENT = "content"
    VARIABLE = "variable"
    TAG = "tag"
    METADATA = "metadata"


@dataclass
class ConflictRegion:
    """Represents a conflict region in a merge."""
    type: ConflictType
    start_line: int
    end_line: int
    current_content: str
    incoming_content: str
    description: str


@dataclass
class MergeResult:
    """Result of a merge operation."""
    success: bool
    merged_content: str
    conflicts: List[ConflictRegion]
    merged_variables: List[Dict[str, Any]]
    merged_tags: List[str]
    warnings: List[str]


class PromptDiffMerge:
    """Advanced diff and merge operations for prompts."""
    
    def __init__(self):
        self.variable_pattern = re.compile(r'\{\{(\w+)\}\}')
        self.section_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
    
    def generate_unified_diff(self, content1: str, content2: str, 
                            filename1: str = "version1", 
                            filename2: str = "version2") -> List[str]:
        """Generate unified diff between two prompt contents."""
        return list(difflib.unified_diff(
            content1.splitlines(keepends=True),
            content2.splitlines(keepends=True),
            fromfile=filename1,
            tofile=filename2,
            lineterm=""
        ))
    
    def generate_side_by_side_diff(self, content1: str, content2: str) -> Dict[str, Any]:
        """Generate side-by-side diff for better visualization."""
        lines1 = content1.splitlines()
        lines2 = content2.splitlines()
        
        differ = difflib.SequenceMatcher(None, lines1, lines2)
        diff_data = []
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'equal':
                for i, line in enumerate(lines1[i1:i2]):
                    diff_data.append({
                        'type': 'equal',
                        'line_num1': i1 + i + 1,
                        'line_num2': j1 + i + 1,
                        'content1': line,
                        'content2': lines2[j1 + i] if j1 + i < len(lines2) else line
                    })
            elif tag == 'delete':
                for i, line in enumerate(lines1[i1:i2]):
                    diff_data.append({
                        'type': 'delete',
                        'line_num1': i1 + i + 1,
                        'line_num2': None,
                        'content1': line,
                        'content2': ''
                    })
            elif tag == 'insert':
                for i, line in enumerate(lines2[j1:j2]):
                    diff_data.append({
                        'type': 'insert',
                        'line_num1': None,
                        'line_num2': j1 + i + 1,
                        'content1': '',
                        'content2': line
                    })
            elif tag == 'replace':
                max_lines = max(i2 - i1, j2 - j1)
                for i in range(max_lines):
                    content1 = lines1[i1 + i] if i1 + i < i2 else ''
                    content2 = lines2[j1 + i] if j1 + i < j2 else ''
                    diff_data.append({
                        'type': 'replace',
                        'line_num1': i1 + i + 1 if i1 + i < i2 else None,
                        'line_num2': j1 + i + 1 if j1 + i < j2 else None,
                        'content1': content1,
                        'content2': content2
                    })
        
        return {
            'diff_data': diff_data,
            'stats': {
                'additions': sum(1 for d in diff_data if d['type'] in ['insert', 'replace'] and d['content2']),
                'deletions': sum(1 for d in diff_data if d['type'] in ['delete', 'replace'] and d['content1']),
                'modifications': sum(1 for d in diff_data if d['type'] == 'replace')
            }
        }
    
    def analyze_semantic_changes(self, content1: str, content2: str) -> Dict[str, Any]:
        """Analyze semantic changes between prompt versions."""
        # Extract variables
        vars1 = set(self.variable_pattern.findall(content1))
        vars2 = set(self.variable_pattern.findall(content2))
        
        # Extract sections (markdown headers)
        sections1 = set(self.section_pattern.findall(content1))
        sections2 = set(self.section_pattern.findall(content2))
        
        # Analyze instruction changes
        instruction_keywords = ['must', 'should', 'will', 'always', 'never', 'required', 'optional']
        instructions1 = self._extract_instructions(content1, instruction_keywords)
        instructions2 = self._extract_instructions(content2, instruction_keywords)
        
        return {
            'variables': {
                'added': list(vars2 - vars1),
                'removed': list(vars1 - vars2),
                'unchanged': list(vars1 & vars2)
            },
            'sections': {
                'added': list(sections2 - sections1),
                'removed': list(sections1 - sections2),
                'unchanged': list(sections1 & sections2)
            },
            'instructions': {
                'added': [inst for inst in instructions2 if inst not in instructions1],
                'removed': [inst for inst in instructions1 if inst not in instructions2],
                'modified': self._find_modified_instructions(instructions1, instructions2)
            },
            'complexity_change': self._calculate_complexity_change(content1, content2)
        }
    
    def three_way_merge(self, base_content: str, current_content: str, 
                       incoming_content: str) -> MergeResult:
        """Perform three-way merge with conflict detection."""
        conflicts = []
        warnings = []
        
        # Split content into lines for processing
        base_lines = base_content.splitlines()
        current_lines = current_content.splitlines()
        incoming_lines = incoming_content.splitlines()
        
        # Perform three-way merge
        merged_lines = []
        i = j = k = 0
        
        while i < len(base_lines) or j < len(current_lines) or k < len(incoming_lines):
            base_line = base_lines[i] if i < len(base_lines) else None
            current_line = current_lines[j] if j < len(current_lines) else None
            incoming_line = incoming_lines[k] if k < len(incoming_lines) else None
            
            if base_line == current_line == incoming_line:
                # No changes
                if base_line is not None:
                    merged_lines.append(base_line)
                i += 1
                j += 1
                k += 1
            elif base_line == current_line and base_line != incoming_line:
                # Only incoming changed
                if incoming_line is not None:
                    merged_lines.append(incoming_line)
                i += 1
                j += 1
                k += 1
            elif base_line == incoming_line and base_line != current_line:
                # Only current changed
                if current_line is not None:
                    merged_lines.append(current_line)
                i += 1
                j += 1
                k += 1
            else:
                # Conflict detected
                conflict_start = len(merged_lines)
                
                # Add conflict markers
                merged_lines.append("<<<<<<< CURRENT")
                if current_line is not None:
                    merged_lines.append(current_line)
                merged_lines.append("=======")
                if incoming_line is not None:
                    merged_lines.append(incoming_line)
                merged_lines.append(">>>>>>> INCOMING")
                
                conflicts.append(ConflictRegion(
                    type=ConflictType.CONTENT,
                    start_line=conflict_start,
                    end_line=len(merged_lines) - 1,
                    current_content=current_line or "",
                    incoming_content=incoming_line or "",
                    description=f"Content conflict at line {conflict_start + 1}"
                ))
                
                i += 1
                j += 1
                k += 1
        
        merged_content = '\n'.join(merged_lines)
        
        # Merge variables and tags would be handled separately
        merged_variables = []
        merged_tags = []
        
        return MergeResult(
            success=len(conflicts) == 0,
            merged_content=merged_content,
            conflicts=conflicts,
            merged_variables=merged_variables,
            merged_tags=merged_tags,
            warnings=warnings
        )
    
    def auto_resolve_conflicts(self, merge_result: MergeResult, 
                              strategy: str = "smart") -> MergeResult:
        """Automatically resolve conflicts based on strategy."""
        if strategy == "smart":
            return self._smart_conflict_resolution(merge_result)
        elif strategy == "current":
            return self._resolve_with_current(merge_result)
        elif strategy == "incoming":
            return self._resolve_with_incoming(merge_result)
        else:
            return merge_result
    
    def generate_merge_preview(self, base_content: str, current_content: str,
                             incoming_content: str) -> Dict[str, Any]:
        """Generate a preview of what the merge would look like."""
        merge_result = self.three_way_merge(base_content, current_content, incoming_content)
        
        return {
            'preview_content': merge_result.merged_content,
            'conflicts': [
                {
                    'type': conflict.type.value,
                    'start_line': conflict.start_line,
                    'end_line': conflict.end_line,
                    'description': conflict.description,
                    'current': conflict.current_content,
                    'incoming': conflict.incoming_content
                }
                for conflict in merge_result.conflicts
            ],
            'auto_resolvable': len([c for c in merge_result.conflicts 
                                  if self._is_auto_resolvable(c)]),
            'manual_resolution_needed': len([c for c in merge_result.conflicts 
                                           if not self._is_auto_resolvable(c)])
        }
    
    def _extract_instructions(self, content: str, keywords: List[str]) -> List[str]:
        """Extract instruction sentences from content."""
        instructions = []
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords):
                instructions.append(sentence)
        
        return instructions
    
    def _find_modified_instructions(self, instructions1: List[str], 
                                  instructions2: List[str]) -> List[Dict[str, str]]:
        """Find instructions that were modified between versions."""
        modified = []
        
        # Simple similarity-based matching
        for inst1 in instructions1:
            best_match = None
            best_ratio = 0
            
            for inst2 in instructions2:
                ratio = difflib.SequenceMatcher(None, inst1, inst2).ratio()
                if ratio > best_ratio and ratio > 0.6:  # 60% similarity threshold
                    best_ratio = ratio
                    best_match = inst2
            
            if best_match and best_match != inst1:
                modified.append({
                    'original': inst1,
                    'modified': best_match,
                    'similarity': best_ratio
                })
        
        return modified
    
    def _calculate_complexity_change(self, content1: str, content2: str) -> Dict[str, Any]:
        """Calculate complexity change between versions."""
        # Simple complexity metrics
        metrics1 = self._calculate_complexity_metrics(content1)
        metrics2 = self._calculate_complexity_metrics(content2)
        
        return {
            'word_count_change': metrics2['word_count'] - metrics1['word_count'],
            'sentence_count_change': metrics2['sentence_count'] - metrics1['sentence_count'],
            'variable_count_change': metrics2['variable_count'] - metrics1['variable_count'],
            'complexity_score_change': metrics2['complexity_score'] - metrics1['complexity_score']
        }
    
    def _calculate_complexity_metrics(self, content: str) -> Dict[str, int]:
        """Calculate complexity metrics for content."""
        words = len(content.split())
        sentences = len(re.split(r'[.!?]+', content))
        variables = len(self.variable_pattern.findall(content))
        
        # Simple complexity score based on various factors
        complexity_score = (
            words * 0.1 +
            sentences * 0.5 +
            variables * 2 +
            len(re.findall(r'\b(if|when|unless|while|for|each)\b', content.lower())) * 3
        )
        
        return {
            'word_count': words,
            'sentence_count': sentences,
            'variable_count': variables,
            'complexity_score': int(complexity_score)
        }
    
    def _smart_conflict_resolution(self, merge_result: MergeResult) -> MergeResult:
        """Smart conflict resolution based on content analysis."""
        resolved_content = merge_result.merged_content
        remaining_conflicts = []
        
        for conflict in merge_result.conflicts:
            if self._is_auto_resolvable(conflict):
                # Apply smart resolution
                if len(conflict.incoming_content) > len(conflict.current_content):
                    # Prefer longer, more detailed content
                    resolution = conflict.incoming_content
                else:
                    resolution = conflict.current_content
                
                # Replace conflict markers with resolution
                conflict_section = resolved_content[conflict.start_line:conflict.end_line + 1]
                resolved_content = resolved_content.replace(conflict_section, resolution)
            else:
                remaining_conflicts.append(conflict)
        
        return MergeResult(
            success=len(remaining_conflicts) == 0,
            merged_content=resolved_content,
            conflicts=remaining_conflicts,
            merged_variables=merge_result.merged_variables,
            merged_tags=merge_result.merged_tags,
            warnings=merge_result.warnings
        )
    
    def _resolve_with_current(self, merge_result: MergeResult) -> MergeResult:
        """Resolve all conflicts by accepting current version."""
        resolved_content = merge_result.merged_content
        
        for conflict in merge_result.conflicts:
            # Replace conflict with current content
            conflict_pattern = f"<<<<<<< CURRENT\n{conflict.current_content}\n=======\n{conflict.incoming_content}\n>>>>>>> INCOMING"
            resolved_content = resolved_content.replace(conflict_pattern, conflict.current_content)
        
        return MergeResult(
            success=True,
            merged_content=resolved_content,
            conflicts=[],
            merged_variables=merge_result.merged_variables,
            merged_tags=merge_result.merged_tags,
            warnings=merge_result.warnings + ["All conflicts resolved by accepting current version"]
        )
    
    def _resolve_with_incoming(self, merge_result: MergeResult) -> MergeResult:
        """Resolve all conflicts by accepting incoming version."""
        resolved_content = merge_result.merged_content
        
        for conflict in merge_result.conflicts:
            # Replace conflict with incoming content
            conflict_pattern = f"<<<<<<< CURRENT\n{conflict.current_content}\n=======\n{conflict.incoming_content}\n>>>>>>> INCOMING"
            resolved_content = resolved_content.replace(conflict_pattern, conflict.incoming_content)
        
        return MergeResult(
            success=True,
            merged_content=resolved_content,
            conflicts=[],
            merged_variables=merge_result.merged_variables,
            merged_tags=merge_result.merged_tags,
            warnings=merge_result.warnings + ["All conflicts resolved by accepting incoming version"]
        )
    
    def _is_auto_resolvable(self, conflict: ConflictRegion) -> bool:
        """Determine if a conflict can be automatically resolved."""
        # Simple heuristics for auto-resolution
        current_len = len(conflict.current_content.strip())
        incoming_len = len(conflict.incoming_content.strip())
        
        # Auto-resolve if one side is empty
        if current_len == 0 or incoming_len == 0:
            return True
        
        # Auto-resolve if content is very similar
        similarity = difflib.SequenceMatcher(
            None, 
            conflict.current_content, 
            conflict.incoming_content
        ).ratio()
        
        return similarity > 0.8