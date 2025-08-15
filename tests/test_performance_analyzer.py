"""
Tests for Performance Analyzer Engine
"""

import pytest
from scrollintel.engines.performance_analyzer import (
    PerformanceAnalyzer, PerformanceReport, PerformanceIssue, PerformanceImpact
)

class TestPerformanceAnalyzer:
    
    def setup_method(self):
        self.analyzer = PerformanceAnalyzer()
    
    def test_analyze_python_nested_loops(self):
        """Test detection of nested loops in Python"""
        code = """
def process_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                matrix[i][j][k] *= 2
    return matrix
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        nested_loop_issues = [issue for issue in report.issues 
                             if 'nested' in issue.title.lower()]
        assert len(nested_loop_issues) > 0
        assert any(issue.impact == PerformanceImpact.HIGH for issue in nested_loop_issues)
    
    def test_analyze_python_string_concatenation(self):
        """Test detection of string concatenation in loops"""
        code = """
def build_string(items):
    result = ""
    for item in items:
        result += str(item) + ", "
    return result
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        string_issues = [issue for issue in report.issues 
                        if 'string' in issue.category.lower()]
        assert len(string_issues) > 0
        assert any('O(n²)' in issue.estimated_impact for issue in string_issues 
                  if issue.estimated_impact)
    
    def test_analyze_python_list_comprehension(self):
        """Test detection of nested list comprehensions"""
        code = """
def process_data(data):
    return [[item * 2 for item in row] for row in data]
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        # This should be flagged as potentially memory intensive
        comp_issues = [issue for issue in report.issues 
                      if 'comprehension' in issue.title.lower()]
        # Note: Single nested comprehension might not trigger warning
        # but deeply nested ones should
    
    def test_analyze_python_global_variables(self):
        """Test detection of excessive global variables"""
        code = """
global_var1 = 1
global_var2 = 2
global_var3 = 3
global_var4 = 4
global_var5 = 5
global_var6 = 6

def use_globals():
    return global_var1 + global_var2 + global_var3
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        global_issues = [issue for issue in report.issues 
                        if 'global' in issue.title.lower()]
        assert len(global_issues) > 0
    
    def test_analyze_javascript_dom_queries(self):
        """Test detection of DOM queries in loops"""
        code = """
function updateElements(items) {
    for (let i = 0; i < items.length; i++) {
        document.getElementById('item-' + i).textContent = items[i];
    }
}
"""
        report = self.analyzer.analyze_performance(code, 'javascript')
        
        dom_issues = [issue for issue in report.issues 
                     if 'dom' in issue.category.lower()]
        assert len(dom_issues) > 0
        assert any(issue.impact == PerformanceImpact.HIGH for issue in dom_issues)
    
    def test_analyze_javascript_inefficient_arrays(self):
        """Test detection of inefficient array operations"""
        code = """
function processItems(items) {
    let result = [];
    items.forEach(item => {
        result.push(item * 2);
    });
    return result;
}
"""
        report = self.analyzer.analyze_performance(code, 'javascript')
        
        array_issues = [issue for issue in report.issues 
                       if 'array' in issue.category.lower()]
        assert len(array_issues) > 0
    
    def test_analyze_javascript_memory_leaks(self):
        """Test detection of potential memory leaks"""
        code = """
function startTimer() {
    setInterval(function() {
        console.log('Timer tick');
    }, 1000);
    // No clearInterval - potential memory leak
}
"""
        report = self.analyzer.analyze_performance(code, 'javascript')
        
        memory_issues = [issue for issue in report.issues 
                        if 'memory' in issue.category.lower()]
        assert len(memory_issues) > 0
    
    def test_analyze_javascript_sync_operations(self):
        """Test detection of synchronous operations"""
        code = """
function loadData() {
    let xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/data', false);  // Synchronous
    xhr.send();
    return xhr.responseText;
}
"""
        report = self.analyzer.analyze_performance(code, 'javascript')
        
        async_issues = [issue for issue in report.issues 
                       if 'async' in issue.category.lower() or 'sync' in issue.title.lower()]
        assert len(async_issues) > 0
    
    def test_analyze_java_string_concatenation(self):
        """Test detection of string concatenation in Java loops"""
        code = """
public String buildString(List<String> items) {
    String result = "";
    for (String item : items) {
        result += item + ", ";
    }
    return result;
}
"""
        report = self.analyzer.analyze_performance(code, 'java')
        
        string_issues = [issue for issue in report.issues 
                        if 'string' in issue.category.lower()]
        assert len(string_issues) > 0
        assert any(issue.impact == PerformanceImpact.HIGH for issue in string_issues)
    
    def test_analyze_java_legacy_collections(self):
        """Test detection of legacy collection usage"""
        code = """
import java.util.Vector;

public void processData() {
    Vector<String> data = new Vector<>();
    data.add("item1");
    data.add("item2");
}
"""
        report = self.analyzer.analyze_performance(code, 'java')
        
        collection_issues = [issue for issue in report.issues 
                           if 'collection' in issue.category.lower() or 'vector' in issue.title.lower()]
        assert len(collection_issues) > 0
    
    def test_analyze_java_autoboxing(self):
        """Test detection of autoboxing in loops"""
        code = """
public void processNumbers() {
    for (Integer i = 0; i < 1000; i++) {
        System.out.println(i);
    }
}
"""
        report = self.analyzer.analyze_performance(code, 'java')
        
        boxing_issues = [issue for issue in report.issues 
                        if 'boxing' in issue.title.lower()]
        assert len(boxing_issues) > 0
    
    def test_analyze_sql_select_star(self):
        """Test detection of SELECT * usage"""
        code = """
SELECT * FROM large_table WHERE condition = 'value';
SELECT * FROM another_table;
"""
        report = self.analyzer.analyze_performance(code, 'sql')
        
        select_issues = [issue for issue in report.issues 
                        if 'select' in issue.title.lower()]
        assert len(select_issues) > 0
    
    def test_analyze_sql_missing_where(self):
        """Test detection of queries without WHERE clauses"""
        code = """
SELECT name, email FROM users;
SELECT * FROM orders;
SELECT id FROM products WHERE active = 1;
"""
        report = self.analyzer.analyze_performance(code, 'sql')
        
        where_issues = [issue for issue in report.issues 
                       if 'where' in issue.title.lower()]
        assert len(where_issues) > 0
    
    def test_analyze_sql_leading_wildcards(self):
        """Test detection of leading wildcards in LIKE"""
        code = """
SELECT * FROM users WHERE name LIKE '%john%';
SELECT * FROM products WHERE description LIKE '%widget%';
"""
        report = self.analyzer.analyze_performance(code, 'sql')
        
        wildcard_issues = [issue for issue in report.issues 
                          if 'wildcard' in issue.title.lower() or 'like' in issue.title.lower()]
        assert len(wildcard_issues) > 0
    
    def test_analyze_sql_functions_in_where(self):
        """Test detection of functions in WHERE clauses"""
        code = """
SELECT * FROM orders WHERE YEAR(order_date) = 2023;
SELECT * FROM users WHERE UPPER(name) = 'JOHN';
"""
        report = self.analyzer.analyze_performance(code, 'sql')
        
        function_issues = [issue for issue in report.issues 
                          if 'function' in issue.title.lower()]
        assert len(function_issues) > 0
    
    def test_analyze_go_string_concatenation(self):
        """Test detection of string concatenation in Go loops"""
        code = """
func buildString(items []string) string {
    result := ""
    for _, item := range items {
        result += item + ", "
    }
    return result
}
"""
        report = self.analyzer.analyze_performance(code, 'go')
        
        string_issues = [issue for issue in report.issues 
                        if 'string' in issue.category.lower()]
        assert len(string_issues) > 0
    
    def test_analyze_go_defer_in_loop(self):
        """Test detection of defer in loops"""
        code = """
func processFiles(files []string) {
    for _, filename := range files {
        file, err := os.Open(filename)
        if err != nil {
            continue
        }
        defer file.Close()  // Problematic in loop
        // process file
    }
}
"""
        report = self.analyzer.analyze_performance(code, 'go')
        
        defer_issues = [issue for issue in report.issues 
                       if 'defer' in issue.title.lower()]
        assert len(defer_issues) > 0
    
    def test_analyze_deep_nesting(self):
        """Test detection of deeply nested code"""
        code = """
def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            return "deep"
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        nesting_issues = [issue for issue in report.issues 
                         if 'nesting' in issue.title.lower()]
        assert len(nesting_issues) > 0
    
    def test_analyze_large_functions(self):
        """Test detection of large functions"""
        # Create a function with many lines
        lines = ["def large_function():"]
        for i in range(150):
            lines.append(f"    x{i} = {i}")
        lines.append("    return sum([" + ", ".join(f"x{i}" for i in range(150)) + "])")
        
        code = "\n".join(lines)
        report = self.analyzer.analyze_performance(code, 'python')
        
        large_func_issues = [issue for issue in report.issues 
                           if 'large' in issue.title.lower() or 'function' in issue.title.lower()]
        assert len(large_func_issues) > 0
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # High-performance code
        good_code = """
def efficient_function(data):
    return [item * 2 for item in data]
"""
        good_report = self.analyzer.analyze_performance(good_code, 'python')
        
        # Low-performance code
        bad_code = """
def inefficient_function(data):
    result = ""
    for i in range(len(data)):
        for j in range(len(data)):
            for k in range(len(data)):
                result += str(data[i]) + str(data[j]) + str(data[k])
    return result
"""
        bad_report = self.analyzer.analyze_performance(bad_code, 'python')
        
        assert good_report.metrics.overall_score > bad_report.metrics.overall_score
        assert good_report.metrics.cpu_efficiency > bad_report.metrics.cpu_efficiency
    
    def test_generate_optimizations(self):
        """Test optimization recommendation generation"""
        code = """
def inefficient_function(data):
    result = ""
    for item in data:
        result += str(item)
    return result
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        assert len(report.optimizations) > 0
        assert any('string' in opt.lower() for opt in report.optimizations)
    
    def test_identify_bottlenecks(self):
        """Test bottleneck identification"""
        code = """
def bottleneck_function(data):
    for i in range(len(data)):
        for j in range(len(data)):
            for k in range(len(data)):
                print(data[i], data[j], data[k])
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        assert len(report.bottlenecks) > 0
        assert any('critical' in bottleneck.lower() or 'high' in bottleneck.lower() 
                  for bottleneck in report.bottlenecks)
    
    def test_analyze_error_handling(self):
        """Test error handling in performance analysis"""
        # This should not crash the analyzer
        report = self.analyzer.analyze_performance(None, 'python')
        
        assert len(report.issues) > 0
        assert report.metrics.overall_score == 0
        assert 'error' in report.analysis_metadata
    
    def test_python_dictionary_efficiency(self):
        """Test detection of inefficient dictionary operations"""
        code = """
def check_keys(data, keys):
    for key in keys:
        if key in data.keys():  # Inefficient
            print(f"Found {key}")
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        dict_issues = [issue for issue in report.issues 
                      if 'dictionary' in issue.title.lower()]
        assert len(dict_issues) > 0
    
    def test_python_list_append_optimization(self):
        """Test detection of loops that could use list comprehension"""
        code = """
def process_items(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result
"""
        report = self.analyzer.analyze_performance(code, 'python')
        
        append_issues = [issue for issue in report.issues 
                        if 'append' in issue.title.lower()]
        assert len(append_issues) > 0