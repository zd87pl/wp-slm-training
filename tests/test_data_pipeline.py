#!/usr/bin/env python3
"""
Tests for data pipeline components.
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.parse_wp_docs import WPDocsParser
from scripts.build_sft_pairs import SFTBuilder, InstructionPair


class TestDataPipeline(unittest.TestCase):
    """Test data processing pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
        
    def test_wp_docs_parser_init(self):
        """Test WPDocsParser initialization."""
        parser = WPDocsParser(
            input_dir=self.test_dir / "raw",
            output_dir=self.test_dir / "processed"
        )
        self.assertTrue((self.test_dir / "processed").exists())
        
    def test_markdown_cleaning(self):
        """Test markdown cleaning functionality."""
        parser = WPDocsParser()
        
        # Test excessive newlines
        dirty = "Title\n\n\n\nContent"
        clean = parser._clean_markdown(dirty)
        self.assertEqual(clean, "Title\n\nContent")
        
        # Test HTML comments
        dirty = "Text <!-- comment --> more text"
        clean = parser._clean_markdown(dirty)
        self.assertEqual(clean, "Text  more text")
        
        # Test shortcodes
        dirty = "Text [shortcode] more [/shortcode] text"
        clean = parser._clean_markdown(dirty)
        self.assertEqual(clean, "Text  more  text")
        
    def test_sft_builder_init(self):
        """Test SFTBuilder initialization."""
        builder = SFTBuilder(
            input_dir=self.test_dir / "processed",
            output_dir=self.test_dir / "sft"
        )
        self.assertTrue((self.test_dir / "sft").exists())
        self.assertIsNotNone(builder.prompt_templates)
        
    def test_section_type_detection(self):
        """Test section type detection."""
        builder = SFTBuilder()
        
        # Test function reference
        section = {
            'title': 'wp_enqueue_script() Function',
            'content': 'This function enqueues scripts...'
        }
        self.assertEqual(builder._detect_section_type(section), "reference")
        
        # Test hook detection
        section = {
            'title': 'init Hook',
            'content': 'The init action hook...'
        }
        self.assertEqual(builder._detect_section_type(section), "hook")
        
        # Test API detection
        section = {
            'title': 'REST API Endpoints',
            'content': 'WordPress REST API...'
        }
        self.assertEqual(builder._detect_section_type(section), "api")
        
    def test_instruction_pair_creation(self):
        """Test instruction pair object creation."""
        pair = InstructionPair(
            id="test-001",
            source_url="https://example.com",
            license="GPL-2.0+",
            prompt="How do I create a custom post type?",
            response="To create a custom post type...",
            tags=["howto", "cpt"],
            wp_version="6.5",
            meta={"difficulty": "intermediate"}
        )
        
        # Convert to dict and check fields
        pair_dict = pair.__dict__
        self.assertEqual(pair_dict['id'], "test-001")
        self.assertEqual(pair_dict['license'], "GPL-2.0+")
        self.assertIn("howto", pair_dict['tags'])
        
    def test_extract_action_from_title(self):
        """Test action extraction from titles."""
        builder = SFTBuilder()
        
        # Test with prefix
        title = "How to Create Custom Post Types"
        action = builder._extract_action_from_title(title)
        self.assertEqual(action, "create custom post types")
        
        # Test without prefix
        title = "Adding Custom Fields"
        action = builder._extract_action_from_title(title)
        self.assertEqual(action, "adding custom fields")
        
    def test_response_cleaning(self):
        """Test response text cleaning."""
        builder = SFTBuilder()
        
        # Test excessive newlines
        response = "Line 1\n\n\n\nLine 2"
        clean = builder._clean_response(response)
        self.assertEqual(clean, "Line 1\n\nLine 2")
        
        # Test length trimming
        long_response = "x" * 5000
        clean = builder._clean_response(long_response)
        self.assertLessEqual(len(clean), 4000)


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create test structure
        for subdir in ['raw', 'processed', 'sft', 'prefs', 'eval']:
            (self.test_dir / subdir).mkdir(parents=True)
            
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
        
    def test_directory_validation(self):
        """Test directory structure validation."""
        from scripts.validate_data import DataValidator
        
        validator = DataValidator(self.test_dir)
        self.assertTrue(validator._validate_directory_structure())
        
    def test_jsonl_validation(self):
        """Test JSONL format validation."""
        # Create valid JSONL file
        valid_data = [
            {"prompt": "Test prompt", "response": "Test response"},
            {"prompt": "Another prompt", "response": "Another response"}
        ]
        
        sft_file = self.test_dir / "sft" / "train.jsonl"
        with sft_file.open('w') as f:
            for item in valid_data:
                f.write(json.dumps(item) + '\n')
                
        from scripts.validate_data import DataValidator
        validator = DataValidator(self.test_dir)
        
        # Should not add errors for valid data
        initial_errors = len(validator.errors)
        validator._validate_sft_data()
        self.assertEqual(len(validator.errors), initial_errors)


if __name__ == "__main__":
    unittest.main()