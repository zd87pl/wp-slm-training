#!/usr/bin/env python3
"""
Tests for prompt templates and formatting.
"""

import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from inference.prompt_templates import (
    PromptTemplates, PromptType, format_prompt, extract_response
)


class TestPromptTemplates(unittest.TestCase):
    """Test prompt template functionality."""
    
    def test_prompt_type_detection(self):
        """Test automatic prompt type detection."""
        # Test API detection
        prompt = "How do I create a post using the WordPress REST API?"
        detected = PromptTemplates.detect_prompt_type(prompt)
        self.assertEqual(detected, PromptType.API)
        
        # Test troubleshooting detection
        prompt = "I'm getting a white screen error after updating my theme"
        detected = PromptTemplates.detect_prompt_type(prompt)
        self.assertEqual(detected, PromptType.TROUBLESHOOTING)
        
        # Test code generation detection
        prompt = "Give me a code snippet to register a custom taxonomy"
        detected = PromptTemplates.detect_prompt_type(prompt)
        self.assertEqual(detected, PromptType.CODE_GENERATION)
        
        # Test how-to detection
        prompt = "How do I add a custom menu to my theme?"
        detected = PromptTemplates.detect_prompt_type(prompt)
        self.assertEqual(detected, PromptType.HOWTO)
        
        # Test general fallback
        prompt = "Tell me about WordPress"
        detected = PromptTemplates.detect_prompt_type(prompt)
        self.assertEqual(detected, PromptType.GENERAL)
        
    def test_prompt_formatting(self):
        """Test prompt formatting with templates."""
        prompt = "How do I create a custom post type?"
        
        # Test with system prompt
        formatted = PromptTemplates.format_prompt(
            prompt, 
            PromptType.HOWTO,
            include_system=True
        )
        self.assertIn("WP-SLM", formatted)
        self.assertIn(prompt, formatted)
        self.assertTrue(formatted.endswith("ASSISTANT: "))
        
        # Test without system prompt
        formatted = PromptTemplates.format_prompt(
            prompt,
            PromptType.GENERAL,
            include_system=False
        )
        self.assertNotIn("WP-SLM", formatted)
        self.assertIn(prompt, formatted)
        
    def test_response_extraction(self):
        """Test response extraction from generated text."""
        # Test normal case
        generated = """USER: How do I create a post?
ASSISTANT: To create a post in WordPress, you can use the following code:

```php
$post_data = array(
    'post_title' => 'My Post',
    'post_content' => 'Post content',
    'post_status' => 'publish'
);
wp_insert_post($post_data);
```"""
        
        response = PromptTemplates.extract_response(generated)
        self.assertIn("To create a post", response)
        self.assertIn("wp_insert_post", response)
        self.assertNotIn("USER:", response)
        
        # Test multiple ASSISTANT markers
        generated = """ASSISTANT: First response
USER: Another question
ASSISTANT: Second response"""
        
        response = PromptTemplates.extract_response(generated)
        self.assertEqual(response.strip(), "Second response")
        
    def test_response_cleaning(self):
        """Test response cleaning functionality."""
        # Test incomplete sentence removal
        response = "This is complete. This is also complete. This is incompl"
        clean = PromptTemplates._clean_response(response)
        self.assertNotIn("incompl", clean)
        
        # Test unclosed code block
        response = "Here's code:\n```php\necho 'hello';"
        clean = PromptTemplates._clean_response(response)
        self.assertTrue(clean.endswith("```"))
        
        # Test excessive whitespace
        response = "Too    many     spaces"
        clean = PromptTemplates._clean_response(response)
        self.assertEqual(clean, "Too many spaces")
        
    def test_few_shot_prompt_creation(self):
        """Test few-shot prompt creation."""
        examples = [
            {
                "prompt": "How do I create a widget?",
                "response": "Use register_widget() function..."
            },
            {
                "prompt": "How do I add a menu?",
                "response": "Use register_nav_menus() function..."
            }
        ]
        
        new_prompt = "How do I create a shortcode?"
        
        few_shot = PromptTemplates.create_few_shot_prompt(examples, new_prompt)
        
        # Check all examples are included
        self.assertIn("How do I create a widget?", few_shot)
        self.assertIn("register_widget()", few_shot)
        self.assertIn("How do I add a menu?", few_shot)
        self.assertIn("register_nav_menus()", few_shot)
        
        # Check new prompt is at the end
        self.assertTrue(few_shot.endswith(f"USER: {new_prompt}\nASSISTANT: "))
        
    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        prompt = "How do I use the WordPress REST API?"
        
        # Test format_prompt function
        formatted = format_prompt(prompt)
        self.assertIn(prompt, formatted)
        self.assertIn("ASSISTANT:", formatted)
        
        # Test extract_response function
        generated = f"Some preamble\nASSISTANT: The REST API response"
        response = extract_response(generated)
        self.assertEqual(response, "The REST API response")


if __name__ == "__main__":
    unittest.main()