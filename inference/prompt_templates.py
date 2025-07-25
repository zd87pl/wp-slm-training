"""
Prompt templates for WordPress SLM inference.
These templates ensure consistency between training and inference.
"""

from typing import Dict, List, Optional
from enum import Enum

class PromptType(Enum):
    """Types of prompts supported by the model."""
    GENERAL = "general"
    HOWTO = "howto"
    REFERENCE = "reference"
    API = "api"
    HOOK = "hook"
    SECURITY = "security"
    TROUBLESHOOTING = "troubleshooting"
    CODE_GENERATION = "code_generation"


class PromptTemplates:
    """Manages prompt templates for WordPress SLM."""
    
    # System prompt that constrains the model to WordPress domain
    SYSTEM_PROMPT = """You are WP-SLM, a WordPress expert assistant. You ONLY answer WordPress-related questions. If a question is out of scope, say so briefly and offer to redirect.

When giving code:
- Prefer snippets that run in functions.php or a site-specific plugin
- Include version caveats when relevant (minimum WP version required)
- Validate and sanitize inputs
- Use WordPress coding standards
- Add security considerations where appropriate

Format your responses clearly with:
- Brief explanation
- Code examples in markdown blocks with language tags
- Any important notes or warnings"""

    # Base conversation template
    BASE_TEMPLATE = """{system_prompt}

USER: {prompt}
ASSISTANT: {response}"""

    # Specialized templates for different query types
    TEMPLATES = {
        PromptType.GENERAL: BASE_TEMPLATE,
        
        PromptType.HOWTO: """{system_prompt}

USER: {prompt}
ASSISTANT: I'll help you with that WordPress task. {response}""",
        
        PromptType.API: """{system_prompt}

USER: {prompt}
ASSISTANT: Here's how to work with the WordPress REST API for your request:

{response}""",
        
        PromptType.CODE_GENERATION: """{system_prompt}

USER: {prompt}
ASSISTANT: I'll provide you with the WordPress code you need:

{response}""",
        
        PromptType.TROUBLESHOOTING: """{system_prompt}

USER: {prompt}
ASSISTANT: Let me help you troubleshoot this WordPress issue:

{response}""",
    }

    @classmethod
    def format_prompt(cls, prompt: str, prompt_type: PromptType = PromptType.GENERAL,
                     include_system: bool = True) -> str:
        """Format a user prompt for inference."""
        template = cls.TEMPLATES.get(prompt_type, cls.BASE_TEMPLATE)
        
        if include_system:
            system = cls.SYSTEM_PROMPT
        else:
            system = ""
            
        # For inference, we only fill in the prompt part
        formatted = template.format(
            system_prompt=system,
            prompt=prompt,
            response=""  # Model will generate this
        )
        
        # Remove the trailing assistant marker so model can complete it
        if formatted.endswith("ASSISTANT: "):
            return formatted
        else:
            # Ensure it ends with "ASSISTANT:" for proper generation
            return formatted.rstrip() + "\nASSISTANT: "
            
    @classmethod
    def detect_prompt_type(cls, prompt: str) -> PromptType:
        """Detect the type of prompt based on content."""
        prompt_lower = prompt.lower()
        
        # Check for API-related keywords
        if any(keyword in prompt_lower for keyword in ['rest api', 'endpoint', 'wp-json', 'curl']):
            return PromptType.API
            
        # Check for troubleshooting keywords
        if any(keyword in prompt_lower for keyword in ['error', 'fix', 'debug', 'troubleshoot', 'problem']):
            return PromptType.TROUBLESHOOTING
            
        # Check for code generation requests
        if any(keyword in prompt_lower for keyword in ['code', 'snippet', 'function', 'example']):
            return PromptType.CODE_GENERATION
            
        # Check for how-to questions
        if any(keyword in prompt_lower for keyword in ['how do i', 'how to', 'how can', 'what\'s the best way']):
            return PromptType.HOWTO
            
        # Default to general
        return PromptType.GENERAL
        
    @classmethod
    def extract_response(cls, generated_text: str) -> str:
        """Extract the assistant's response from generated text."""
        # Find the last occurrence of "ASSISTANT:" and take everything after it
        parts = generated_text.split("ASSISTANT:")
        if len(parts) > 1:
            response = parts[-1].strip()
            
            # Clean up any repeated tokens or artifacts
            response = cls._clean_response(response)
            
            return response
        
        # Fallback: return cleaned version of entire text
        return cls._clean_response(generated_text)
        
    @staticmethod
    def _clean_response(response: str) -> str:
        """Clean up generated response."""
        # Remove any trailing incomplete sentences
        sentences = response.split('. ')
        if sentences and not sentences[-1].endswith(('.', '!', '?', '```')):
            # Last sentence might be incomplete
            if len(sentences) > 1:
                response = '. '.join(sentences[:-1]) + '.'
                
        # Remove excessive whitespace
        response = ' '.join(response.split())
        
        # Ensure code blocks are properly closed
        code_block_count = response.count('```')
        if code_block_count % 2 != 0:
            response += '\n```'
            
        return response.strip()
        
    @classmethod
    def create_few_shot_prompt(cls, examples: List[Dict[str, str]], 
                              new_prompt: str) -> str:
        """Create a few-shot prompt with examples."""
        formatted_examples = []
        
        for example in examples:
            formatted = cls.BASE_TEMPLATE.format(
                system_prompt="",  # No system prompt for examples
                prompt=example['prompt'],
                response=example['response']
            )
            formatted_examples.append(formatted)
            
        # Combine examples with new prompt
        few_shot = "\n\n".join(formatted_examples)
        few_shot += f"\n\nUSER: {new_prompt}\nASSISTANT: "
        
        return few_shot


# Convenience functions
def format_prompt(prompt: str, include_system: bool = True) -> str:
    """Format a user prompt for inference with auto-detected type."""
    prompt_type = PromptTemplates.detect_prompt_type(prompt)
    return PromptTemplates.format_prompt(prompt, prompt_type, include_system)


def extract_response(generated_text: str) -> str:
    """Extract the assistant's response from generated text."""
    return PromptTemplates.extract_response(generated_text)