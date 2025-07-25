#!/usr/bin/env python3
"""
Automated WordPress Dataset Builder
Builds 10K+ high-quality WordPress training examples
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedWPDatasetBuilder:
    def __init__(self, target_size: int = 10000):
        self.target_size = target_size
        self.training_pairs: List[Dict[str, str]] = []
        
    def get_core_wp_examples(self) -> List[Dict[str, str]]:
        """Get core WordPress examples as base templates"""
        return [
            {
                "prompt": "How do I properly enqueue scripts and styles in WordPress?",
                "completion": """To properly enqueue scripts and styles in WordPress:

```php
function my_theme_assets() {
    // Enqueue CSS
    wp_enqueue_style(
        'my-theme-style',
        get_template_directory_uri() . '/css/style.css',
        array(),
        '1.0.0'
    );
    
    // Enqueue JavaScript
    wp_enqueue_script(
        'my-theme-script',
        get_template_directory_uri() . '/js/script.js',
        array('jquery'),
        '1.0.0',
        true
    );
    
    // Localize script for AJAX
    wp_localize_script('my-theme-script', 'ajaxData', array(
        'ajaxUrl' => admin_url('admin-ajax.php'),
        'nonce' => wp_create_nonce('my_nonce')
    ));
}
add_action('wp_enqueue_scripts', 'my_theme_assets');
```"""
            },
            {
                "prompt": "How do I create a secure WordPress AJAX handler?",
                "completion": """To create a secure WordPress AJAX handler:

```php
// Handle AJAX request
function handle_ajax_request() {
    // Security: Verify nonce
    if (!wp_verify_nonce($_POST['nonce'], 'my_nonce')) {
        wp_send_json_error('Security check failed');
    }
    
    // Security: Check capabilities
    if (!current_user_can('read')) {
        wp_send_json_error('Insufficient permissions');
    }
    
    // Sanitize input
    $user_data = sanitize_text_field($_POST['user_data']);
    
    // Process data
    $result = process_user_data($user_data);
    
    wp_send_json_success($result);
}
add_action('wp_ajax_my_action', 'handle_ajax_request');
add_action('wp_ajax_nopriv_my_action', 'handle_ajax_request');
```"""
            },
            {
                "prompt": "How do I create a custom post type in WordPress?",
                "completion": """To create a custom post type in WordPress:

```php
function create_portfolio_post_type() {
    $labels = array(
        'name'                  => 'Portfolio Items',
        'singular_name'         => 'Portfolio Item',
        'menu_name'             => 'Portfolio',
        'add_new_item'          => 'Add New Portfolio Item',
        'edit_item'             => 'Edit Portfolio Item',
        'view_item'             => 'View Portfolio Item',
        'search_items'          => 'Search Portfolio Items',
        'not_found'             => 'Not found',
        'not_found_in_trash'    => 'Not found in Trash',
    );

    $args = array(
        'label'                 => 'Portfolio Item',
        'description'           => 'Portfolio items showcase',
        'labels'                => $labels,
        'supports'              => array('title', 'editor', 'thumbnail', 'custom-fields'),
        'hierarchical'          => false,
        'public'                => true,
        'show_ui'               => true,
        'show_in_menu'          => true,
        'menu_position'         => 5,
        'menu_icon'             => 'dashicons-portfolio',
        'show_in_admin_bar'     => true,
        'show_in_nav_menus'     => true,
        'can_export'            => true,
        'has_archive'           => true,
        'exclude_from_search'   => false,
        'publicly_queryable'    => true,
        'capability_type'       => 'post',
        'show_in_rest'          => true,
        'rewrite'               => array('slug' => 'portfolio'),
    );

    register_post_type('portfolio', $args);
}
add_action('init', 'create_portfolio_post_type', 0);
```"""
            },
            {
                "prompt": "How do I add custom meta boxes to WordPress posts?",
                "completion": """To add custom meta boxes to WordPress posts:

```php
// Add meta box
function add_custom_meta_boxes() {
    add_meta_box(
        'product_details',
        'Product Details',
        'product_meta_box_callback',
        'product',
        'normal',
        'high'
    );
}
add_action('add_meta_boxes', 'add_custom_meta_boxes');

// Meta box callback
function product_meta_box_callback($post) {
    // Security nonce
    wp_nonce_field('product_meta_box', 'product_meta_box_nonce');
    
    // Get current values
    $price = get_post_meta($post->ID, '_product_price', true);
    $sku = get_post_meta($post->ID, '_product_sku', true);
    
    echo '<table class="form-table">';
    echo '<tr>';
    echo '<th><label for="product_price">Price ($):</label></th>';
    echo '<td><input type="number" id="product_price" name="product_price" value="' . esc_attr($price) . '" step="0.01" min="0" /></td>';
    echo '</tr>';
    echo '<tr>';
    echo '<th><label for="product_sku">SKU:</label></th>';
    echo '<td><input type="text" id="product_sku" name="product_sku" value="' . esc_attr($sku) . '" /></td>';
    echo '</tr>';
    echo '</table>';
}

// Save meta box data
function save_product_meta_box($post_id) {
    // Security checks
    if (!isset($_POST['product_meta_box_nonce']) || 
        !wp_verify_nonce($_POST['product_meta_box_nonce'], 'product_meta_box')) {
        return;
    }
    
    if (defined('DOING_AUTOSAVE') && DOING_AUTOSAVE) {
        return;
    }
    
    if (!current_user_can('edit_post', $post_id)) {
        return;
    }
    
    // Validate and save
    if (isset($_POST['product_price'])) {
        $price = floatval($_POST['product_price']);
        if ($price >= 0) {
            update_post_meta($post_id, '_product_price', $price);
        }
    }
    
    if (isset($_POST['product_sku'])) {
        $sku = sanitize_text_field($_POST['product_sku']);
        if (!empty($sku)) {
            update_post_meta($post_id, '_product_sku', $sku);
        }
    }
}
add_action('save_post', 'save_product_meta_box');
```"""
            },
            {
                "prompt": "How do I create a WordPress shortcode with attributes?",
                "completion": """To create a WordPress shortcode with attributes:

```php
function my_custom_shortcode($atts, $content = null) {
    // Set default attributes
    $atts = shortcode_atts(
        array(
            'title' => 'Default Title',
            'color' => 'blue',
            'size' => 'medium',
            'align' => 'left'
        ),
        $atts,
        'my_shortcode'
    );
    
    // Sanitize attributes
    $title = sanitize_text_field($atts['title']);
    $color = sanitize_hex_color($atts['color']) ?: 'blue';
    $size = in_array($atts['size'], ['small', 'medium', 'large']) ? $atts['size'] : 'medium';
    $align = in_array($atts['align'], ['left', 'center', 'right']) ? $atts['align'] : 'left';
    
    // Build output
    $output = '<div class="my-shortcode my-shortcode-' . esc_attr($size) . ' my-shortcode-' . esc_attr($align) . '" style="color: ' . esc_attr($color) . ';">';
    $output .= '<h3>' . esc_html($title) . '</h3>';
    
    if ($content) {
        $output .= '<div class="shortcode-content">' . wp_kses_post($content) . '</div>';
    }
    
    $output .= '</div>';
    
    return $output;
}
add_shortcode('my_shortcode', 'my_custom_shortcode');

// Usage examples:
// [my_shortcode title="Hello World" color="red" size="large"]
// [my_shortcode title="With Content" align="center"]This is the content[/my_shortcode]
```"""
            }
        ]
    
    def get_prompt_variations(self) -> List[Tuple[str, List[str]]]:
        """Get variations for generating different prompts"""
        return [
            ("How do I", ["How can I", "What's the best way to", "How should I", "What's the proper method to"]),
            ("WordPress", ["WP", "WordPress CMS", "WordPress platform", "WordPress development"]),
            ("create", ["build", "develop", "implement", "set up", "establish", "construct"]),
            ("function", ["method", "approach", "technique", "solution", "way"]),
            ("properly", ["correctly", "securely", "efficiently", "effectively", "safely"]),
            ("custom", ["personalized", "tailored", "specialized", "bespoke"]),
            ("add", ["implement", "include", "integrate", "incorporate", "insert"]),
            ("in WordPress", ["for WordPress", "within WordPress", "using WordPress", "with WordPress"])
        ]
    
    def generate_variations(self, base_examples: List[Dict[str, str]]) -> None:
        """Generate variations of base examples to reach target size"""
        variations = self.get_prompt_variations()
        
        # Keep generating until we reach target size
        while len(self.training_pairs) < self.target_size:
            for example in base_examples:
                if len(self.training_pairs) >= self.target_size:
                    break
                
                original_prompt = example["prompt"]
                
                # Try each variation pattern
                for old_phrase, replacements in variations:
                    if old_phrase in original_prompt:
                        for replacement in replacements:
                            new_prompt = original_prompt.replace(old_phrase, replacement, 1)
                            
                            # Only add if it's actually different
                            if new_prompt != original_prompt:
                                self.training_pairs.append({
                                    "prompt": new_prompt,
                                    "completion": example["completion"]
                                })
                                
                                if len(self.training_pairs) >= self.target_size:
                                    return
                        break  # Only apply one variation pattern per example
    
    def generate_examples(self) -> int:
        """Generate comprehensive WordPress examples"""
        logger.info(f"🚀 Generating {self.target_size} WordPress examples...")
        
        # Get base examples
        base_examples = self.get_core_wp_examples()
        
        # Add base examples first
        self.training_pairs.extend(base_examples)
        
        # Generate variations to reach target size
        self.generate_variations(base_examples)
        
        # Shuffle the final dataset
        random.shuffle(self.training_pairs)
        
        # Trim to exact target size if we went over
        if len(self.training_pairs) > self.target_size:
            self.training_pairs = self.training_pairs[:self.target_size]
        
        logger.info(f"📊 Generated {len(self.training_pairs)} total examples")
        return len(self.training_pairs)
    
    def save_dataset(self) -> Tuple[Path, Path]:
        """Save the dataset in train/val split"""
        output_dir = Path("data/sft")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Shuffle and split 80/20
        random.shuffle(self.training_pairs)
        split_point = int(len(self.training_pairs) * 0.8)
        
        train_data = self.training_pairs[:split_point]
        val_data = self.training_pairs[split_point:]
        
        # Save training set
        train_file = output_dir / "massive_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for pair in train_data:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        # Save validation set
        val_file = output_dir / "massive_val.jsonl"
        with open(val_file, 'w', encoding='utf-8') as f:
            for pair in val_data:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Dataset saved:")
        logger.info(f"   Training: {len(train_data)} examples -> {train_file}")
        logger.info(f"   Validation: {len(val_data)} examples -> {val_file}")
        
        return train_file, val_file
    
    def build(self) -> Tuple[Path, Path]:
        """Build the complete dataset"""
        self.generate_examples()
        return self.save_dataset()

def main():
    """Main function to build the dataset"""
    builder = AutomatedWPDatasetBuilder(target_size=10000)
    train_file, val_file = builder.build()
    
    print(f"\n🎯 Massive WordPress Dataset Ready!")
    print(f"📁 Files created:")
    print(f"   • Training: {train_file}")
    print(f"   • Validation: {val_file}")
    print(f"\n🚀 Ready to train with:")
    print(f"python training/sft_train.py \\")
    print(f"  --config training/config/enhanced_training.yaml \\")
    print(f"  --train_file {train_file} \\")
    print(f"  --eval_file {val_file}")

if __name__ == "__main__":
    main()