<?php
/**
 * Plugin Name: WP SLM Assistant
 * Plugin URI: https://github.com/your-org/wp-slm
 * Description: AI-powered WordPress assistant using locally hosted language model
 * Version: 1.0.0
 * Author: WP SLM Team
 * License: GPL v2 or later
 * Text Domain: wp-slm-assistant
 */

// Prevent direct access
if (!defined('ABSPATH')) {
    exit;
}

// Define plugin constants
define('WP_SLM_VERSION', '1.0.0');
define('WP_SLM_PLUGIN_DIR', plugin_dir_path(__FILE__));
define('WP_SLM_PLUGIN_URL', plugin_dir_url(__FILE__));
define('WP_SLM_API_ENDPOINT', get_option('wp_slm_api_endpoint', 'http://localhost:8000/v1/chat/completions'));

// Activation hook
register_activation_hook(__FILE__, 'wp_slm_activate');
function wp_slm_activate() {
    // Set default options
    add_option('wp_slm_api_endpoint', 'http://localhost:8000/v1/chat/completions');
    add_option('wp_slm_api_timeout', 60);
    add_option('wp_slm_enable_history', true);
    add_option('wp_slm_max_history', 50);
    
    // Create database table for chat history
    wp_slm_create_history_table();
}

// Create history table
function wp_slm_create_history_table() {
    global $wpdb;
    
    $table_name = $wpdb->prefix . 'wp_slm_history';
    $charset_collate = $wpdb->get_charset_collate();
    
    $sql = "CREATE TABLE $table_name (
        id mediumint(9) NOT NULL AUTO_INCREMENT,
        user_id bigint(20) NOT NULL,
        prompt text NOT NULL,
        response text NOT NULL,
        created_at datetime DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id),
        KEY user_id (user_id)
    ) $charset_collate;";
    
    require_once(ABSPATH . 'wp-admin/includes/upgrade.php');
    dbDelta($sql);
}

// Deactivation hook
register_deactivation_hook(__FILE__, 'wp_slm_deactivate');
function wp_slm_deactivate() {
    // Clean up if needed
}

// Add admin menu
add_action('admin_menu', 'wp_slm_add_admin_menu');
function wp_slm_add_admin_menu() {
    add_menu_page(
        __('WP Assistant', 'wp-slm-assistant'),
        __('WP Assistant', 'wp-slm-assistant'),
        'manage_options',
        'wp-slm-assistant',
        'wp_slm_render_admin_page',
        'dashicons-format-chat',
        30
    );
    
    add_submenu_page(
        'wp-slm-assistant',
        __('Settings', 'wp-slm-assistant'),
        __('Settings', 'wp-slm-assistant'),
        'manage_options',
        'wp-slm-settings',
        'wp_slm_render_settings_page'
    );
    
    add_submenu_page(
        'wp-slm-assistant',
        __('History', 'wp-slm-assistant'),
        __('History', 'wp-slm-assistant'),
        'manage_options',
        'wp-slm-history',
        'wp_slm_render_history_page'
    );
}

// Render admin page
function wp_slm_render_admin_page() {
    ?>
    <div class="wrap">
        <h1><?php echo esc_html(get_admin_page_title()); ?></h1>
        
        <div id="wp-slm-chat-container">
            <div id="wp-slm-messages" class="wp-slm-messages"></div>
            
            <div class="wp-slm-input-container">
                <textarea id="wp-slm-input" 
                          class="wp-slm-input" 
                          placeholder="<?php esc_attr_e('Ask me anything about WordPress...', 'wp-slm-assistant'); ?>"
                          rows="3"></textarea>
                <button id="wp-slm-submit" class="button button-primary">
                    <?php esc_html_e('Send', 'wp-slm-assistant'); ?>
                </button>
            </div>
            
            <div class="wp-slm-suggestions">
                <p><?php esc_html_e('Try asking:', 'wp-slm-assistant'); ?></p>
                <ul>
                    <li><a href="#" class="wp-slm-suggestion">How do I create a custom post type?</a></li>
                    <li><a href="#" class="wp-slm-suggestion">Show me how to add a REST API endpoint</a></li>
                    <li><a href="#" class="wp-slm-suggestion">How can I optimize my WordPress site for speed?</a></li>
                    <li><a href="#" class="wp-slm-suggestion">What's the best way to secure wp-admin?</a></li>
                </ul>
            </div>
        </div>
    </div>
    
    <style>
        #wp-slm-chat-container {
            max-width: 800px;
            margin: 20px 0;
        }
        
        .wp-slm-messages {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 20px;
            height: 400px;
            overflow-y: auto;
            background: #f9f9f9;
            margin-bottom: 20px;
        }
        
        .wp-slm-message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 4px;
        }
        
        .wp-slm-message.user {
            background: #0073aa;
            color: white;
            margin-left: 20%;
        }
        
        .wp-slm-message.assistant {
            background: white;
            border: 1px solid #ddd;
            margin-right: 20%;
        }
        
        .wp-slm-message pre {
            background: #f1f1f1;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        
        .wp-slm-message code {
            background: #f1f1f1;
            padding: 2px 4px;
            border-radius: 2px;
        }
        
        .wp-slm-input-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .wp-slm-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .wp-slm-suggestions {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
        }
        
        .wp-slm-suggestions ul {
            list-style: disc;
            margin-left: 20px;
        }
        
        .wp-slm-suggestion {
            text-decoration: none;
        }
        
        .wp-slm-loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: wp-slm-spin 1s linear infinite;
        }
        
        @keyframes wp-slm-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <?php
}

// Render settings page
function wp_slm_render_settings_page() {
    if (isset($_POST['submit'])) {
        check_admin_referer('wp_slm_settings');
        
        update_option('wp_slm_api_endpoint', sanitize_text_field($_POST['api_endpoint']));
        update_option('wp_slm_api_timeout', intval($_POST['api_timeout']));
        update_option('wp_slm_enable_history', isset($_POST['enable_history']));
        update_option('wp_slm_max_history', intval($_POST['max_history']));
        
        echo '<div class="notice notice-success"><p>' . esc_html__('Settings saved.', 'wp-slm-assistant') . '</p></div>';
    }
    
    $api_endpoint = get_option('wp_slm_api_endpoint');
    $api_timeout = get_option('wp_slm_api_timeout');
    $enable_history = get_option('wp_slm_enable_history');
    $max_history = get_option('wp_slm_max_history');
    ?>
    <div class="wrap">
        <h1><?php echo esc_html(get_admin_page_title()); ?></h1>
        
        <form method="post" action="">
            <?php wp_nonce_field('wp_slm_settings'); ?>
            
            <table class="form-table">
                <tr>
                    <th scope="row">
                        <label for="api_endpoint"><?php esc_html_e('API Endpoint', 'wp-slm-assistant'); ?></label>
                    </th>
                    <td>
                        <input type="url" 
                               id="api_endpoint" 
                               name="api_endpoint" 
                               value="<?php echo esc_attr($api_endpoint); ?>" 
                               class="regular-text" />
                        <p class="description">
                            <?php esc_html_e('The URL of your local WP-SLM server endpoint.', 'wp-slm-assistant'); ?>
                        </p>
                    </td>
                </tr>
                
                <tr>
                    <th scope="row">
                        <label for="api_timeout"><?php esc_html_e('API Timeout', 'wp-slm-assistant'); ?></label>
                    </th>
                    <td>
                        <input type="number" 
                               id="api_timeout" 
                               name="api_timeout" 
                               value="<?php echo esc_attr($api_timeout); ?>" 
                               min="10" 
                               max="300" /> 
                        <?php esc_html_e('seconds', 'wp-slm-assistant'); ?>
                    </td>
                </tr>
                
                <tr>
                    <th scope="row">
                        <?php esc_html_e('Chat History', 'wp-slm-assistant'); ?>
                    </th>
                    <td>
                        <label>
                            <input type="checkbox" 
                                   name="enable_history" 
                                   value="1" 
                                   <?php checked($enable_history, true); ?> />
                            <?php esc_html_e('Enable chat history', 'wp-slm-assistant'); ?>
                        </label>
                    </td>
                </tr>
                
                <tr>
                    <th scope="row">
                        <label for="max_history"><?php esc_html_e('Max History Items', 'wp-slm-assistant'); ?></label>
                    </th>
                    <td>
                        <input type="number" 
                               id="max_history" 
                               name="max_history" 
                               value="<?php echo esc_attr($max_history); ?>" 
                               min="10" 
                               max="1000" />
                    </td>
                </tr>
            </table>
            
            <?php submit_button(); ?>
        </form>
        
        <hr>
        
        <h2><?php esc_html_e('Connection Test', 'wp-slm-assistant'); ?></h2>
        <p>
            <button id="wp-slm-test-connection" class="button">
                <?php esc_html_e('Test Connection', 'wp-slm-assistant'); ?>
            </button>
            <span id="wp-slm-test-result"></span>
        </p>
    </div>
    <?php
}

// Render history page
function wp_slm_render_history_page() {
    global $wpdb;
    $table_name = $wpdb->prefix . 'wp_slm_history';
    
    // Handle clear history
    if (isset($_POST['clear_history'])) {
        check_admin_referer('wp_slm_clear_history');
        $wpdb->query("TRUNCATE TABLE $table_name");
        echo '<div class="notice notice-success"><p>' . esc_html__('History cleared.', 'wp-slm-assistant') . '</p></div>';
    }
    
    // Get history
    $current_user_id = get_current_user_id();
    $history = $wpdb->get_results(
        $wpdb->prepare(
            "SELECT * FROM $table_name WHERE user_id = %d ORDER BY created_at DESC LIMIT 100",
            $current_user_id
        )
    );
    ?>
    <div class="wrap">
        <h1><?php echo esc_html(get_admin_page_title()); ?></h1>
        
        <form method="post" action="" style="margin-bottom: 20px;">
            <?php wp_nonce_field('wp_slm_clear_history'); ?>
            <input type="submit" 
                   name="clear_history" 
                   class="button" 
                   value="<?php esc_attr_e('Clear History', 'wp-slm-assistant'); ?>"
                   onclick="return confirm('<?php esc_attr_e('Are you sure you want to clear your chat history?', 'wp-slm-assistant'); ?>');" />
        </form>
        
        <?php if (empty($history)) : ?>
            <p><?php esc_html_e('No chat history found.', 'wp-slm-assistant'); ?></p>
        <?php else : ?>
            <table class="wp-list-table widefat fixed striped">
                <thead>
                    <tr>
                        <th style="width: 150px;"><?php esc_html_e('Date', 'wp-slm-assistant'); ?></th>
                        <th><?php esc_html_e('Prompt', 'wp-slm-assistant'); ?></th>
                        <th><?php esc_html_e('Response', 'wp-slm-assistant'); ?></th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($history as $item) : ?>
                        <tr>
                            <td><?php echo esc_html(wp_date('Y-m-d H:i', strtotime($item->created_at))); ?></td>
                            <td><?php echo esc_html(substr($item->prompt, 0, 100)) . (strlen($item->prompt) > 100 ? '...' : ''); ?></td>
                            <td><?php echo esc_html(substr($item->response, 0, 100)) . (strlen($item->response) > 100 ? '...' : ''); ?></td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        <?php endif; ?>
    </div>
    <?php
}

// Register REST API endpoints
add_action('rest_api_init', 'wp_slm_register_rest_routes');
function wp_slm_register_rest_routes() {
    register_rest_route('wp-slm/v1', '/chat', array(
        'methods' => 'POST',
        'callback' => 'wp_slm_handle_chat',
        'permission_callback' => function() {
            return current_user_can('manage_options');
        },
        'args' => array(
            'prompt' => array(
                'required' => true,
                'type' => 'string',
                'sanitize_callback' => 'sanitize_textarea_field',
            ),
        ),
    ));
    
    register_rest_route('wp-slm/v1', '/test', array(
        'methods' => 'GET',
        'callback' => 'wp_slm_test_connection',
        'permission_callback' => function() {
            return current_user_can('manage_options');
        },
    ));
}

// Handle chat API request
function wp_slm_handle_chat($request) {
    $prompt = $request->get_param('prompt');
    
    // Prepare API request
    $api_endpoint = get_option('wp_slm_api_endpoint');
    $api_timeout = get_option('wp_slm_api_timeout');
    
    $body = json_encode(array(
        'model' => 'wp-slm',
        'messages' => array(
            array(
                'role' => 'user',
                'content' => $prompt
            )
        ),
        'temperature' => 0.7,
        'max_tokens' => 1024,
    ));
    
    $response = wp_remote_post($api_endpoint, array(
        'headers' => array(
            'Content-Type' => 'application/json',
        ),
        'body' => $body,
        'timeout' => $api_timeout,
    ));
    
    if (is_wp_error($response)) {
        return new WP_Error('api_error', $response->get_error_message(), array('status' => 500));
    }
    
    $body = wp_remote_retrieve_body($response);
    $data = json_decode($body, true);
    
    if (!isset($data['choices'][0]['message']['content'])) {
        return new WP_Error('invalid_response', 'Invalid response from API', array('status' => 500));
    }
    
    $assistant_response = $data['choices'][0]['message']['content'];
    
    // Save to history if enabled
    if (get_option('wp_slm_enable_history')) {
        global $wpdb;
        $wpdb->insert(
            $wpdb->prefix . 'wp_slm_history',
            array(
                'user_id' => get_current_user_id(),
                'prompt' => $prompt,
                'response' => $assistant_response,
            )
        );
        
        // Clean up old history
        wp_slm_cleanup_history();
    }
    
    return rest_ensure_response(array(
        'response' => $assistant_response,
        'usage' => $data['usage'] ?? null,
    ));
}

// Test connection
function wp_slm_test_connection() {
    $api_endpoint = str_replace('/chat/completions', '/models', get_option('wp_slm_api_endpoint'));
    
    $response = wp_remote_get($api_endpoint, array(
        'timeout' => 10,
    ));
    
    if (is_wp_error($response)) {
        return new WP_Error('connection_failed', $response->get_error_message(), array('status' => 500));
    }
    
    $status_code = wp_remote_retrieve_response_code($response);
    if ($status_code !== 200) {
        return new WP_Error('connection_failed', 'Server returned status ' . $status_code, array('status' => 500));
    }
    
    return rest_ensure_response(array(
        'success' => true,
        'message' => 'Connection successful',
    ));
}

// Cleanup old history
function wp_slm_cleanup_history() {
    global $wpdb;
    $max_history = get_option('wp_slm_max_history', 50);
    $user_id = get_current_user_id();
    
    $wpdb->query($wpdb->prepare(
        "DELETE FROM {$wpdb->prefix}wp_slm_history 
         WHERE user_id = %d 
         AND id NOT IN (
             SELECT id FROM (
                 SELECT id FROM {$wpdb->prefix}wp_slm_history 
                 WHERE user_id = %d 
                 ORDER BY created_at DESC 
                 LIMIT %d
             ) AS recent
         )",
        $user_id, $user_id, $max_history
    ));
}

// Enqueue admin scripts
add_action('admin_enqueue_scripts', 'wp_slm_enqueue_admin_scripts');
function wp_slm_enqueue_admin_scripts($hook) {
    if (!in_array($hook, array('toplevel_page_wp-slm-assistant', 'wp-assistant_page_wp-slm-settings'))) {
        return;
    }
    
    wp_enqueue_script(
        'wp-slm-admin',
        WP_SLM_PLUGIN_URL . 'admin.js',
        array('jquery', 'wp-api'),
        WP_SLM_VERSION,
        true
    );
    
    wp_localize_script('wp-slm-admin', 'wpSlm', array(
        'apiUrl' => rest_url('wp-slm/v1/'),
        'nonce' => wp_create_nonce('wp_rest'),
        'strings' => array(
            'sending' => __('Sending...', 'wp-slm-assistant'),
            'error' => __('An error occurred. Please try again.', 'wp-slm-assistant'),
            'connectionSuccess' => __('Connection successful!', 'wp-slm-assistant'),
            'connectionFailed' => __('Connection failed:', 'wp-slm-assistant'),
        ),
    ));
    
    // Add Prism.js for syntax highlighting
    wp_enqueue_script(
        'prism-js',
        'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js',
        array(),
        '1.29.0',
        true
    );
    
    wp_enqueue_script(
        'prism-php',
        'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-php.min.js',
        array('prism-js'),
        '1.29.0',
        true
    );
    
    wp_enqueue_style(
        'prism-css',
        'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css',
        array(),
        '1.29.0'
    );
}