# WP SLM Assistant Plugin

WordPress plugin that provides an AI-powered assistant using the locally hosted WP-SLM model.

## Features

- **Admin Dashboard Chat Interface**: Interactive chat with the WordPress expert model
- **Code Syntax Highlighting**: Automatic highlighting for PHP, JavaScript, and other languages
- **Chat History**: Saves conversations for future reference
- **Customizable Settings**: Configure API endpoint and behavior
- **REST API Integration**: OpenAI-compatible API interface

## Installation

1. Ensure the WP-SLM inference server is running:
   ```bash
   python inference/serve_vllm.py --model outputs/wp-slm-merged
   ```

2. Install the plugin:
   - Copy the `wp-plugin` directory to `wp-content/plugins/wp-slm-assistant/`
   - Or zip the directory and upload via WordPress admin

3. Activate the plugin in WordPress admin

4. Configure settings:
   - Go to **WP Assistant → Settings**
   - Verify the API endpoint (default: `http://localhost:8000/v1/chat/completions`)
   - Test the connection

## Usage

### Chat Interface

1. Navigate to **WP Assistant** in the admin menu
2. Type your WordPress-related question
3. Press Enter or click Send
4. The AI will respond with helpful information and code examples

### Example Questions

- "How do I create a custom post type?"
- "Show me how to add a REST API endpoint"
- "What's the best way to enqueue scripts in WordPress?"
- "How can I add a custom Gutenberg block?"
- "Help me debug this WordPress error: [error message]"

## API Endpoints

The plugin registers the following REST API endpoints:

- `POST /wp-json/wp-slm/v1/chat` - Send a chat message
- `GET /wp-json/wp-slm/v1/test` - Test connection to the model server

## Requirements

- WordPress 5.0 or higher
- PHP 7.2 or higher
- WP-SLM inference server running locally
- Administrator privileges (for accessing the assistant)

## Security

- Only administrators can access the assistant
- All API requests are nonce-protected
- Input is sanitized before processing
- Chat history is user-specific

## Customization

### Modify Suggestions

Edit the suggestions in `wp_slm_render_admin_page()` function to provide relevant prompts for your use case.

### Styling

The plugin includes basic styling. You can override styles by adding custom CSS to your theme.

### API Timeout

Adjust the API timeout in settings if you're working with longer responses or slower hardware.

## Troubleshooting

### Connection Failed

1. Ensure the inference server is running
2. Check the API endpoint URL in settings
3. Verify firewall settings allow local connections
4. Check the browser console for detailed errors

### Slow Responses

1. Increase the API timeout in settings
2. Ensure your GPU is properly configured
3. Consider using a smaller model or quantization

### No Response

1. Check WordPress debug log for errors
2. Verify the model is loaded correctly
3. Test with a simple prompt like "Hello"

## Development

### Hooks

The plugin provides several hooks for customization:

```php
// Filter the API request before sending
add_filter('wp_slm_api_request', function($request) {
    // Modify request
    return $request;
});

// Action after saving chat history
add_action('wp_slm_history_saved', function($history_id, $prompt, $response) {
    // Custom logging or processing
}, 10, 3);
```

### Adding Custom Commands

You can extend the assistant with custom commands by filtering the prompt:

```php
add_filter('wp_slm_process_prompt', function($prompt) {
    if (strpos($prompt, '/help') === 0) {
        // Return custom help response
        return "Here are available commands...";
    }
    return $prompt;
});
```

## License

GPL v2 or later, consistent with WordPress.