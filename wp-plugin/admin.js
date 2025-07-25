/**
 * WP SLM Assistant Admin JavaScript
 */

(function($) {
    'use strict';

    // Initialize when DOM is ready
    $(document).ready(function() {
        // Chat functionality
        if ($('#wp-slm-chat-container').length) {
            initializeChat();
        }
        
        // Settings page functionality
        if ($('#wp-slm-test-connection').length) {
            $('#wp-slm-test-connection').on('click', testConnection);
        }
    });

    /**
     * Initialize chat interface
     */
    function initializeChat() {
        const $messages = $('#wp-slm-messages');
        const $input = $('#wp-slm-input');
        const $submit = $('#wp-slm-submit');
        
        // Handle submit button click
        $submit.on('click', function() {
            sendMessage();
        });
        
        // Handle enter key in textarea
        $input.on('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Handle suggestion clicks
        $('.wp-slm-suggestion').on('click', function(e) {
            e.preventDefault();
            $input.val($(this).text());
            sendMessage();
        });
        
        // Focus input on load
        $input.focus();
    }

    /**
     * Send message to API
     */
    function sendMessage() {
        const $input = $('#wp-slm-input');
        const $submit = $('#wp-slm-submit');
        const $messages = $('#wp-slm-messages');
        
        const prompt = $input.val().trim();
        if (!prompt) {
            return;
        }
        
        // Disable input
        $input.prop('disabled', true);
        $submit.prop('disabled', true).text(wpSlm.strings.sending);
        
        // Add user message
        addMessage(prompt, 'user');
        
        // Clear input
        $input.val('');
        
        // Add loading indicator
        const loadingId = 'loading-' + Date.now();
        $messages.append(
            '<div id="' + loadingId + '" class="wp-slm-message assistant">' +
            '<div class="wp-slm-loading"></div> ' +
            'Thinking...' +
            '</div>'
        );
        
        // Scroll to bottom
        $messages.scrollTop($messages[0].scrollHeight);
        
        // Make API request
        $.ajax({
            url: wpSlm.apiUrl + 'chat',
            method: 'POST',
            beforeSend: function(xhr) {
                xhr.setRequestHeader('X-WP-Nonce', wpSlm.nonce);
            },
            contentType: 'application/json',
            data: JSON.stringify({
                prompt: prompt
            }),
            success: function(response) {
                // Remove loading indicator
                $('#' + loadingId).remove();
                
                // Add assistant response
                addMessage(response.response, 'assistant');
                
                // Re-enable input
                $input.prop('disabled', false).focus();
                $submit.prop('disabled', false).text('Send');
            },
            error: function(xhr) {
                // Remove loading indicator
                $('#' + loadingId).remove();
                
                // Show error
                let errorMessage = wpSlm.strings.error;
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMessage = xhr.responseJSON.message;
                }
                
                addMessage(errorMessage, 'assistant error');
                
                // Re-enable input
                $input.prop('disabled', false).focus();
                $submit.prop('disabled', false).text('Send');
            }
        });
    }

    /**
     * Add message to chat
     */
    function addMessage(content, type) {
        const $messages = $('#wp-slm-messages');
        
        // Process content for markdown and code blocks
        const processedContent = processMessageContent(content);
        
        // Create message element
        const $message = $('<div>')
            .addClass('wp-slm-message ' + type)
            .html(processedContent);
        
        // Append to messages
        $messages.append($message);
        
        // Apply syntax highlighting to code blocks
        if (type === 'assistant' && typeof Prism !== 'undefined') {
            $message.find('pre code').each(function() {
                Prism.highlightElement(this);
            });
        }
        
        // Scroll to bottom
        $messages.scrollTop($messages[0].scrollHeight);
    }

    /**
     * Process message content for display
     */
    function processMessageContent(content) {
        // Escape HTML
        let processed = escapeHtml(content);
        
        // Convert markdown code blocks
        processed = processed.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
            lang = lang || 'php';
            return '<pre><code class="language-' + lang + '">' + code.trim() + '</code></pre>';
        });
        
        // Convert inline code
        processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Convert line breaks
        processed = processed.replace(/\n/g, '<br>');
        
        return processed;
    }

    /**
     * Escape HTML entities
     */
    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        
        return text.replace(/[&<>"']/g, function(m) {
            return map[m];
        });
    }

    /**
     * Test connection to API
     */
    function testConnection() {
        const $button = $(this);
        const $result = $('#wp-slm-test-result');
        
        $button.prop('disabled', true);
        $result.html('<span class="wp-slm-loading" style="width: 16px; height: 16px;"></span> Testing...');
        
        $.ajax({
            url: wpSlm.apiUrl + 'test',
            method: 'GET',
            beforeSend: function(xhr) {
                xhr.setRequestHeader('X-WP-Nonce', wpSlm.nonce);
            },
            success: function(response) {
                $result.html('<span style="color: green;">✓ ' + wpSlm.strings.connectionSuccess + '</span>');
                $button.prop('disabled', false);
            },
            error: function(xhr) {
                let errorMessage = wpSlm.strings.connectionFailed;
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMessage += ' ' + xhr.responseJSON.message;
                } else if (xhr.status === 0) {
                    errorMessage += ' Server is not reachable. Make sure the WP-SLM server is running.';
                } else {
                    errorMessage += ' Status: ' + xhr.status;
                }
                
                $result.html('<span style="color: red;">✗ ' + errorMessage + '</span>');
                $button.prop('disabled', false);
            }
        });
    }

})(jQuery);