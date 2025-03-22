# email_service.py
import os
import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# Set up logging
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        # Load API key from environment variable
        self.api_key = os.environ.get('SENDGRID_API_KEY')
        self.default_sender = os.environ.get('DEFAULT_EMAIL_SENDER', 'noreply@yourdomain.com')
        
        if not self.api_key:
            logger.warning("SendGrid API key not found in environment variables")
    
    def send_password_reset_email(self, recipient_email, reset_link):
        """
        Send password reset email using SendGrid
        
        Args:
            recipient_email (str): Recipient's email address
            reset_link (str): Password reset URL
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        try:
            # Configure message
            message = Mail(
                from_email=self.default_sender,
                to_emails=recipient_email,
                subject='Password Reset Request',
                html_content=self._get_password_reset_template(reset_link)
            )
            
            # Send email
            sg = SendGridAPIClient(self.api_key)
            response = sg.send(message)
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Reset email sent to {recipient_email}, status code: {response.status_code}")
                return True
            else:
                logger.error(f"Failed to send email. Status code: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending password reset email: {e}")
            return False
    
    def _get_password_reset_template(self, reset_link):
        """
        Returns the HTML template for password reset emails
        
        Args:
            reset_link (str): Password reset URL
            
        Returns:
            str: HTML content for the email
        """
        return f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px;">
                <h2 style="color: #004e98;">Password Reset Request</h2>
                <p>You have requested to reset your password. Click the link below to set a new password:</p>
                <p><a href="{reset_link}" style="display: inline-block; padding: 10px 20px; background-color: #00aaff; color: #ffffff; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
                <p>If you did not request this reset, please ignore this email.</p>
                <p>This link will expire in 24 hours.</p>
            </div>
        '''
