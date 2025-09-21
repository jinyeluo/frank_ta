import base64
import mimetypes
import os
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


class GmailSender:
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        """
        Initialize Gmail sender with OAuth2 authentication

        Args:
            credentials_file: Path to your Google API credentials JSON file
            token_file: Path to store/load the access token
        """
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.send']
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Handle OAuth2 authentication"""
        creds = None

        # Load existing token if available
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)

        # If there are no valid credentials, request authorization
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)

            # Save credentials for future use
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

        # Build the Gmail service
        self.service = build('gmail', 'v1', credentials=creds)

    def create_message_with_attachments(self, sender, to, subject, png_file_path, html_string):
        """
        Create a message with PNG attachment and HTML content

        Args:
            sender: Email address of the sender
            to: Email address of the recipient
            subject: Email subject
            png_file_path: Path to the PNG file to attach
            html_string: HTML content as string

        Returns:
            Dict containing the message in the required format
        """
        # Create multipart message
        message = MIMEMultipart('alternative')
        message['to'] = to
        message['from'] = sender
        message['subject'] = subject

        # Add plain text body
        text_part = MIMEText(html_string, 'html')
        message.attach(text_part)

        # Add HTML content as text attachment
        # html_attachment = MIMEText(html_string, 'plain')
        # html_attachment.add_header('Content-Disposition',
        #                            'attachment; filename="content.html"')
        # message.attach(html_attachment)

        # Add PNG file attachment
        if os.path.exists(png_file_path):
            with open(png_file_path, 'rb') as f:
                img_data = f.read()

            # Determine content type
            content_type, _ = mimetypes.guess_type(png_file_path)
            if content_type is None:
                content_type = 'application/octet-stream'

            main_type, sub_type = content_type.split('/', 1)

            # Create attachment
            attachment = MIMEBase(main_type, sub_type)
            attachment.set_payload(img_data)
            encoders.encode_base64(attachment)

            filename = os.path.basename(png_file_path)
            attachment.add_header('Content-Disposition',
                                  f'attachment; filename="{filename}"')
            message.attach(attachment)
        else:
            print(f"Warning: PNG file not found at {png_file_path}")

        # Encode the message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {'raw': raw_message}

    def send_email(self, sender, to, subject, png_file_path, html_string):
        """
        Send email with PNG attachment and HTML string

        Args:
            sender: Your Gmail address
            to: Recipient's email address
            subject: Email subject
            png_file_path: Path to PNG file
            html_string: HTML content as string

        Returns:
            Dict containing the sent message info
        """
        try:
            # Create the message
            message = self.create_message_with_attachments(sender, to, subject, png_file_path, html_string)

            # Send the message
            sent_message = self.service.users().messages().send(
                userId='me', body=message
            ).execute()

            print(f"Message sent successfully! Message ID: {sent_message['id']}")
            return sent_message

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


