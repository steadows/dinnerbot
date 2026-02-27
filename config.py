import os
from dotenv import load_dotenv
from google.cloud import secretmanager

# Load environment variables from .env file for local development
load_dotenv()

# Fix gRPC DNS resolution on macOS — must be set before any gRPC client initializes
if os.getenv("GRPC_DNS_RESOLVER"):
    os.environ["GRPC_DNS_RESOLVER"] = os.getenv("GRPC_DNS_RESOLVER")

# Sentinel to distinguish "not yet initialized" from "initialization failed"
_NOT_INITIALIZED = object()


class Config:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self._secret_client = _NOT_INITIALIZED

    @property
    def secret_client(self):
        if self._secret_client is _NOT_INITIALIZED:
            try:
                self._secret_client = secretmanager.SecretManagerServiceClient()
            except Exception as e:
                # This might fail if running locally without proper Google credentials
                print(f"Warning: Could not initialize Secret Manager client: {e}")
                self._secret_client = None  # Mark as "tried and failed"
        return self._secret_client

    def get_value(self, key, is_secret=False):
        """
        Retrieves a configuration value.
        1. Checks environment variables first (local override).
        2. If is_secret is True and project_id is set, tries Google Secret Manager.
        """
        # 1. Try environment variable
        env_val = os.getenv(key)
        if env_val:
            return env_val

        # 2. Try Secret Manager if it's a secret and we have a project ID
        if is_secret and self.project_id:
            if not self.secret_client:
                return None
                
            name = f"projects/{self.project_id}/secrets/{key}/versions/latest"
            try:
                response = self.secret_client.access_secret_version(request={"name": name})
                return response.payload.data.decode("UTF-8")
            except Exception as e:
                print(f"Error fetching secret {key} from Secret Manager: {e}")
                return None
        
        return None

    @property
    def GEMINI_API_KEY(self):
        return self.get_value("GEMINI_API_KEY", is_secret=True)

    # ==================== TELEGRAM ====================

    @property
    def TELEGRAM_BOT_TOKEN(self):
        """Telegram Bot API token from @BotFather."""
        return self.get_value("TELEGRAM_BOT_TOKEN", is_secret=True)

    @property
    def TELEGRAM_CHAT_ID(self):
        """Telegram chat ID for the family chat."""
        return self.get_value("TELEGRAM_CHAT_ID", is_secret=False)

    @property
    def TELEGRAM_WEBHOOK_SECRET(self):
        """Secret token for validating Telegram webhook requests."""
        return self.get_value("TELEGRAM_WEBHOOK_SECRET", is_secret=True)

    @property
    def ALLOWED_CHAT_IDS(self) -> list:
        """
        List of authorized Telegram chat IDs.
        Reads from ALLOWED_CHAT_IDS env var (comma-separated) or falls back to TELEGRAM_CHAT_ID.
        """
        raw = self.get_value("ALLOWED_CHAT_IDS", is_secret=False)
        if raw:
            return [int(cid.strip()) for cid in raw.split(",") if cid.strip()]

        # Fall back to the single configured chat ID
        single = self.TELEGRAM_CHAT_ID
        if single:
            return [int(single)]

        return []

    # ==================== TWILIO (Extension A) ====================

    @property
    def TWILIO_ACCOUNT_SID(self):
        return self.get_value("TWILIO_ACCOUNT_SID", is_secret=True)

    @property
    def TWILIO_AUTH_TOKEN(self):
        return self.get_value("TWILIO_AUTH_TOKEN", is_secret=True)

    @property
    def TWILIO_FROM_NUMBER(self):
        return self.get_value("TWILIO_FROM_NUMBER", is_secret=False)

    @property
    def USER_PHONE_NUMBER(self):
        return self.get_value("USER_PHONE_NUMBER", is_secret=False)

    @property
    def GCP_PROJECT_ID(self):
        return self.project_id

# Instantiate a global config object
config = Config()
