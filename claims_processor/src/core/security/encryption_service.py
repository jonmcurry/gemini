import base64
import os # For generating nonce
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
import structlog
from typing import Optional # Added for Optional type hint

from ..config.settings import get_settings # To fetch APP_ENCRYPTION_KEY

logger = structlog.get_logger(__name__)

class EncryptionService:
    """
    Service for encrypting and decrypting data using AES-256-GCM.
    """
    AES_NONCE_BYTES = 12 # Standard nonce size for AES-GCM is 12 bytes (96 bits)
    # AES-GCM tag is typically 16 bytes (128 bits)

    def __init__(self, encryption_key: Optional[str] = None): # Key can be passed or fetched from settings
        """
        Initializes the EncryptionService.

        Args:
            encryption_key: A string representing the encryption key.
                            If provided, it's used directly. If None, fetched from settings.
                            The key from settings is expected to be a string that can be
                            either base64-encoded 32 bytes, or a raw string from which
                            32 bytes will be derived using SHA-256 (for dev/test only).
        """
        if encryption_key:
            key_str = encryption_key
        else:
            app_settings = get_settings()
            key_str = app_settings.APP_ENCRYPTION_KEY

        key_bytes: Optional[bytes] = None

        # Attempt to decode as base64 first (preferred for production keys)
        if len(key_str) == 44 and key_str.endswith("="): # Common length for base64 encoded 32 bytes
             try:
                decoded_key = base64.urlsafe_b64decode(key_str)
                if len(decoded_key) == 32:
                    key_bytes = decoded_key
                    logger.info("EncryptionService initialized with base64 decoded key from settings/input.")
                else:
                    logger.warn("Decoded key from base64 is not 32 bytes. Will attempt SHA-256 derivation.")
             except Exception:
                logger.warn("Failed to decode key as base64 despite length match. Will attempt SHA-256 derivation.")

        if key_bytes is None:
            # If not successfully decoded as base64, or if it wasn't typical base64 length,
            # derive 32 bytes using SHA-256. This is for dev/test from simple string keys.
            logger.warn("APP_ENCRYPTION_KEY is not a valid base64 encoded 32-byte string. "
                        "Deriving a 32-byte key using SHA-256 from the provided string. "
                        "This is suitable for DEV/TEST ONLY. DO NOT USE THIS IN PRODUCTION "
                        "without a securely generated, base64-encoded 32-byte key.")
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend
            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(key_str.encode('utf-8'))
            key_bytes = digest.finalize()

        if len(key_bytes) != 32:
            # This should ideally not be reached if SHA256 derivation is used, as it always produces 32 bytes.
            # But as a safeguard if logic changes or direct key_str was passed and not 32 bytes.
            logger.error("Derived encryption key is not 32 bytes long for AES-256.")
            raise ValueError("Derived encryption key must be 32 bytes long.")

        self.key = key_bytes
        self.aesgcm = AESGCM(self.key)
        logger.info("EncryptionService initialized successfully.")


    def encrypt(self, plaintext: str) -> Optional[str]:
        """
        Encrypts a plaintext string using AES-256-GCM.

        Args:
            plaintext: The string to encrypt.

        Returns:
            A base64 URL-safe encoded string representing "nonce + ciphertext + tag",
            or None if encryption fails.
        """
        if not isinstance(plaintext, str): # Ensure input is a string
            # logger.error("Plaintext must be a string for encryption.") # Can be noisy
            # Consider raising TypeError or returning a specific error response if that's API contract
            return None
        # Allow encryption of empty string

        try:
            plaintext_bytes = plaintext.encode('utf-8')
            nonce = os.urandom(self.AES_NONCE_BYTES)

            ciphertext_with_tag = self.aesgcm.encrypt(nonce, plaintext_bytes, None)

            encrypted_blob_bytes = nonce + ciphertext_with_tag

            return base64.urlsafe_b64encode(encrypted_blob_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}", exc_info=True)
            return None

    def decrypt(self, ciphertext_blob_b64: str) -> Optional[str]:
        """
        Decrypts a base64 URL-safe encoded string (nonce + ciphertext + tag) using AES-256-GCM.

        Args:
            ciphertext_blob_b64: The base64 encoded string to decrypt.

        Returns:
            The decrypted plaintext string, or None if decryption fails (e.g., invalid tag, tampered).
        """
        if not isinstance(ciphertext_blob_b64, str) or not ciphertext_blob_b64:
            # logger.warn("Ciphertext blob must be a non-empty string for decryption.")
            return None

        try:
            encrypted_blob_bytes = base64.urlsafe_b64decode(ciphertext_blob_b64.encode('utf-8'))

            nonce = encrypted_blob_bytes[:self.AES_NONCE_BYTES]
            ciphertext_with_tag = encrypted_blob_bytes[self.AES_NONCE_BYTES:]

            if len(nonce) != self.AES_NONCE_BYTES:
                logger.error(f"Invalid nonce length after decoding. Expected {self.AES_NONCE_BYTES}, got {len(nonce)}.")
                return None
            # Also check if ciphertext_with_tag is too short (e.g., less than tag length, typically 16)
            # AESGCM decrypt will handle tag verification, but a length check can be an early exit.
            if len(ciphertext_with_tag) < 16: # Assuming a 128-bit (16-byte) tag
                 logger.error(f"Ciphertext with tag is too short. Length: {len(ciphertext_with_tag)}")
                 return None


            decrypted_bytes = self.aesgcm.decrypt(nonce, ciphertext_with_tag, None)

            return decrypted_bytes.decode('utf-8')
        except InvalidTag:
            logger.warn("Decryption failed: Invalid authentication tag. Ciphertext may have been tampered or wrong key used.") # Changed to warn for InvalidTag
            return None
        except Exception as e:
            logger.error(f"Decryption failed due to an unexpected error: {e}", exc_info=True)
            return None
