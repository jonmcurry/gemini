import pytest
from unittest.mock import patch, MagicMock
import base64
import os

from claims_processor.src.core.security.encryption_service import EncryptionService
from claims_processor.src.core.config.settings import Settings # To provide settings for key

# A valid 32-byte key, base64 URL-safe encoded.
# (Generated from: os.urandom(32) -> b64encode -> decode)
# Example: b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
# -> AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=
TEST_B64_KEY = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="
TEST_PLAINTEXT = "This is some sensitive data for testing!"
DEV_PLACEHOLDER_KEY = "must_be_32_bytes_long_for_aes256_key!" # From settings

@pytest.fixture
def service_with_b64_key() -> EncryptionService:
    # Pass key directly to constructor to bypass settings mock for this specific key type
    return EncryptionService(encryption_key=TEST_B64_KEY)


@pytest.fixture
def service_with_dev_key_string() -> EncryptionService:
    # Test the path where settings provide a non-base64 string, triggering SHA256 derivation
    with patch('claims_processor.src.core.security.encryption_service.get_settings') as mock_get_settings:
        mock_get_settings.return_value = Settings(APP_ENCRYPTION_KEY=DEV_PLACEHOLDER_KEY)
        service = EncryptionService() # Fetches from mocked settings
    return service

@pytest.fixture
def service_default_init_dev_key() -> EncryptionService:
    # This relies on the default key in Settings being the DEV_PLACEHOLDER_KEY
    # This test ensures default init path (no key passed, actual settings used via get_settings())
    # also results in a usable service (SHA256 derived key).
    # We must ensure the actual Settings default for APP_ENCRYPTION_KEY is DEV_PLACEHOLDER_KEY
    # For isolated test, better to patch get_settings as in service_with_dev_key_string
    # This fixture is more of an integration test with actual settings default.
    # Let's make it explicit by patching, similar to service_with_dev_key_string
    with patch('claims_processor.src.core.security.encryption_service.get_settings') as mock_get_settings:
        mock_get_settings.return_value = Settings(APP_ENCRYPTION_KEY=DEV_PLACEHOLDER_KEY)
        service = EncryptionService()
    return service


def test_encryption_decryption_round_trip_b64_key(service_with_b64_key: EncryptionService):
    encrypted = service_with_b64_key.encrypt(TEST_PLAINTEXT)
    assert encrypted is not None
    assert encrypted != TEST_PLAINTEXT

    decrypted = service_with_b64_key.decrypt(encrypted)
    assert decrypted == TEST_PLAINTEXT

def test_encryption_decryption_round_trip_dev_key(service_with_dev_key_string: EncryptionService):
    # This tests the SHA256 key derivation path via settings
    encrypted = service_with_dev_key_string.encrypt(TEST_PLAINTEXT)
    assert encrypted is not None
    assert encrypted != TEST_PLAINTEXT

    decrypted = service_with_dev_key_string.decrypt(encrypted)
    assert decrypted == TEST_PLAINTEXT

def test_encryption_decryption_round_trip_direct_dev_key_passed_to_constructor():
    # Test SHA256 derivation when a non-b64 key is passed directly to constructor
    service = EncryptionService(encryption_key=DEV_PLACEHOLDER_KEY)
    encrypted = service.encrypt(TEST_PLAINTEXT)
    assert encrypted is not None
    assert encrypted != TEST_PLAINTEXT
    decrypted = service.decrypt(encrypted)
    assert decrypted == TEST_PLAINTEXT


def test_decrypt_invalid_token(service_with_b64_key: EncryptionService):
    assert service_with_b64_key.decrypt("this_is_not_valid_base64_or_ciphertext") is None
    # Test with valid base64 but wrong content (e.g., too short, bad nonce/tag)
    # Nonce (12) + min ciphertext (1) + tag (16) = 29 bytes
    assert service_with_b64_key.decrypt(base64.urlsafe_b64encode(os.urandom(10)).decode()) is None # Too short for nonce
    assert service_with_b64_key.decrypt(base64.urlsafe_b64encode(os.urandom(28)).decode()) is None # Valid nonce, but ciphertext+tag too short

    encrypted_real = service_with_b64_key.encrypt(TEST_PLAINTEXT)
    assert encrypted_real is not None # Ensure encryption worked before trying to tamper

    tampered_b64_bytes = base64.urlsafe_b64decode(encrypted_real.encode())

    # Tamper with nonce
    if len(tampered_b64_bytes) > 0:
        tampered_nonce_list = list(tampered_b64_bytes)
        tampered_nonce_list[0] = tampered_nonce_list[0] ^ 1 # Flip a bit in nonce
        tampered_nonce_b64 = base64.urlsafe_b64encode(bytes(tampered_nonce_list)).decode()
        assert service_with_b64_key.decrypt(tampered_nonce_b64) is None

    # Tamper with ciphertext (after nonce, before tag)
    # Nonce is 12 bytes. Tag is 16 bytes at end. Ciphertext is in between.
    if len(tampered_b64_bytes) > (12 + 16): # Ensure there's ciphertext to tamper
        tampered_ciphertext_list = list(tampered_b64_bytes)
        tampered_ciphertext_list[12] = tampered_ciphertext_list[12] ^ 1 # Flip a bit in ciphertext
        tampered_ciphertext_b64 = base64.urlsafe_b64encode(bytes(tampered_ciphertext_list)).decode()
        assert service_with_b64_key.decrypt(tampered_ciphertext_b64) is None

    # Tamper with tag
    if len(tampered_b64_bytes) > 0: # Ensure there's a tag to tamper
        tampered_tag_list = list(tampered_b64_bytes)
        tampered_tag_list[-1] = tampered_tag_list[-1] ^ 1 # Flip a bit in tag
        tampered_tag_b64 = base64.urlsafe_b64encode(bytes(tampered_tag_list)).decode()
        assert service_with_b64_key.decrypt(tampered_tag_b64) is None


def test_encrypt_decrypt_empty_string(service_with_b64_key: EncryptionService):
    encrypted = service_with_b64_key.encrypt("")
    assert encrypted is not None
    decrypted = service_with_b64_key.decrypt(encrypted)
    assert decrypted == ""

def test_encrypt_non_string_input(service_with_b64_key: EncryptionService):
    assert service_with_b64_key.encrypt(123) is None # type: ignore

def test_decrypt_non_string_input(service_with_b64_key: EncryptionService):
    assert service_with_b64_key.decrypt(123) is None # type: ignore

def test_decrypt_empty_string_input(service_with_b64_key: EncryptionService):
    assert service_with_b64_key.decrypt("") is None

def test_init_key_handling():
    # 1. Valid base64-encoded 32-byte key passed directly
    service1 = EncryptionService(encryption_key=TEST_B64_KEY)
    assert len(service1.key) == 32
    # Decrypt something encrypted with a known key if we had an example ciphertext
    # For now, just check key length and successful init.

    # 2. Non-base64 string key passed directly (dev placeholder style) -> SHA256 derivation
    service2 = EncryptionService(encryption_key=DEV_PLACEHOLDER_KEY)
    assert len(service2.key) == 32
    # Key should not be the placeholder string itself after processing
    assert service2.key != DEV_PLACEHOLDER_KEY.encode('utf-8')

    # 3. Base64-encoded key but NOT 32 bytes -> Should use SHA256 derivation of the original b64 string
    short_b64_key_data = base64.urlsafe_b64encode(os.urandom(16)).decode() # Encodes 16 bytes
    service3 = EncryptionService(encryption_key=short_b64_key_data)
    assert len(service3.key) == 32
    # Check it derived from the original b64 string, not its decoded (short) form
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(short_b64_key_data.encode('utf-8'))
    expected_key_from_short_b64 = digest.finalize()
    assert service3.key == expected_key_from_short_b64

    # 4. Settings provide a non-base64 string (dev placeholder) -> SHA256 derivation
    with patch('claims_processor.src.core.security.encryption_service.get_settings') as mock_get_settings:
        mock_get_settings.return_value = Settings(APP_ENCRYPTION_KEY=DEV_PLACEHOLDER_KEY)
        service4 = EncryptionService() # Fetches from mocked settings
        assert len(service4.key) == 32
        digest_settings = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest_settings.update(DEV_PLACEHOLDER_KEY.encode('utf-8'))
        expected_key_from_settings_dev_key = digest_settings.finalize()
        assert service4.key == expected_key_from_settings_dev_key

    # 5. Settings provide a valid base64 encoded 32-byte key
    with patch('claims_processor.src.core.security.encryption_service.get_settings') as mock_get_settings:
        mock_get_settings.return_value = Settings(APP_ENCRYPTION_KEY=TEST_B64_KEY)
        service5 = EncryptionService() # Fetches from mocked settings
        assert len(service5.key) == 32
        assert service5.key == base64.urlsafe_b64decode(TEST_B64_KEY)

    # 6. Test with a key that is not typical b64 length (no padding) but decodes to 32 bytes
    # Example: Generate 32 random bytes, b64 encode, remove padding if any.
    raw_32_bytes = os.urandom(32)
    b64_unpadded_key_str = base64.urlsafe_b64encode(raw_32_bytes).decode('utf-8').rstrip('=')
    # Check if this unpadded string is different from padded, if so, test it.
    # If raw_32_bytes happens to not need padding, this test is less distinct but still valid.
    service6 = EncryptionService(encryption_key=b64_unpadded_key_str)
    assert len(service6.key) == 32
    # It should try b64 decode. If that fails (e.g. due to no padding if strict), it will SHA256 the string.
    # The current init logic's length check for b64 (len==44 and endswith("=")) might miss this.
    # Let's trace: if length is not 44 or no padding, it goes to SHA256.
    # Expected: SHA256 of b64_unpadded_key_str
    digest_unpadded = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest_unpadded.update(b64_unpadded_key_str.encode('utf-8'))
    expected_key_from_unpadded_b64_str = digest_unpadded.finalize()
    assert service6.key == expected_key_from_unpadded_b64_str
```
