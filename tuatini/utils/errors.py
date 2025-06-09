"""Contains all custom errors."""

from pathlib import Path
from typing import Optional, Union

import packaging.version
from requests import HTTPError, Response

V2_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v2.0 which is not backward compatible with v1.x.
Please, use our conversion script. Modify the following command with your own task description:
```
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \\
    --repo-id {repo_id} \\
    --single-task "TASK DESCRIPTION."  # <---- /!\\ Replace TASK DESCRIPTION /!\\
```

A few examples to replace TASK DESCRIPTION: "Pick up the blue cube and place it into the bin.", "Insert the
peg into the socket.", "Slide open the ziploc bag.", "Take the elevator to the 1st floor.", "Open the top
cabinet, store the pot inside it then close the cabinet.", "Push the T-shaped block onto the T-shaped
target.", "Grab the spray paint on the shelf and place it in the bin on top of the robot dog.", "Fold the
sweatshirt.", ...

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

V21_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.
While current version of LeRobot is backward-compatible with it, the version of your dataset still uses global
stats instead of per-episode stats. Update your dataset stats to the new format using this command:
```
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py --repo-id={repo_id}
```

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = V2_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


class CacheNotFound(Exception):
    """Exception thrown when the Huggingface cache is not found."""

    cache_dir: Union[str, Path]

    def __init__(self, msg: str, cache_dir: Union[str, Path], *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
        self.cache_dir = cache_dir


class CorruptedCacheException(Exception):
    """Exception for any unexpected structure in the Huggingface cache-system."""


# HEADERS ERRORS


class LocalTokenNotFoundError(EnvironmentError):
    """Raised if local token is required but not found."""


# HTTP ERRORS


class OfflineModeIsEnabled(ConnectionError):
    """Raised when a request is made but `HF_HUB_OFFLINE=1` is set as environment variable."""


class HfHubHTTPError(HTTPError):
    """
    HTTPError to inherit from for any custom HTTP Error raised in HF Hub.

    Any HTTPError is converted at least into a `HfHubHTTPError`. If some information is
    sent back by the server, it will be added to the error message.

    Added details:
    - Request id from "X-Request-Id" header if exists. If not, fallback to "X-Amzn-Trace-Id" header if exists.
    - Server error message from the header "X-Error-Message".
    - Server error message if we can found one in the response body.

    Example:
    ```py
        import requests
        from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

        response = get_session().post(...)
        try:
            hf_raise_for_status(response)
        except HfHubHTTPError as e:
            print(str(e)) # formatted message
            e.request_id, e.server_message # details returned by server

            # Complete the error message with additional information once it's raised
            e.append_to_message("\n`create_commit` expects the repository to exist.")
            raise
    ```
    """

    def __init__(self, message: str, response: Optional[Response] = None, *, server_message: Optional[str] = None):
        self.request_id = (
            response.headers.get("x-request-id") or response.headers.get("X-Amzn-Trace-Id")
            if response is not None
            else None
        )
        self.server_message = server_message

        super().__init__(
            message,
            response=response,  # type: ignore [arg-type]
            request=response.request if response is not None else None,  # type: ignore [arg-type]
        )

    def append_to_message(self, additional_message: str) -> None:
        """Append additional information to the `HfHubHTTPError` initial message."""
        self.args = (self.args[0] + additional_message,) + self.args[1:]


# INFERENCE CLIENT ERRORS


class InferenceTimeoutError(HTTPError, TimeoutError):
    """Error raised when a model is unavailable or the request times out."""


# INFERENCE ENDPOINT ERRORS


class InferenceEndpointError(Exception):
    """Generic exception when dealing with Inference Endpoints."""


class InferenceEndpointTimeoutError(InferenceEndpointError, TimeoutError):
    """Exception for timeouts while waiting for Inference Endpoint."""


# SAFETENSORS ERRORS


class SafetensorsParsingError(Exception):
    """Raised when failing to parse a safetensors file metadata.

    This can be the case if the file is not a safetensors file or does not respect the specification.
    """


class NotASafetensorsRepoError(Exception):
    """Raised when a repo is not a Safetensors repo i.e. doesn't have either a `model.safetensors` or a
    `model.safetensors.index.json` file.
    """


# TEXT GENERATION ERRORS


class TextGenerationError(HTTPError):
    """Generic error raised if text-generation went wrong."""


# Text Generation Inference Errors
class ValidationError(TextGenerationError):
    """Server-side validation error."""


class GenerationError(TextGenerationError):
    pass


class OverloadedError(TextGenerationError):
    pass


class IncompleteGenerationError(TextGenerationError):
    pass


class UnknownError(TextGenerationError):
    pass


# VALIDATION ERRORS


class HFValidationError(ValueError):
    """Generic exception thrown by `huggingface_hub` validators.

    Inherits from [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError).
    """


# FILE METADATA ERRORS


class FileMetadataError(OSError):
    """Error triggered when the metadata of a file on the Hub cannot be retrieved (missing ETag or commit_hash).

    Inherits from `OSError` for backward compatibility.
    """


# REPOSITORY ERRORS


class RepositoryNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with an invalid repository name, or
    with a private repo name the user does not have access to.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<non_existent_repository>")
    (...)
    huggingface_hub.utils._errors.RepositoryNotFoundError: 401 Client Error. (Request ID: PvMw_VjBMjVdMz53WKIzP)

    Repository Not Found for url: https://huggingface.co/api/models/%3Cnon_existent_repository%3E.
    Please make sure you specified the correct `repo_id` and `repo_type`.
    If the repo is private, make sure you are authenticated.
    Invalid username or password.
    ```
    """


class GatedRepoError(RepositoryNotFoundError):
    """
    Raised when trying to access a gated repository for which the user is not on the
    authorized list.

    Note: derives from `RepositoryNotFoundError` to ensure backward compatibility.

    Example:

    ```py
    >>> from huggingface_hub import model_info
    >>> model_info("<gated_repository>")
    (...)
    huggingface_hub.utils._errors.GatedRepoError: 403 Client Error. (Request ID: ViT1Bf7O_026LGSQuVqfa)

    Cannot access gated repo for url https://huggingface.co/api/models/ardent-figment/gated-model.
    Access to model ardent-figment/gated-model is restricted and you are not in the authorized list.
    Visit https://huggingface.co/ardent-figment/gated-model to ask for access.
    ```
    """


class DisabledRepoError(HfHubHTTPError):
    """
    Raised when trying to access a repository that has been disabled by its author.

    Example:

    ```py
    >>> from huggingface_hub import dataset_info
    >>> dataset_info("laion/laion-art")
    (...)
    huggingface_hub.utils._errors.DisabledRepoError: 403 Client Error. (Request ID: Root=1-659fc3fa-3031673e0f92c71a2260dbe2;bc6f4dfb-b30a-4862-af0a-5cfe827610d8)

    Cannot access repository for url https://huggingface.co/api/datasets/laion/laion-art.
    Access to this resource is disabled.
    ```
    """


# REVISION ERROR


class RevisionNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository but an invalid
    revision.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', 'config.json', revision='<non-existent-revision>')
    (...)
    huggingface_hub.utils._errors.RevisionNotFoundError: 404 Client Error. (Request ID: Mwhe_c3Kt650GcdKEFomX)

    Revision Not Found for url: https://huggingface.co/bert-base-cased/resolve/%3Cnon-existent-revision%3E/config.json.
    ```
    """


# ENTRY ERRORS
class EntryNotFoundError(HfHubHTTPError):
    """
    Raised when trying to access a hf.co URL with a valid repository and revision
    but an invalid filename.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-existent-file>')
    (...)
    huggingface_hub.utils._errors.EntryNotFoundError: 404 Client Error. (Request ID: 53pNl6M0MxsnG5Sw8JA6x)

    Entry Not Found for url: https://huggingface.co/bert-base-cased/resolve/main/%3Cnon-existent-file%3E.
    ```
    """


class LocalEntryNotFoundError(EntryNotFoundError, FileNotFoundError, ValueError):
    """
    Raised when trying to access a file or snapshot that is not on the disk when network is
    disabled or unavailable (connection issue). The entry may exist on the Hub.

    Note: `ValueError` type is to ensure backward compatibility.
    Note: `LocalEntryNotFoundError` derives from `HTTPError` because of `EntryNotFoundError`
          even when it is not a network issue.

    Example:

    ```py
    >>> from huggingface_hub import hf_hub_download
    >>> hf_hub_download('bert-base-cased', '<non-cached-file>',  local_files_only=True)
    (...)
    huggingface_hub.utils._errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.
    ```
    """

    def __init__(self, message: str):
        super().__init__(message, response=None)


# REQUEST ERROR
class BadRequestError(HfHubHTTPError, ValueError):
    """
    Raised by `hf_raise_for_status` when the server returns a HTTP 400 error.

    Example:

    ```py
    >>> resp = requests.post("hf.co/api/check", ...)
    >>> hf_raise_for_status(resp, endpoint_name="check")
    huggingface_hub.utils._errors.BadRequestError: Bad request for check endpoint: {details} (Request ID: XXX)
    ```
    """


# DDUF file format ERROR


class DDUFError(Exception):
    """Base exception for errors related to the DDUF format."""


class DDUFCorruptedFileError(DDUFError):
    """Exception thrown when the DDUF file is corrupted."""


class DDUFExportError(DDUFError):
    """Base exception for errors during DDUF export."""


class DDUFInvalidEntryNameError(DDUFExportError):
    """Exception thrown when the entry name is invalid."""


# STRICT DATACLASSES ERRORS


class StrictDataclassError(Exception):
    """Base exception for strict dataclasses."""


class StrictDataclassDefinitionError(StrictDataclassError):
    """Exception thrown when a strict dataclass is defined incorrectly."""


class StrictDataclassFieldValidationError(StrictDataclassError):
    """Exception thrown when a strict dataclass fails validation for a given field."""

    def __init__(self, field: str, cause: Exception):
        error_message = f"Validation error for field '{field}':"
        error_message += f"\n    {cause.__class__.__name__}: {cause}"
        super().__init__(error_message)


class StrictDataclassClassValidationError(StrictDataclassError):
    """Exception thrown when a strict dataclass fails validation on a class validator."""

    def __init__(self, validator: str, cause: Exception):
        error_message = f"Class validation error for validator '{validator}':"
        error_message += f"\n    {cause.__class__.__name__}: {cause}"
        super().__init__(error_message)


# XET ERRORS


class XetError(Exception):
    """Base exception for errors related to Xet Storage."""


class XetAuthorizationError(XetError):
    """Exception thrown when the user does not have the right authorization to use Xet Storage."""


class XetRefreshTokenError(XetError):
    """Exception thrown when the refresh token is invalid."""


class XetDownloadError(Exception):
    """Exception thrown when the download from Xet Storage fails."""


class DeviceAlreadyConnectedError(ConnectionError):
    """Exception raised when the device is already connected."""

    def __init__(
        self,
        message="This device is already connected. Try not calling `connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


class DeviceNotCalibratedError(ConnectionError):
    """Exception raised when the device is not calibrated."""

    def __init__(
        self,
        message="This device is not calibrated. Try calling `device.run_calibration()` first.",
    ):
        self.message = message
        super().__init__(self.message)


class DeviceNotConnectedError(ConnectionError):
    """Exception raised when the device is not connected."""

    def __init__(self, message="This device is not connected. Try calling `connect()` first."):
        self.message = message
        super().__init__(self.message)
