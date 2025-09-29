"""Base Response."""

from fastapi.responses import JSONResponse, ORJSONResponse
from starlette import status


class BaseResponse:
    """Base Response."""

    @staticmethod
    def success_response(
            message: str = "API success",
            status_code: int = status.HTTP_200_OK,
            data=None,
    ):
        """Success response with message and status code.

        Args:
            message: API Message.
            status_code: API status code.
            message_code: API message code.
            data: Response data.

        Returns:
            Success response.
        """
        if data is None:
            return JSONResponse(
                status_code=status_code,
                content={
                    "message": message,
                },
            )
        
        return ORJSONResponse(
            status_code=status_code,
            content={
                "message": message,
                "data": data,
            },
        )

    @staticmethod
    def error_response(message: str = "API error", status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        """Error response with message and status code.

        Args:
            message: API Message.
            status_code: API status code.
            message_code: API message code.

        Returns:
            Error response.
        """
        return JSONResponse(
            status_code=status_code,
            content={
                "message": message,
            }
        )
