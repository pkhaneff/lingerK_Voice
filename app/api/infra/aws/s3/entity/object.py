from dataclasses import dataclass, fields
from datetime import datetime
from typing import List

from botocore.response import StreamingBody

from app.api.utils.string import underscore


@dataclass
class S3Object:
    body: StreamingBody
    content_length: int
    content_type: str
    last_modified: datetime
    key: str

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            key = underscore(k)
            if key in names:
                setattr(self, key, v)


@dataclass
class S3ListObjectContents:
    key: str
    last_modified: datetime
    size: int

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            key = underscore(k)
            if key in names:
                setattr(self, key, v)


@dataclass
class S3ListObject:
    contents: List[S3ListObjectContents]
    name: str
    prefix: str
    max_keys: int
    delimiter: str
    key_count: int

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            key = underscore(k)
            if key in names:
                if key == "contents":
                    self.contents = [S3ListObjectContents(**c) for c in v]
                else:
                    setattr(self, key, v)
