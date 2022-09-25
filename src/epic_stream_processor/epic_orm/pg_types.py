from typing import Union, Optional
from typing import Callable
from xml.etree.ElementTree import Element
import sqlalchemy
import xml.etree.ElementTree as etree
from sqlalchemy.types import TypeEngine


class XMLType(sqlalchemy.types.UserDefinedType):  # type: ignore [type-arg]
    def get_col_spec(self) -> str:
        return "XML"

    def bind_processor(  # type: ignore [no-untyped-def]
        self, dialect
    ) -> Callable[..., Optional[Union[str, bytes]]]:
        def process(
            value: Optional[Union[str, Element]]
        ) -> Optional[Union[str, bytes]]:
            if value is not None:
                if isinstance(value, str):
                    return value
                else:
                    return etree.tostring(value)
            else:
                return None

        return process

    def result_processor( # type: ignore [no-untyped-def]
        self, dialect, coltype: str 
    ) -> Callable[..., Element]:
        def process(value: Union[str, bytes]) -> Element:
            if value is not None:
                value_el = etree.fromstring(value)
            return value_el

        return process
