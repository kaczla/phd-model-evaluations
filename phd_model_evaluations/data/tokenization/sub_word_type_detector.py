from typing import Any, Dict, Optional

from pydantic import BaseModel, root_validator


class SubWordTypeDetector(BaseModel):
    # Beginning and ending string identifying whole words
    whole_word_begging: Optional[str] = None
    whole_word_ending: Optional[str] = None
    # Beginning and ending string identifying inner words
    inner_word_begging: Optional[str] = None
    inner_word_ending: Optional[str] = None

    # Flags to help identifying whole/inner words
    whole_word_empty: bool = False
    inner_word_empty: bool = False
    both_empty: bool = False

    @root_validator
    def check_flags(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set flags base on the initialization data.

        Args:
            values: initialization data

        Returns:
            initialization data with flags set

        """
        whole_word_empty = values.get("whole_word_begging") is None and values.get("whole_word_ending") is None
        inner_word_empty = values.get("inner_word_begging") is None and values.get("inner_word_ending") is None

        values["whole_word_empty"] = whole_word_empty
        values["inner_word_empty"] = inner_word_empty
        values["both_empty"] = whole_word_empty and inner_word_empty
        return values

    def is_inner_word(self, token: str) -> bool:
        """
        Check given token is inner word.

        If all string are empty in the class, it will return `True`.
        If strings for detecting inner words missing, then will be used whole word method
        for detecting inner word - result will be negation of whole word method.

        Args:
            token: token to check

        Returns:
            status of inner word, `True` - is inner token, `False` - is not

        """
        if self.both_empty:
            return True

        if self.inner_word_empty:
            return not self.is_whole_word(token)

        return (self.inner_word_begging is not None and token.startswith(self.inner_word_begging)) or (
            self.inner_word_ending is not None and token.endswith(self.inner_word_ending)
        )

    def is_whole_word(self, token: str) -> bool:
        """
        Check given token is whole word.

        If all string are empty in the class, it will return `True`.
        If strings for detecting whole words missing, then will be used inner word method
        for detecting whole word - result will be negation of inner word method.

        Args:
            token: token to check

        Returns:
            status of whole word, `True` - is whole token, `False` - is not

        """
        if self.both_empty:
            return True

        if self.whole_word_empty:
            return not self.is_inner_word(token)

        return (self.whole_word_begging is not None and token.startswith(self.whole_word_begging)) or (
            self.whole_word_ending is not None and token.endswith(self.whole_word_ending)
        )
