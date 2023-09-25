from typing import Optional, Tuple

from pydantic import BaseModel

DATASET_PREDEFINED_SUB_SET_NAMES = ("mismatched", "matched")


class DatasetInfo(BaseModel):
    dataset_name: str
    configuration_name: Optional[str] = None
    set_name: str

    def get_dataset_info(self) -> "DatasetInfo":
        return DatasetInfo(
            dataset_name=self.dataset_name,
            configuration_name=self.configuration_name,
            set_name=self.set_name,
        )

    def get_name(self) -> str:
        if self.configuration_name is None:
            return self.dataset_name

        return f"{self.dataset_name}-{self.configuration_name}"

    def get_name_with_sub_split_name(self) -> str:
        name = self.get_name()

        for name_to_check in DATASET_PREDEFINED_SUB_SET_NAMES:
            if name_to_check in self.set_name:
                name += "-" + name_to_check
                break

        return name

    def get_data_set_names(self) -> Tuple[str, ...]:
        if self.configuration_name is None:
            return self.dataset_name, self.set_name

        return self.dataset_name, self.configuration_name, self.set_name
