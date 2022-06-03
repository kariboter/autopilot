from typing import Union
import json


class GetFreeId:

    def __call__(self, *args, **kwargs):
        return self.get_all_data()

    def read_from_json(self):
        with open("app/test.json", "r") as file:
            data = json.load(file)
            return data

    def get_json_length(self):
        return len(self.read_from_json())

    def get_all_data(self):
        for item in range(self.get_json_length()):
            self.pk = self.get_free_id(self.read_from_json()[f'usr_{item + 1}'])
            self._set_busy_data(f'usr_{item + 1}')
            if self.pk is not None:
                return self.pk
        return None

    def _set_busy_data(self, pk):
        data = self.read_from_json()
        data[pk]['free'] = '0'
        self.write_in_json(data)

    @staticmethod
    def get_free_id(item: dict) -> Union[str, None]:
        if item['free'] == '1':
            return item['pk']
        return None

    @staticmethod
    def write_in_json(data):
        with open("app/test.json", "w") as file:
            json.dump(data, file)


class SetFreeId:

    def __init__(self, pk):
        self.pk = pk

    def read_from_json(self):
        with open("app/test.json", "r") as file:
            data = json.load(file)
            file.close()
            return data

    def set_free_id(self):
        data = self.read_from_json()
        data[f"{self.pk}"]['free'] = '1'
        self.write_in_json(data)

    @staticmethod
    def write_in_json(data):
        with open("app/test.json", "w") as file:
            json.dump(data, file)


# SetFreeId('usr_1').set_free_id()
