from app.memory.base_memory import FileMemory


class SemanticMemory(FileMemory):
    def get_user_info(self, user_id: str = "default") -> dict:
        data = self.load(user_id) or {}
        if isinstance(data, dict):
            return data
        return {}

    def update_user_info(self, key: str, value: str, user_id: str = "default"):
        data = self.get_user_info(user_id)
        data[key] = value
        self.save(user_id, data)

    def get_preferences(self, user_id: str = "default") -> dict:
        info = self.get_user_info(user_id)
        return info.get("preferences", {})

    def set_preference(self, key: str, value: str, user_id: str = "default"):
        info = self.get_user_info(user_id)
        if "preferences" not in info:
            info["preferences"] = {}
        info["preferences"][key] = value
        self.save(user_id, info)
