from app.memory.base_memory import FileMemory


class ProceduralMemory(FileMemory):
    def save_pattern(self, task_type: str, pattern: dict):
        existing = self.load(task_type) or []
        if isinstance(existing, list):
            existing.append(pattern)
        else:
            existing = [pattern]
        self.save(task_type, existing)

    def find_pattern(self, task_type: str) -> list[dict]:
        data = self.load(task_type)
        if isinstance(data, list):
            return data
        return []

    def get_similar_pattern(self, objective: str) -> dict | None:
        data = self.all_data
        objective_lower = objective.lower()
        best_match = None
        best_score = 0
        for task_type, patterns in data.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    desc = pattern.get("objective", "")
                    score = len(set(objective_lower.split()) & set(desc.lower().split()))
                    if score > best_score:
                        best_score = score
                        best_match = pattern
        return best_match
